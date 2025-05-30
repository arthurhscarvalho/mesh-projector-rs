// src/projector/rasterization.rs
use crate::projector::config::ViewConfig;
use crate::projector::constants::DEFAULT_MISSING_TEXTURE_COLOR;
use crate::projector::geometry::{
    barycentric, calculate_bounding_box, is_inside_triangle, is_triangle_visible,
};
use crate::projector::lighting::calculate_lighting;
use crate::projector::math::normalize_vec3;
use crate::projector::mesh::{MeshGeometry, MeshMaterial, MeshMode, MeshTexture};
use crate::projector::sampling::sample_texture_color_bilinear;
use nd::{Array1, ArrayView1, ArrayViewMut2, ArrayViewMut3};
use std::f64;

/// Rasterizes a single triangle based on the projection mesh.mode.
pub fn rasterize_triangle(
    submesh_geometry: &MeshGeometry, // Geometry of the current submesh
    submesh_material: &MeshMaterial, // Material of the current submesh
    shared_texture: &MeshTexture,    // Shared texture data for the whole mesh
    mesh_mode: MeshMode,             // Overall rendering mode for the mesh
    local_triangle_index: usize, // Index of the current triangle within the submesh (for geometry)
    uv_triangle_id: usize, // Global triangle ID for UV lookup from shared_texture.triangle_uvs
    current_face_normal_ws: ArrayView1<f64>, // Precomputed world-space face normal for this triangle
    projection_config: &ViewConfig,          // Configuration for the current projection view
    z_slice: &mut ArrayViewMut2<f64>,        // Mutable view to the Z-buffer slice
    image_slice: &mut ArrayViewMut3<u8>,     // Mutable view to the color image buffer slice
    normal_map_slice: &mut ArrayViewMut3<u8>, // Mutable view to the normal map slice
    depth_map_slice_param: &mut ArrayViewMut3<u8>, // Renamed to avoid conflict with a local var
) {
    // World-space face normal for the current triangle (already precomputed)
    let face_normal_ws_arr = [
        current_face_normal_ws[0],
        current_face_normal_ws[1],
        current_face_normal_ws[2],
    ];

    // Back-face Culling (using world-space face normal and world-space view direction)
    if !is_triangle_visible(
        face_normal_ws_arr[0],
        face_normal_ws_arr[1],
        face_normal_ws_arr[2],
        projection_config.view_dir, // This is the world-space normal of the projection plane
    ) {
        return;
    }

    // Define projection plane axes from the rotation matrix
    let rot_mat = &projection_config.rotation_matrix;
    let u_axis_ws = Array1::from_vec(vec![rot_mat[[0, 0]], rot_mat[[1, 0]], rot_mat[[2, 0]]]);
    let v_axis_ws = Array1::from_vec(vec![rot_mat[[0, 1]], rot_mat[[1, 1]], rot_mat[[2, 1]]]);
    let plane_normal_ws = Array1::from_vec(vec![
        projection_config.view_dir.0,
        projection_config.view_dir.1,
        projection_config.view_dir.2,
    ]);

    let cube_center_coord = (projection_config.max_bound_f64 - 1.0) / 2.0;
    let c_cube_vec = Array1::from_vec(vec![
        cube_center_coord,
        cube_center_coord,
        cube_center_coord,
    ]);
    let offset_u_on_plane = c_cube_vec.dot(&u_axis_ws) - cube_center_coord;
    let offset_v_on_plane = c_cube_vec.dot(&v_axis_ws) - cube_center_coord;

    // Get Vertex Data Indices & Original (world-space, pre-scaled) Positions from submesh_geometry
    let world_vertices = submesh_geometry.vertices.view();
    let triangles = submesh_geometry.triangles.view();
    let tri_indices = triangles.row(local_triangle_index);
    let i0 = tri_indices[0];
    let i1 = tri_indices[1];
    let i2 = tri_indices[2];

    let p0_ws_vec = world_vertices.row(i0);
    let p1_ws_vec = world_vertices.row(i1);
    let p2_ws_vec = world_vertices.row(i2);

    // Convert ArrayView1 to [f64; 3] for functions expecting it and for dot products
    let p0_ws_arr_nd = Array1::from_vec(vec![p0_ws_vec[0], p0_ws_vec[1], p0_ws_vec[2]]);
    let p1_ws_arr_nd = Array1::from_vec(vec![p1_ws_vec[0], p1_ws_vec[1], p1_ws_vec[2]]);
    let p2_ws_arr_nd = Array1::from_vec(vec![p2_ws_vec[0], p2_ws_vec[1], p2_ws_vec[2]]);

    let p0_ws_arr = [p0_ws_vec[0], p0_ws_vec[1], p0_ws_vec[2]];
    let p1_ws_arr = [p1_ws_vec[0], p1_ws_vec[1], p1_ws_vec[2]];
    let p2_ws_arr = [p2_ws_vec[0], p2_ws_vec[1], p2_ws_vec[2]];

    // Project world-space vertices onto the view plane
    let p0_proj_x = p0_ws_arr_nd.dot(&u_axis_ws) - offset_u_on_plane;
    let p0_proj_y = p0_ws_arr_nd.dot(&v_axis_ws) - offset_v_on_plane;
    let z0 = p0_ws_arr_nd.dot(&plane_normal_ws);

    let p1_proj_x = p1_ws_arr_nd.dot(&u_axis_ws) - offset_u_on_plane;
    let p1_proj_y = p1_ws_arr_nd.dot(&v_axis_ws) - offset_v_on_plane;
    let z1 = p1_ws_arr_nd.dot(&plane_normal_ws);

    let p2_proj_x = p2_ws_arr_nd.dot(&u_axis_ws) - offset_u_on_plane;
    let p2_proj_y = p2_ws_arr_nd.dot(&v_axis_ws) - offset_v_on_plane;
    let z2 = p2_ws_arr_nd.dot(&plane_normal_ws);

    let p0_proj = (p0_proj_x, p0_proj_y);
    let p1_proj = (p1_proj_x, p1_proj_y);
    let p2_proj = (p2_proj_x, p2_proj_y);

    // Setup Data for Interpolation
    let mut uv0 = (0.0, 0.0);
    let mut uv1 = (0.0, 0.0);
    let mut uv2 = (0.0, 0.0);
    let mut c0 = (1.0, 1.0, 1.0);
    let mut c1 = (1.0, 1.0, 1.0);
    let mut c2 = (1.0, 1.0, 1.0);

    let mut n0_ws_arr = face_normal_ws_arr; // Default to face normal
    let mut n1_ws_arr = face_normal_ws_arr;
    let mut n2_ws_arr = face_normal_ws_arr;

    if let Some(v_normals_ws) = &submesh_geometry.vertex_normals {
        let v_normals_ws_view = v_normals_ws.view();
        let n0_v_ws = v_normals_ws_view.row(i0);
        let n1_v_ws = v_normals_ws_view.row(i1);
        let n2_v_ws = v_normals_ws_view.row(i2);
        // Store world-space normals for later interpolation
        n0_ws_arr = [n0_v_ws[0], n0_v_ws[1], n0_v_ws[2]];
        n1_ws_arr = [n1_v_ws[0], n1_v_ws[1], n1_v_ws[2]];
        n2_ws_arr = [n2_v_ws[0], n2_v_ws[1], n2_v_ws[2]];
    }

    if mesh_mode == MeshMode::Full || mesh_mode == MeshMode::ObjTexture {
        if let Some(uvs) = &shared_texture.triangle_uvs {
            // Use uv_triangle_id for indexing into the global shared_texture.triangle_uvs
            // Each entry in triangle_uvs is (U,V) for one corner. So, 3 entries per triangle.
            let uv_idx_base = uv_triangle_id * 3;
            if uv_idx_base + 2 < uvs.shape()[0] {
                let uv0_data = uvs.row(uv_idx_base);
                let uv1_data = uvs.row(uv_idx_base + 1);
                let uv2_data = uvs.row(uv_idx_base + 2);
                uv0 = (uv0_data[0], uv0_data[1]);
                uv1 = (uv1_data[0], uv1_data[1]);
                uv2 = (uv2_data[0], uv2_data[1]);
            } else {
                uv0 = (0.0, 0.0);
                uv1 = (0.0, 0.0);
                uv2 = (0.0, 0.0);
                log::warn!("UV lookup out of bounds. Global Triangle ID: {}, UV Idx Base: {}, UVs shape: {:?}. Using default UVs.",
                            uv_triangle_id, uv_idx_base, uvs.shape());
            }
        } else {
            uv0 = (0.0, 0.0);
            uv1 = (0.0, 0.0);
            uv2 = (0.0, 0.0);
        }
    }

    // Setup vertex colors if available (primarily for ObjOnly mesh_mode)
    if mesh_mode == MeshMode::ObjOnly {
        if let Some(v_colors) = &submesh_geometry.vertex_colors {
            // Ensure we have colors for all vertices indexed by this triangle
            if i0 < v_colors.nrows() && i1 < v_colors.nrows() && i2 < v_colors.nrows() {
                let v_colors_view = v_colors.view();
                let c0v = v_colors_view.row(i0);
                let c1v = v_colors_view.row(i1);
                let c2v = v_colors_view.row(i2);
                // Assume vertex colors are sRGB [0,1], store linear for interpolation
                // If they are already linear, powf(1.0) does nothing.
                // Typically, OBJ vertex colors are not specified as linear or sRGB, assume sRGB for safety.
                c0 = (c0v[0].powf(2.2), c0v[1].powf(2.2), c0v[2].powf(2.2));
                c1 = (c1v[0].powf(2.2), c1v[1].powf(2.2), c1v[2].powf(2.2));
                c2 = (c2v[0].powf(2.2), c2v[1].powf(2.2), c2v[2].powf(2.2));
            } else {
                log::warn!("Vertex color index out of bounds for submesh. Using default white.");
                // Default white (1,1,1) is already set
            }
        } else {
            // No vertex colors provided for this submesh, default white is used.
            // log::debug!("No vertex colors for submesh in ObjOnly mode, using default white.");
        }
    }

    let (rows, cols) = (z_slice.shape()[0], z_slice.shape()[1]);
    let Some((min_row, max_row, min_col, max_col)) =
        calculate_bounding_box(p0_proj, p1_proj, p2_proj, rows, cols)
    else {
        return;
    };

    for row in min_row..max_row {
        for col in min_col..max_col {
            let pixel_center = (col as f64 + 0.5, row as f64 + 0.5);
            let Some((w0, w1, w2)) = barycentric(p0_proj, p1_proj, p2_proj, pixel_center) else {
                continue;
            };
            if !is_inside_triangle(w0, w1, w2) {
                continue;
            }
            let interpolated_z = w0 * z0 + w1 * z1 + w2 * z2;
            let current_z = z_slice[[row, col]];
            let depth_test_passed = if projection_config.is_front_view {
                interpolated_z < current_z
            } else {
                interpolated_z > current_z
            };

            if depth_test_passed {
                // Interpolate world-space vertex position for the current pixel
                let interpolated_vx_ws = w0 * p0_ws_arr[0] + w1 * p1_ws_arr[0] + w2 * p2_ws_arr[0];
                let interpolated_vy_ws = w0 * p0_ws_arr[1] + w1 * p1_ws_arr[1] + w2 * p2_ws_arr[1];
                let interpolated_vz_ws = w0 * p0_ws_arr[2] + w1 * p1_ws_arr[2] + w2 * p2_ws_arr[2];
                let interpolated_vertex_pos_ws =
                    [interpolated_vx_ws, interpolated_vy_ws, interpolated_vz_ws];

                // Interpolate world-space normal for the current pixel
                // nX_ws_arr is either per-vertex normal or face normal based on `has_normals` logic path above
                let interpolated_nx_ws = w0 * n0_ws_arr[0] + w1 * n1_ws_arr[0] + w2 * n2_ws_arr[0];
                let interpolated_ny_ws = w0 * n0_ws_arr[1] + w1 * n1_ws_arr[1] + w2 * n2_ws_arr[1];
                let interpolated_nz_ws = w0 * n0_ws_arr[2] + w1 * n1_ws_arr[2] + w2 * n2_ws_arr[2];
                let mut interpolated_normal_ws =
                    [interpolated_nx_ws, interpolated_ny_ws, interpolated_nz_ws];
                normalize_vec3(&mut interpolated_normal_ws);

                let (interpolated_u, interpolated_v) =
                    if mesh_mode == MeshMode::Full || mesh_mode == MeshMode::ObjTexture {
                        (
                            w0 * uv0.0 + w1 * uv1.0 + w2 * uv2.0,
                            w0 * uv0.1 + w1 * uv1.1 + w2 * uv2.1,
                        )
                    } else {
                        (0.0, 0.0)
                    };

                let linear_base_color = match mesh_mode {
                    MeshMode::ObjOnly => {
                        // Interpolate vertex colors (c0, c1, c2 are default white for now)
                        let r = w0 * c0.0 + w1 * c1.0 + w2 * c2.0;
                        let g = w0 * c0.1 + w1 * c1.1 + w2 * c2.1;
                        let b = w0 * c0.2 + w1 * c1.2 + w2 * c2.2;
                        [r.max(0.0), g.max(0.0), b.max(0.0)]
                    }
                    MeshMode::ObjTexture | MeshMode::Full => {
                        if let Some(tex_image) = &shared_texture.texture {
                            // Use shared_texture for texture image
                            let sampled_srgb = sample_texture_color_bilinear(
                                interpolated_u,
                                interpolated_v,
                                tex_image.view(),
                                shared_texture.channels,
                                shared_texture.height_f64,
                                shared_texture.width_f64,
                            );
                            [
                                sampled_srgb[0].powf(2.2),
                                sampled_srgb[1].powf(2.2),
                                sampled_srgb[2].powf(2.2),
                            ]
                        } else {
                            DEFAULT_MISSING_TEXTURE_COLOR
                        }
                    }
                };

                // For orthographic projection, the camera is on the projection plane
                // Get the view direction from the config
                let view_dir_ws = [
                    projection_config.view_dir.0,
                    projection_config.view_dir.1,
                    projection_config.view_dir.2,
                ];

                let final_color_linear = calculate_lighting(
                    linear_base_color,
                    interpolated_vertex_pos_ws,
                    interpolated_normal_ws,
                    &projection_config.view_matrix,
                    view_dir_ws,
                    submesh_material,
                    mesh_mode,
                    &projection_config.light_source,
                );

                let final_color_srgb = [
                    (final_color_linear[0].powf(1.0 / 2.2) * 255.0)
                        .round()
                        .max(0.0)
                        .min(255.0) as u8,
                    (final_color_linear[1].powf(1.0 / 2.2) * 255.0)
                        .round()
                        .max(0.0)
                        .min(255.0) as u8,
                    (final_color_linear[2].powf(1.0 / 2.2) * 255.0)
                        .round()
                        .max(0.0)
                        .min(255.0) as u8,
                ];

                z_slice[[row, col]] = interpolated_z;
                image_slice[[row, col, 0]] = final_color_srgb[0];
                image_slice[[row, col, 1]] = final_color_srgb[1];
                image_slice[[row, col, 2]] = final_color_srgb[2];

                // Depth map (scaled to 0-255)
                let depth_val_norm = (interpolated_z / projection_config.max_bound_f64)
                    .max(0.0)
                    .min(1.0);
                let depth_byte = (depth_val_norm * 255.0).round() as u8;
                depth_map_slice_param[[row, col, 0]] = depth_byte; // Use renamed param
                depth_map_slice_param[[row, col, 1]] = depth_byte; // Use renamed param
                depth_map_slice_param[[row, col, 2]] = depth_byte; // Use renamed param

                // Normal map (view-space normals, scaled to 0-255)
                let interpolated_normal_vs_for_map = crate::projector::math::transform_vector_3d(
                    interpolated_normal_ws,
                    &projection_config.view_matrix,
                );

                normal_map_slice[[row, col, 0]] =
                    ((interpolated_normal_vs_for_map[0] * 0.5 + 0.5) * 255.0).round() as u8;
                normal_map_slice[[row, col, 1]] =
                    ((interpolated_normal_vs_for_map[1] * 0.5 + 0.5) * 255.0).round() as u8;
                normal_map_slice[[row, col, 2]] =
                    ((interpolated_normal_vs_for_map[2] * 0.5 + 0.5) * 255.0).round() as u8;
            }
        }
    }
}
