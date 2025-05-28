// src/projector/rasterization.rs
use crate::projector::config::ViewConfig;
use crate::projector::constants::{DEFAULT_LIGHT_COLOR, DEFAULT_MISSING_TEXTURE_COLOR};
use crate::projector::geometry::{
    barycentric, calculate_bounding_box, is_inside_triangle, is_triangle_visible,
};
use crate::projector::lighting::calculate_lighting;
use crate::projector::math::{normalize_vec3, rotate_vertex};
use crate::projector::mesh::{Mesh, MeshMode};
use crate::projector::sampling::sample_texture_color_bilinear;
use nd::{Array1, ArrayView1, ArrayViewMut2, ArrayViewMut3};
use std::f64;

/// Rasterizes a single triangle based on the projection mesh.mode.
pub fn rasterize_triangle(
    mesh: &Mesh, // Mesh with material properties (contains world-space, pre-scaled vertices)
    triangle_index: usize, // Index of the current triangle
    current_face_normal_ws: ArrayView1<f64>, // Precomputed world-space face normal for this triangle
    projection_config: &ViewConfig, // Configuration for the current projection view (contains rotation_matrix)
    z_slice: &mut ArrayViewMut2<f64>, // Mutable view to the Z-buffer slice
    image_slice: &mut ArrayViewMut3<u8>, // Mutable view to the color image buffer slice
    depth_map_slice: &mut ArrayViewMut3<u8>, // Mutable view to the depth map slice
    normal_map_slice: &mut ArrayViewMut3<u8>, // Mutable view to the normal map slice
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

    // Define projection plane axes from the rotation matrix (these are view-space axes in world coordinates)
    // u_axis_ws = rotation_matrix.column(0) (new X in world)
    // v_axis_ws = rotation_matrix.column(1) (new Y in world)
    // normal_ws = rotation_matrix.column(2) (new Z in world - same as projection_config.view_dir)
    let rot_mat = &projection_config.rotation_matrix;
    let u_axis_ws = Array1::from_vec(vec![rot_mat[[0, 0]], rot_mat[[1, 0]], rot_mat[[2, 0]]]);
    let v_axis_ws = Array1::from_vec(vec![rot_mat[[0, 1]], rot_mat[[1, 1]], rot_mat[[2, 1]]]);
    let plane_normal_ws = Array1::from_vec(vec![
        projection_config.view_dir.0,
        projection_config.view_dir.1,
        projection_config.view_dir.2,
    ]);

    // Calculate the scalar coordinate of the center of the target projection cube
    let cube_center_coord = (projection_config.max_bound_f64 - 1.0) / 2.0;
    let c_cube_vec = Array1::from_vec(vec![
        cube_center_coord,
        cube_center_coord,
        cube_center_coord,
    ]);

    // Calculate the offset needed to ensure C_cube projects to (cube_center_coord, cube_center_coord)
    let offset_u_on_plane = c_cube_vec.dot(&u_axis_ws) - cube_center_coord;
    let offset_v_on_plane = c_cube_vec.dot(&v_axis_ws) - cube_center_coord;
    // Depth offset is not applied to keep depth values consistent with current normalization
    // let offset_z_on_plane = c_cube_vec.dot(&plane_normal_ws);

    // Get Vertex Data Indices & Original (world-space, pre-scaled) Positions
    let world_vertices = mesh.geometry.vertices.view();
    let triangles = mesh.geometry.triangles.view();
    let tri_indices = triangles.row(triangle_index);
    let i0 = tri_indices[0];
    let i1 = tri_indices[1];
    let i2 = tri_indices[2];

    let p0_ws = world_vertices.row(i0);
    let p1_ws = world_vertices.row(i1);
    let p2_ws = world_vertices.row(i2);

    // Project world-space vertices onto the view plane, adjusting for the desired center
    let p0_proj_x = p0_ws.dot(&u_axis_ws) - offset_u_on_plane;
    let p0_proj_y = p0_ws.dot(&v_axis_ws) - offset_v_on_plane;
    let z0 = p0_ws.dot(&plane_normal_ws); // Depth calculation remains absolute

    let p1_proj_x = p1_ws.dot(&u_axis_ws) - offset_u_on_plane;
    let p1_proj_y = p1_ws.dot(&v_axis_ws) - offset_v_on_plane;
    let z1 = p1_ws.dot(&plane_normal_ws);

    let p2_proj_x = p2_ws.dot(&u_axis_ws) - offset_u_on_plane;
    let p2_proj_y = p2_ws.dot(&v_axis_ws) - offset_v_on_plane;
    let z2 = p2_ws.dot(&plane_normal_ws);

    // Projected 2D points
    let p0_proj = (p0_proj_x, p0_proj_y);
    let p1_proj = (p1_proj_x, p1_proj_y);
    let p2_proj = (p2_proj_x, p2_proj_y);

    // Setup Data for Interpolation
    let z0 = z0;
    let z1 = z1;
    let z2 = z2;
    // Store raw attributes for linear interpolation
    let mut n0 = (0.0, 0.0, 0.0);
    let mut n1 = (0.0, 0.0, 0.0);
    let mut n2 = (0.0, 0.0, 0.0);
    let mut uv0 = (0.0, 0.0);
    let mut uv1 = (0.0, 0.0);
    let mut uv2 = (0.0, 0.0);
    // Default vertex color is white (linear)
    let mut c0 = (1.0, 1.0, 1.0);
    let mut c1 = (1.0, 1.0, 1.0);
    let mut c2 = (1.0, 1.0, 1.0);
    let mut has_normals = false;
    // Setup normals if available
    if let Some(v_normals_ws) = &mesh.geometry.vertex_normals {
        // These are world-space normals
        let v_normals_ws_view = v_normals_ws.view();
        let n0_ws_vec = v_normals_ws_view.row(i0);
        let n1_ws_vec = v_normals_ws_view.row(i1);
        let n2_ws_vec = v_normals_ws_view.row(i2);

        // Rotate world-space vertex normals to view-space for lighting
        let n0_vs_arr = rotate_vertex(&n0_ws_vec, rot_mat);
        let n1_vs_arr = rotate_vertex(&n1_ws_vec, rot_mat);
        let n2_vs_arr = rotate_vertex(&n2_ws_vec, rot_mat);

        n0 = (n0_vs_arr[0], n0_vs_arr[1], n0_vs_arr[2]);
        n1 = (n1_vs_arr[0], n1_vs_arr[1], n1_vs_arr[2]);
        n2 = (n2_vs_arr[0], n2_vs_arr[1], n2_vs_arr[2]);
        has_normals = true;
    }
    // Setup UVs if available (required for texture modes)
    if mesh.mode == MeshMode::Full || mesh.mode == MeshMode::ObjTexture {
        if let Some(uvs) = &mesh.texture.triangle_uvs {
            let uvs_view = uvs.view();
            let uv_idx_base = triangle_index * 3;
            let uv0_data = uvs_view.row(uv_idx_base);
            let uv1_data = uvs_view.row(uv_idx_base + 1);
            let uv2_data = uvs_view.row(uv_idx_base + 2);
            // Store raw UVs
            uv0 = (uv0_data[0], uv0_data[1]);
            uv1 = (uv1_data[0], uv1_data[1]);
            uv2 = (uv2_data[0], uv2_data[1]);
        } else {
            // Default UVs are (0,0)
            uv0 = (0.0, 0.0);
            uv1 = (0.0, 0.0);
            uv2 = (0.0, 0.0);
        }
    }
    // Setup vertex colors if available (required for ObjOnly mesh.mode)
    if mesh.mode == MeshMode::ObjOnly {
        if let Some(v_colors) = &mesh.texture.vertex_colors {
            let v_colors_view = v_colors.view();
            let c0v = v_colors_view.row(i0);
            let c1v = v_colors_view.row(i1);
            let c2v = v_colors_view.row(i2);
            // Assume vertex colors are sRGB [0,1], store linear for interpolation
            c0 = (c0v[0].powf(2.2), c0v[1].powf(2.2), c0v[2].powf(2.2));
            c1 = (c1v[0].powf(2.2), c1v[1].powf(2.2), c1v[2].powf(2.2));
            c2 = (c2v[0].powf(2.2), c2v[1].powf(2.2), c2v[2].powf(2.2));
        }
        // If None, the default white setup (1,1,1) remains.
    }
    // Calculate Bounding Box
    let (rows, cols) = (z_slice.shape()[0], z_slice.shape()[1]);
    let Some((min_row, max_row, min_col, max_col)) =
        calculate_bounding_box(p0_proj, p1_proj, p2_proj, rows, cols)
    else {
        return;
    };
    // Pixel Loop
    for row in min_row..max_row {
        for col in min_col..max_col {
            let pixel_center = (col as f64 + 0.5, row as f64 + 0.5);
            // Barycentric Coordinates
            let Some((w0, w1, w2)) = barycentric(p0_proj, p1_proj, p2_proj, pixel_center) else {
                continue;
            };
            // Check if inside using Barycentric coordinates
            if !is_inside_triangle(w0, w1, w2) {
                continue;
            }
            // Interpolate depth linearly
            let interpolated_z = w0 * z0 + w1 * z1 + w2 * z2;
            // Depth Test
            let current_z = z_slice[[row, col]];
            let depth_test_passed = if projection_config.is_front_view {
                interpolated_z < current_z
            } else {
                interpolated_z > current_z
            };
            if depth_test_passed {
                // Interpolate Attributes
                let interpolated_normal_vs = if has_normals {
                    // These are now view-space normals
                    let bw0 = w0;
                    let bw1 = w1;
                    let bw2 = w2;
                    // Linear interpolation of view-space normals
                    let nx = bw0 * n0.0 + bw1 * n1.0 + bw2 * n2.0;
                    let ny = bw0 * n0.1 + bw1 * n1.1 + bw2 * n2.1;
                    let nz = bw0 * n0.2 + bw1 * n1.2 + bw2 * n2.2;
                    let mut norm_vs = [nx, ny, nz];
                    normalize_vec3(&mut norm_vs); // Normalize the interpolated view-space normal
                    norm_vs
                } else {
                    // Rotate the world-space face normal to view-space for lighting
                    let face_normal_ws_nd = Array1::from_vec(face_normal_ws_arr.to_vec());
                    let face_normal_vs_nd = rotate_vertex(&face_normal_ws_nd.view(), rot_mat);
                    let mut norm_vs = [
                        face_normal_vs_nd[0],
                        face_normal_vs_nd[1],
                        face_normal_vs_nd[2],
                    ];
                    normalize_vec3(&mut norm_vs); // Ensure it's normalized
                    norm_vs
                };
                let (interpolated_u, interpolated_v) =
                    if mesh.mode == MeshMode::Full || mesh.mode == MeshMode::ObjTexture {
                        // Linear interpolation of UVs
                        let u = w0 * uv0.0 + w1 * uv1.0 + w2 * uv2.0;
                        let v = w0 * uv0.1 + w1 * uv1.1 + w2 * uv2.1;
                        (u, v)
                    } else {
                        (0.0, 0.0)
                    };
                let linear_base_color = match mesh.mode {
                    MeshMode::ObjOnly => {
                        // Linear interpolation of vertex colors (already in linear space)
                        let r = w0 * c0.0 + w1 * c1.0 + w2 * c2.0;
                        let g = w0 * c0.1 + w1 * c1.1 + w2 * c2.1;
                        let b = w0 * c0.2 + w1 * c1.2 + w2 * c2.2;
                        [r.max(0.0), g.max(0.0), b.max(0.0)]
                    }
                    MeshMode::ObjTexture | MeshMode::Full => {
                        if let Some(tex) = &mesh.texture.texture {
                            let sampled_srgb = sample_texture_color_bilinear(
                                interpolated_u,
                                interpolated_v,
                                tex.view(),
                                mesh.texture.channels,
                                mesh.texture.height_f64,
                                mesh.texture.width_f64,
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
                // Calculate Final Color (Lighting)
                // For lighting, view direction in view-space is always (0,0,1) if using standard view transform.
                // Light direction also needs to be in view-space.
                // Assuming a directional light that is always opposite to the view direction *in world space*.
                // So, light_dir_ws = -projection_config.view_dir.
                // We need to transform this light_dir_ws to view-space.
                // view_matrix * light_dir_ws. If rotation_matrix is view_to_world, then its transpose (or inverse for ortho) is world_to_view.
                // Since rot_mat is world -> view orientation (cols are view axes in world),
                // to transform a world vector V_w to view vector V_v: V_v = rot_mat_transpose * V_w.
                // rot_mat_transpose is rot_mat.t() in ndarray.

                let view_dir_vs = [0.0, 0.0, 1.0]; // In view space, camera looks along positive Z.

                // Light direction in world space (opposite to camera's Z / view direction)
                let light_dir_ws_tuple = (
                    -projection_config.view_dir.0,
                    -projection_config.view_dir.1,
                    -projection_config.view_dir.2,
                );
                let light_dir_ws_nd = Array1::from_vec(vec![
                    light_dir_ws_tuple.0,
                    light_dir_ws_tuple.1,
                    light_dir_ws_tuple.2,
                ]);

                // Transform light direction from world to view space
                let rot_mat_transpose = rot_mat.t();
                let light_dir_vs_nd = rot_mat_transpose.dot(&light_dir_ws_nd);
                let mut light_dir_vs_arr =
                    [light_dir_vs_nd[0], light_dir_vs_nd[1], light_dir_vs_nd[2]];
                normalize_vec3(&mut light_dir_vs_arr);

                let light_color = DEFAULT_LIGHT_COLOR; // Assuming light color is constant

                let final_linear_color = calculate_lighting(
                    &mesh.material,
                    linear_base_color,
                    interpolated_normal_vs, // This is now in view-space
                    view_dir_vs,            // View direction in view-space
                    light_dir_vs_arr,       // Light direction in view-space
                    light_color,
                );
                // Convert linear to sRGB and quantize to [0, 255]
                let srgb_r = final_linear_color[0].powf(1.0 / 2.2);
                let srgb_g = final_linear_color[1].powf(1.0 / 2.2);
                let srgb_b = final_linear_color[2].powf(1.0 / 2.2);
                let r_byte = (srgb_r * 255.0).min(255.0).max(0.0) as u8;
                let g_byte = (srgb_g * 255.0).min(255.0).max(0.0) as u8;
                let b_byte = (srgb_b * 255.0).min(255.0).max(0.0) as u8;
                // Update Z-buffer and color buffer
                z_slice[[row, col]] = interpolated_z;
                image_slice[[row, col, 0]] = r_byte;
                image_slice[[row, col, 1]] = g_byte;
                image_slice[[row, col, 2]] = b_byte;

                // Populate Depth Map (assuming further = brighter/higher value)
                // Assumes interpolated_z (world coordinate along depth_axis) is normalized to be
                // within [0, max_bound_f64] due to preprocessing in io.rs.
                let max_bound = projection_config.max_bound_f64;
                let normalized_depth_factor = if projection_config.is_front_view {
                    // Front view: z=0 is closer (darker), z=max_bound is further (brighter).
                    // Maps [0, M] to [0, 1].
                    (1.0 - (interpolated_z / max_bound)).clamp(0.0, 1.0)
                } else {
                    // Back view: z=0 is further (brighter), z=max_bound is closer (darker).
                    // Maps [0, M] to [1, 0] (further is 1, closer is 0), effectively (M - z) / M.
                    (interpolated_z / max_bound).clamp(0.0, 1.0)
                };
                let depth_u8 = (normalized_depth_factor * 255.0).round() as u8;
                depth_map_slice[[row, col, 0]] = depth_u8;
                depth_map_slice[[row, col, 1]] = depth_u8;
                depth_map_slice[[row, col, 2]] = depth_u8;

                // Populate Normal Map (map [-1,1] to [0,255])
                // Normals should be saved in a consistent space. If they are view-space normals,
                // then the interpretation of the normal map needs to be consistent.
                // Typically, normal maps store object-space or tangent-space normals.
                // For simplicity here, we'll save the view-space normals.
                let normal_x_u8 = ((interpolated_normal_vs[0] + 1.0) / 2.0 * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                let normal_y_u8 = ((interpolated_normal_vs[1] + 1.0) / 2.0 * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                let normal_z_u8 = ((interpolated_normal_vs[2] + 1.0) / 2.0 * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                normal_map_slice[[row, col, 0]] = normal_x_u8;
                normal_map_slice[[row, col, 1]] = normal_y_u8;
                normal_map_slice[[row, col, 2]] = normal_z_u8;
            }
        }
    }
}
