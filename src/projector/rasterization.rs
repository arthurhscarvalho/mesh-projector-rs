//! This module provides functions for rasterizing triangles onto a 2D projection plane.
//! It handles triangle visibility, barycentric interpolation, texture sampling, and
//! per-pixel lighting calculations, supporting various mesh modes for flexible rendering.

use crate::projector::buffers::Buffers;
use crate::projector::config::ViewConfig;
use crate::projector::constants::DEFAULT_MISSING_TEXTURE_COLOR;
use crate::projector::geometry::{
    barycentric, calculate_bounding_box, is_inside_triangle, is_triangle_visible,
};
use crate::projector::lighting::calculate_lighting;
use crate::projector::sampling::sample_texture_color_bilinear;
use crate::types::color::Color;
use crate::types::coordinate::Coordinate;
use crate::types::mesh::{Mesh, MeshMode};

/// Stores the interpolated attributes at a point on a triangle surface.
///
/// This struct is used to hold the results of barycentric interpolation for a point
/// within a triangle, including its world-space position, normal, and base color.
/// These attributes are then used for lighting calculations and final color output.
pub struct TrianglePoint {
    /// The interpolated position of the point in world space.
    pub position_ws: Coordinate,
    /// The interpolated normal vector at the point in world space.
    pub normal_ws: Coordinate,
    /// The interpolated base color at the point (in linear color space).
    pub base_color: Color,
}

/// Stores the per-triangle data required for rasterization and attribute interpolation.
///
/// This struct contains all the geometric and attribute information for a single triangle,
/// including world-space vertex positions, normals, UV coordinates, per-vertex colors,
/// and the face normal. It is used as an input to attribute interpolation and lighting
/// calculations during rasterization.
pub struct TriangleSurface {
    /// The positions of the triangle's three vertices in world space.
    pub vertices_ws: [Coordinate; 3],
    /// The normals at each vertex in world space.
    pub normals_ws: [Coordinate; 3],
    /// The UV texture coordinates for each vertex, as (u, v) pairs.
    pub uvs: [(f64, f64); 3],
    /// The linear-space color at each vertex, used in ObjOnly mode.
    pub colors: [Color; 3],
    /// The face normal of the triangle in world space.
    pub face_normal_ws: Coordinate,
}

/// Extracts triangle data from mesh for processing
pub fn extract_triangle_data(
    mesh: &Mesh,
    submesh_index: usize,
    triangle_index: usize,
) -> TriangleSurface {
    // Get submesh
    let submesh = &mesh.submeshes[submesh_index];
    // Calculate global triangle offset for UV lookup
    let global_triangle_offset: usize = (0..submesh_index)
        .into_iter()
        .map(|i| mesh.submeshes[i].geometry.triangles.nrows())
        .sum();
    let uv_triangle_id: usize = global_triangle_offset + triangle_index;
    // Get face normal
    let current_face_normal_ws = submesh.geometry.face_normals.row(triangle_index);
    let face_normal_ws = Coordinate::new(
        current_face_normal_ws[0],
        current_face_normal_ws[1],
        current_face_normal_ws[2],
    );
    // Get vertex indices
    let world_vertices = submesh.geometry.vertices.view();
    let triangles = submesh.geometry.triangles.view();
    let tri_indices = triangles.row(triangle_index);
    let i0 = tri_indices[0];
    let i1 = tri_indices[1];
    let i2 = tri_indices[2];
    // Get world-space positions
    let p0_ws_vec = world_vertices.row(i0);
    let p1_ws_vec = world_vertices.row(i1);
    let p2_ws_vec = world_vertices.row(i2);
    let vertices_ws = [
        Coordinate::new(p0_ws_vec[0], p0_ws_vec[1], p0_ws_vec[2]),
        Coordinate::new(p1_ws_vec[0], p1_ws_vec[1], p1_ws_vec[2]),
        Coordinate::new(p2_ws_vec[0], p2_ws_vec[1], p2_ws_vec[2]),
    ];
    // Get normals (use vertex normals if available, otherwise face normal)
    let mut normals_ws = [face_normal_ws; 3];
    if let Some(v_normals_ws) = &submesh.geometry.vertex_normals {
        let v_normals_ws_view = v_normals_ws.view();
        let n0_v_ws = v_normals_ws_view.row(i0);
        let n1_v_ws = v_normals_ws_view.row(i1);
        let n2_v_ws = v_normals_ws_view.row(i2);
        normals_ws = [
            Coordinate::new(n0_v_ws[0], n0_v_ws[1], n0_v_ws[2]),
            Coordinate::new(n1_v_ws[0], n1_v_ws[1], n1_v_ws[2]),
            Coordinate::new(n2_v_ws[0], n2_v_ws[1], n2_v_ws[2]),
        ];
    }
    // Get UVs
    let mut uvs = [(0.0, 0.0); 3];
    if mesh.mode == MeshMode::Full || mesh.mode == MeshMode::ObjTexture {
        if let Some(texture_uvs) = &mesh.texture.triangle_uvs {
            let uv_idx_base = uv_triangle_id * 3;
            if uv_idx_base + 2 < texture_uvs.shape()[0] {
                let uv0_data = texture_uvs.row(uv_idx_base);
                let uv1_data = texture_uvs.row(uv_idx_base + 1);
                let uv2_data = texture_uvs.row(uv_idx_base + 2);
                uvs = [
                    (uv0_data[0], uv0_data[1]),
                    (uv1_data[0], uv1_data[1]),
                    (uv2_data[0], uv2_data[1]),
                ];
            }
        }
    }
    // Get vertex colors
    let mut colors = [Color::white(); 3];
    if mesh.mode == MeshMode::ObjOnly {
        if let Some(v_colors) = &submesh.geometry.vertex_colors {
            if i0 < v_colors.nrows() && i1 < v_colors.nrows() && i2 < v_colors.nrows() {
                let v_colors_view = v_colors.view();
                let c0v = v_colors_view.row(i0);
                let c1v = v_colors_view.row(i1);
                let c2v = v_colors_view.row(i2);
                colors = [
                    Color::from_srgb(c0v[0], c0v[1], c0v[2]),
                    Color::from_srgb(c1v[0], c1v[1], c1v[2]),
                    Color::from_srgb(c2v[0], c2v[1], c2v[2]),
                ];
            }
        }
    }
    // Return triangle data
    TriangleSurface {
        vertices_ws,
        normals_ws,
        uvs,
        colors,
        face_normal_ws,
    }
}

/// Interpolates attributes at a specific barycentric coordinate within a triangle
pub fn interpolate_triangle_attributes(
    triangle_data: &TriangleSurface,
    barycentric_coords: (f64, f64, f64), // (w0, w1, w2)
    mesh: &Mesh,
) -> TrianglePoint {
    let (w0, w1, w2) = barycentric_coords;
    // Interpolate position
    let position_ws = triangle_data.vertices_ws[0] * w0
        + triangle_data.vertices_ws[1] * w1
        + triangle_data.vertices_ws[2] * w2;
    // Interpolate normal and normalize
    let mut normal_ws = triangle_data.normals_ws[0] * w0
        + triangle_data.normals_ws[1] * w1
        + triangle_data.normals_ws[2] * w2;
    normal_ws.normalize();
    // Interpolate UV
    let uv = (
        w0 * triangle_data.uvs[0].0 + w1 * triangle_data.uvs[1].0 + w2 * triangle_data.uvs[2].0,
        w0 * triangle_data.uvs[0].1 + w1 * triangle_data.uvs[1].1 + w2 * triangle_data.uvs[2].1,
    );
    // Calculate base color based on mesh mode
    let base_color = match mesh.mode {
        MeshMode::ObjOnly => {
            // Interpolate vertex colors
            let interpolated_color = triangle_data.colors[0] * w0
                + triangle_data.colors[1] * w1
                + triangle_data.colors[2] * w2;
            Color::new(
                interpolated_color.r().max(0.0),
                interpolated_color.g().max(0.0),
                interpolated_color.b().max(0.0),
            )
        }
        MeshMode::ObjTexture | MeshMode::Full => {
            if let Some(tex_image) = &mesh.texture.texture {
                sample_texture_color_bilinear(
                    uv.0,
                    uv.1,
                    tex_image.view(),
                    mesh.texture.channels,
                    mesh.texture.height_f64,
                    mesh.texture.width_f64,
                )
            } else {
                DEFAULT_MISSING_TEXTURE_COLOR
            }
        }
    };
    // Return interpolated attributes
    TrianglePoint {
        position_ws,
        normal_ws,
        base_color,
    }
}

/// Rasterizes a single triangle based on the projection mesh.mode.
pub fn rasterize_triangle(
    mesh: &Mesh,
    view_config: &ViewConfig,
    buffers: &mut Buffers,
    submesh_index: usize,
    triangle_index: usize, // Local triangle index for geometry
) {
    // Get submesh
    let submesh = &mesh.submeshes[submesh_index];
    // Extract triangle data
    let triangle_data = extract_triangle_data(mesh, submesh_index, triangle_index);
    // Back-face Culling (using world-space face normal and world-space view direction)
    if !is_triangle_visible(triangle_data.face_normal_ws, view_config.view_direction) {
        return;
    }
    // Define projection plane axes from the rotation matrix
    let view_mat = &view_config.transformation_matrix;
    let u_axis_ws = Coordinate::new(view_mat[[0, 0]], view_mat[[1, 0]], view_mat[[2, 0]]);
    let v_axis_ws = Coordinate::new(view_mat[[0, 1]], view_mat[[1, 1]], view_mat[[2, 1]]);
    let plane_normal_ws = view_config.view_direction;
    // Calculate cube center coordinates
    let cube_center = (view_config.max_bound_f64 - 1.0) / 2.0;
    let cube_center_coordinate = Coordinate::new(cube_center, cube_center, cube_center);
    let offset_u_on_plane = cube_center_coordinate.dot(&u_axis_ws) - cube_center;
    let offset_v_on_plane = cube_center_coordinate.dot(&v_axis_ws) - cube_center;
    // Project world-space vertices onto the view plane
    let vertices_proj: [(f64, f64, f64); 3] = triangle_data
        .vertices_ws
        .iter()
        .map(|&vertex_ws| {
            let proj_x = vertex_ws.dot(&u_axis_ws) - offset_u_on_plane;
            let proj_y = vertex_ws.dot(&v_axis_ws) - offset_v_on_plane;
            let z = vertex_ws.dot(&plane_normal_ws);
            (proj_x, proj_y, z)
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    // Project vertices onto the view plane
    let p0_proj = (vertices_proj[0].0, vertices_proj[0].1);
    let p1_proj = (vertices_proj[1].0, vertices_proj[1].1);
    let p2_proj = (vertices_proj[2].0, vertices_proj[2].1);
    // Get the bounding box for the triangle
    let (rows, cols) = (buffers.z_buffer.shape()[0], buffers.z_buffer.shape()[1]);
    let Some((min_row, max_row, min_col, max_col)) =
        calculate_bounding_box(p0_proj, p1_proj, p2_proj, rows, cols)
    else {
        return;
    };
    // Rasterize the triangle
    for row in min_row..max_row {
        for col in min_col..max_col {
            let pixel_center = (col as f64 + 0.5, row as f64 + 0.5);
            let Some((w0, w1, w2)) = barycentric(p0_proj, p1_proj, p2_proj, pixel_center) else {
                continue;
            };
            if !is_inside_triangle(w0, w1, w2) {
                continue;
            }
            let interpolated_z =
                w0 * vertices_proj[0].2 + w1 * vertices_proj[1].2 + w2 * vertices_proj[2].2;
            let current_z = buffers.z_buffer[[row, col]];
            let depth_test_passed = interpolated_z < current_z;
            // If the depth test passed, interpolate attributes and update buffers
            if !depth_test_passed {
                continue;
            }
            // Interpolate attributes using the new shared function
            let attributes = interpolate_triangle_attributes(&triangle_data, (w0, w1, w2), mesh);
            // Calculate final color with lighting
            let final_color_linear =
                calculate_lighting(&attributes, &view_config, &submesh.material, &mesh.mode);
            // Convert the final color to sRGB
            let final_color_srgb = final_color_linear.to_srgb_denormalized();
            // Update buffers
            buffers.z_buffer[[row, col]] = interpolated_z;
            buffers.image[[row, col, 0]] = final_color_srgb.r() as u8;
            buffers.image[[row, col, 1]] = final_color_srgb.g() as u8;
            buffers.image[[row, col, 2]] = final_color_srgb.b() as u8;
            // Depth map (scaled to 0-255)
            let depth_val_norm = {
                let mut depth_val_norm = (interpolated_z / view_config.max_bound_f64)
                    .abs()
                    .clamp(0.0, 1.0);
                if interpolated_z > 0.0 {
                    depth_val_norm = 1.0 - depth_val_norm;
                }
                depth_val_norm
            };
            let depth_byte = (depth_val_norm * 255.0).round() as u8;
            buffers.depth_map[[row, col, 0]] = depth_byte;
            buffers.depth_map[[row, col, 1]] = depth_byte;
            buffers.depth_map[[row, col, 2]] = depth_byte;
            // View-space normals
            let interpolated_normal_vs = attributes
                .normal_ws
                .to_view_space_as_vector(&view_config.transformation_matrix);
            // Normalize the normal vector from [-1, 1] to the [0, 1] range
            let normals_normalized: Color = interpolated_normal_vs.map(|x| (x + 1.0) / 2.0).into();
            let normals_srgb: Color = normals_normalized.to_srgb_denormalized();
            // Update the normal map
            let normal_map = &mut buffers.normal_map;
            normal_map[[row, col, 0]] = normals_srgb.r() as u8;
            normal_map[[row, col, 1]] = normals_srgb.g() as u8;
            normal_map[[row, col, 2]] = normals_srgb.b() as u8;
        }
    }
}
