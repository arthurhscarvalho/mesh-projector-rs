//! This module provides functions for rasterizing triangles onto a 2D projection plane.
//! It handles triangle visibility, barycentric interpolation, texture sampling, and
//! per-pixel lighting calculations, supporting various mesh modes for flexible rendering.

use crate::projector::config::PointCloudConfig;
use crate::renderer::buffers::{ImageBuffers, PointCloudBuffers};
use crate::renderer::config::ViewConfig;
use crate::renderer::constants::{
    DEFAULT_MISSING_TEXTURE_COLOR, TRIANGLE_AREA_DEGENERACY_THRESHOLD,
};
use crate::renderer::geometry::{
    barycentric, calculate_bounding_box, calculate_triangle_area, is_inside_triangle,
    is_triangle_visible,
};
use crate::renderer::lighting::calculate_lighting;
use crate::renderer::sampling::sample_texture_color_bilinear;
use crate::types::color::Color;
use crate::types::coordinate::Coordinate;
use crate::types::mesh::{Mesh, ShadingMode, Scene};

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
    pub uvs: [(f32, f32); 3],
    /// The linear-space color at each vertex, used in VertexColor mode.
    pub colors: [Color; 3],
    /// The face normal of the triangle in world space.
    pub face_normal_ws: Coordinate,
}

#[inline]
pub fn extract_triangle_vertices_ws(submesh: &Mesh, triangle_index: usize) -> [Coordinate; 3] {
    let world_vertices = submesh.geometry.vertices.view();
    let triangles = submesh.geometry.triangles.view();
    // Get vertex indices
    let tri_indices = triangles.row(triangle_index);
    let i0 = &tri_indices[0];
    let i1 = &tri_indices[1];
    let i2 = &tri_indices[2];
    // Get world-space positions
    let p0_ws_vec = world_vertices.row(*i0);
    let p1_ws_vec = world_vertices.row(*i1);
    let p2_ws_vec = world_vertices.row(*i2);
    let vertices_ws = [
        Coordinate::new(p0_ws_vec[0], p0_ws_vec[1], p0_ws_vec[2]),
        Coordinate::new(p1_ws_vec[0], p1_ws_vec[1], p1_ws_vec[2]),
        Coordinate::new(p2_ws_vec[0], p2_ws_vec[1], p2_ws_vec[2]),
    ];
    vertices_ws
}

/// Extracts triangle data from mesh for processing
/// Returns an error if the triangle is degenerate (has zero or near-zero area)
pub fn extract_triangle_surface(
    mesh: &Scene,
    submesh_index: usize,
    triangle_index: usize,
) -> Result<TriangleSurface, ()> {
    // Get submesh
    let submesh = &mesh.meshes[submesh_index];
    // Local triangle index for UV lookup within submesh
    let uv_triangle_id: usize = triangle_index;
    // Get face normal
    let current_face_normal_ws = submesh.geometry.face_normals.row(triangle_index);
    let face_normal_ws = Coordinate::new(
        current_face_normal_ws[0],
        current_face_normal_ws[1],
        current_face_normal_ws[2],
    );
    // Get vertex indices
    let triangles = submesh.geometry.triangles.view();
    let tri_indices = triangles.row(triangle_index);
    let i0 = &tri_indices[0];
    let i1 = &tri_indices[1];
    let i2 = &tri_indices[2];
    // Get world-space positions
    let vertices_ws = extract_triangle_vertices_ws(submesh, triangle_index);
    // Check for degenerate triangle (zero or near-zero area)
    let triangle_area = calculate_triangle_area(&vertices_ws);
    if triangle_area < TRIANGLE_AREA_DEGENERACY_THRESHOLD {
        log::warn!(
            "Degenerate triangle found at submesh {}, triangle {}, skipping",
            submesh_index,
            triangle_index
        );
        return Err(());
    }
    // Get normals (use vertex normals if available, otherwise face normal)
    let mut normals_ws = [face_normal_ws; 3];
    if let Some(v_normals_ws) = &submesh.geometry.vertex_normals {
        let v_normals_ws_view = v_normals_ws.view();
        let n0_v_ws = v_normals_ws_view.row(*i0);
        let n1_v_ws = v_normals_ws_view.row(*i1);
        let n2_v_ws = v_normals_ws_view.row(*i2);
        normals_ws = [
            Coordinate::new(n0_v_ws[0], n0_v_ws[1], n0_v_ws[2]),
            Coordinate::new(n1_v_ws[0], n1_v_ws[1], n1_v_ws[2]),
            Coordinate::new(n2_v_ws[0], n2_v_ws[1], n2_v_ws[2]),
        ];
    }
    // Get UVs
    let mut uvs = [(0.0, 0.0); 3];
    if mesh.mode == ShadingMode::BlinnPhong || mesh.mode == ShadingMode::TextureMapped {
        if let Some(submesh_texture) = &submesh.texture {
            if let Some(texture_uvs) = &submesh_texture.triangle_uvs {
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
    }
    // Get vertex colors
    let mut colors = [Color::white(); 3];
    if mesh.mode == ShadingMode::VertexColor {
        if let Some(v_colors) = &submesh.geometry.vertex_colors {
            if *i0 < v_colors.nrows() && *i1 < v_colors.nrows() && *i2 < v_colors.nrows() {
                let v_colors_view = v_colors.view();
                let c0v = v_colors_view.row(*i0);
                let c1v = v_colors_view.row(*i1);
                let c2v = v_colors_view.row(*i2);
                colors = [
                    Color::from_srgb(c0v[0], c0v[1], c0v[2]),
                    Color::from_srgb(c1v[0], c1v[1], c1v[2]),
                    Color::from_srgb(c2v[0], c2v[1], c2v[2]),
                ];
            }
        }
    }
    // Return triangle data
    Ok(TriangleSurface {
        vertices_ws,
        normals_ws,
        uvs,
        colors,
        face_normal_ws,
    })
}

#[inline]
pub fn interpolate_coordinates(
    barycentric_coords: &(f32, f32, f32),
    coordinates: &[Coordinate; 3],
) -> Coordinate {
    let interpolated_coordinate = coordinates[0] * barycentric_coords.0
        + coordinates[1] * barycentric_coords.1
        + coordinates[2] * barycentric_coords.2;
    interpolated_coordinate
}

#[inline]
pub fn interpolate_uvs(barycentric_coords: &(f32, f32, f32), uvs: &[(f32, f32); 3]) -> (f32, f32) {
    let interpolated_uv = (
        uvs[0].0 * barycentric_coords.0
            + uvs[1].0 * barycentric_coords.1
            + uvs[2].0 * barycentric_coords.2,
        uvs[0].1 * barycentric_coords.0
            + uvs[1].1 * barycentric_coords.1
            + uvs[2].1 * barycentric_coords.2,
    );
    interpolated_uv
}

#[inline]
pub fn interpolate_colors(barycentric_coords: &(f32, f32, f32), colors: &[Color; 3]) -> Color {
    let interpolated_color = colors[0] * barycentric_coords.0
        + colors[1] * barycentric_coords.1
        + colors[2] * barycentric_coords.2;
    interpolated_color
}

/// Interpolates triangle surface at a specific barycentric coordinate within a triangle
/// returning a TrianglePoint struct with the interpolated triangle.
pub fn interpolate_triangle_surface(
    triangle_surface: &TriangleSurface,
    barycentric_coords: (f32, f32, f32), // (w0, w1, w2)
    mesh: &Scene,
    submesh: &Mesh,
) -> TrianglePoint {
    // Interpolate position
    let position_ws = interpolate_coordinates(&barycentric_coords, &triangle_surface.vertices_ws);
    // Interpolate normal and normalize
    let mut normal_ws = interpolate_coordinates(&barycentric_coords, &triangle_surface.normals_ws);
    normal_ws.normalize();
    // Interpolate UV
    let uv = interpolate_uvs(&barycentric_coords, &triangle_surface.uvs);
    // Calculate base color based on mesh mode
    let base_color = match mesh.mode {
        ShadingMode::VertexColor => {
            // Interpolate vertex colors
            let interpolated_color =
                interpolate_colors(&barycentric_coords, &triangle_surface.colors);
            Color::new(
                interpolated_color.r().max(0.0),
                interpolated_color.g().max(0.0),
                interpolated_color.b().max(0.0),
            )
        }
        ShadingMode::TextureMapped | ShadingMode::BlinnPhong => {
            if let Some(submesh_texture) = &submesh.texture {
                if let Some(tex_image) = &submesh_texture.texture {
                    sample_texture_color_bilinear(
                        uv.0,
                        uv.1,
                        tex_image.view(),
                        submesh_texture.channels,
                        submesh_texture.height_f32,
                        submesh_texture.width_f32,
                    )
                } else {
                    DEFAULT_MISSING_TEXTURE_COLOR
                }
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
pub fn rasterize_triangle_to_images(
    mesh: &Scene,
    view_config: &ViewConfig,
    buffers: &mut ImageBuffers,
    submesh_index: usize,
    triangle_index: usize, // Local triangle index for geometry
) {
    // Get submesh
    let submesh = &mesh.meshes[submesh_index];
    // Extract triangle data
    let triangle_surface = match extract_triangle_surface(mesh, submesh_index, triangle_index) {
        Ok(surface) => surface,
        Err(()) => return, // Skip degenerate triangles
    };
    // Back-face Culling (using world-space face normal and world-space view direction)
    if !is_triangle_visible(triangle_surface.face_normal_ws, view_config.view_direction) {
        return;
    }
    // Define projection plane axes from the rotation matrix
    let view_mat = &view_config.transformation_matrix;
    let u_axis_ws = Coordinate::new(view_mat[[0, 0]], view_mat[[1, 0]], view_mat[[2, 0]]);
    let v_axis_ws = Coordinate::new(view_mat[[0, 1]], view_mat[[1, 1]], view_mat[[2, 1]]);
    let plane_normal_ws = view_config.view_direction;
    // Calculate cube center coordinates
    let cube_center = (view_config.max_bound_f32 - 1.0) / 2.0;
    let cube_center_coordinate = Coordinate::new(cube_center, cube_center, cube_center);
    let offset_u_on_plane = cube_center_coordinate.dot(&u_axis_ws) - cube_center;
    let offset_v_on_plane = cube_center_coordinate.dot(&v_axis_ws) - cube_center;
    // Project world-space vertices onto the view plane
    let vertices_proj: [Coordinate; 3] = triangle_surface.vertices_ws.map(|vertex_ws| {
        let proj_x = vertex_ws.dot(&u_axis_ws) - offset_u_on_plane;
        let proj_y = vertex_ws.dot(&v_axis_ws) - offset_v_on_plane;
        let z = vertex_ws.dot(&plane_normal_ws);
        Coordinate::new(proj_x, proj_y, z)
    });
    // Project vertices onto the view plane
    let p0_proj = (vertices_proj[0].x(), vertices_proj[0].y());
    let p1_proj = (vertices_proj[1].x(), vertices_proj[1].y());
    let p2_proj = (vertices_proj[2].x(), vertices_proj[2].y());
    // Get the bounding box for the triangle
    let z_buffer_shape = buffers.z_buffer.shape();
    let (rows, cols) = (z_buffer_shape[0], z_buffer_shape[1]);
    let Some((min_row, max_row, min_col, max_col)) =
        calculate_bounding_box(p0_proj, p1_proj, p2_proj, rows, cols)
    else {
        return;
    };
    // Rasterize the triangle
    for row in min_row..max_row {
        for col in min_col..max_col {
            let pixel_center = (col as f32 + 0.5, row as f32 + 0.5);
            let Some((w0, w1, w2)) = barycentric(p0_proj, p1_proj, p2_proj, pixel_center) else {
                continue;
            };
            if !is_inside_triangle(w0, w1, w2) {
                continue;
            }
            let interpolated_z =
                w0 * vertices_proj[0].z() + w1 * vertices_proj[1].z() + w2 * vertices_proj[2].z();
            let current_z = buffers.z_buffer[[row, col]];
            let depth_test_passed = interpolated_z < current_z;
            // If the depth test passed, interpolate attributes and update buffers, else skip
            if !depth_test_passed {
                continue;
            }
            // Interpolate triangle
            let interpolated_point =
                interpolate_triangle_surface(&triangle_surface, (w0, w1, w2), &mesh, &submesh);
            // Calculate final color with lighting (linear space)
            let final_color_linear = calculate_lighting(
                &interpolated_point,
                &view_config,
                &submesh.material,
                &mesh.mode,
            );
            // Fast linear→sRGB conversion using LUT
            let final_color_srgb = final_color_linear.to_srgb8();
            // Update buffers
            buffers.z_buffer[[row, col]] = interpolated_z;
            buffers.image[[row, col, 0]] = final_color_srgb[0];
            buffers.image[[row, col, 1]] = final_color_srgb[1];
            buffers.image[[row, col, 2]] = final_color_srgb[2];
            buffers.occupancy_map[[row, col]] = 255;
            // Compute the signed depth range spanned by the bounding cube along the current
            // view direction.  For each axis we take either 0 or `max_bound_f32` depending
            // on the sign of the corresponding component of the view direction, thereby
            // obtaining the minimum and maximum possible dot-products without enumerating
            // all eight corners explicitly.  This gives the *exact* depth range even for
            // arbitrary (non-axis aligned) view directions.
            let vd = view_config.view_direction;
            let max_depth_along_view =
                view_config.max_bound_f32 * (vd.x().max(0.0) + vd.y().max(0.0) + vd.z().max(0.0));
            let min_depth_along_view =
                view_config.max_bound_f32 * (vd.x().min(0.0) + vd.y().min(0.0) + vd.z().min(0.0));
            let depth_range = max_depth_along_view - min_depth_along_view;
            // Normalise such that "near" (smaller z) becomes 1 and "far" (larger z) becomes
            // 0, independent of the sign of the view direction components.
            let depth_val_norm =
                ((max_depth_along_view - interpolated_z) / depth_range).clamp(0.0, 1.0);
            let depth_byte = (depth_val_norm * 255.0).round() as u8;
            buffers.depth_map[[row, col]] = depth_byte;
            // View-space normals
            let interpolated_normal_vs = interpolated_point
                .normal_ws
                .to_view_space_as_vector(&view_config.transformation_matrix);
            // Normalize the normal vector from [-1,1] to [0,1] and encode to 8-bit directly
            let nx = ((interpolated_normal_vs.x() + 1.0) * 0.5 * 255.0)
                .round()
                .clamp(0.0, 255.0);
            let ny = ((interpolated_normal_vs.y() + 1.0) * 0.5 * 255.0)
                .round()
                .clamp(0.0, 255.0);
            let nz = ((interpolated_normal_vs.z() + 1.0) * 0.5 * 255.0)
                .round()
                .clamp(0.0, 255.0);
            let normal_map = &mut buffers.normal_map;
            normal_map[[row, col, 0]] = nx as u8;
            normal_map[[row, col, 1]] = ny as u8;
            normal_map[[row, col, 2]] = nz as u8;
        }
    }
}

#[inline]
fn process_point(
    barycentric_coords: (f32, f32, f32),
    triangle_surface: &TriangleSurface,
    mesh: &Scene,
    view_config: &ViewConfig,
    submesh: &Mesh,
    buffers: &mut PointCloudBuffers,
) {
    if !is_inside_triangle(
        barycentric_coords.0,
        barycentric_coords.1,
        barycentric_coords.2,
    ) {
        return;
    }
    // Use the shared interpolation function
    let interpolated_point =
        interpolate_triangle_surface(&triangle_surface, barycentric_coords, &mesh, &submesh);
    // Calculate final color using the shared lighting function
    let final_color = calculate_lighting(
        &interpolated_point,
        &view_config,
        &submesh.material,
        &mesh.mode,
    )
    .to_srgb();
    // Add point and color to the flatten arrays
    buffers.add_point(interpolated_point.position_ws, final_color);
}

pub fn rasterize_triangle_to_point_cloud(
    mesh: &Scene,
    settings: &PointCloudConfig,
    view_config: &ViewConfig,
    buffers: &mut PointCloudBuffers,
    submesh_index: usize,
    triangle_index: usize, // Local triangle index for geometry
) {
    // Get submesh
    let submesh = &mesh.meshes[submesh_index];
    // Extract triangle data
    let triangle_surface = match extract_triangle_surface(&mesh, submesh_index, triangle_index) {
        Ok(surface) => surface,
        Err(()) => return, // Skip degenerate triangles
    };
    // Calculate triangle area
    let triangle_area = calculate_triangle_area(&triangle_surface.vertices_ws);
    // Triangular number formula (inclusive): T(n) = (n + 1) * (n + 2) / 2
    // We want at least s points, such that T(n) ≥ s,
    // where s is the target number of points.
    //
    // So:
    //      (n + 1)(n + 2) / 2 ≥ s
    //
    // Multiply both sides by 2:
    //      (n + 1)(n + 2) ≥ 2s
    //
    // Expand:
    //      n^2 + 3n + 2 ≥ 2s
    //
    // Solve the quadratic inequality:
    //      n^2 + 3n + (2 - 2s) ≥ 0
    //
    // Solve n from the quadratic equation:
    //      n = (-3 ± sqrt(9 + 8s)) / 2
    //
    // We take the positive root and round up:
    //      n = ceil((-3 + sqrt(9 + 8s)) / 2)
    // Compute subdivision level where n is a triangular number
    let n = {
        let s = (triangle_area * settings.point_cloud_density_factor).round();
        let n = ((-3.0 + (9.0 + 8.0 * s).sqrt()) * 0.5).ceil();
        n as usize
    };
    // Deterministic centroid sampling of subtriangles
    for i in 0..=n {
        for j in 0..=(n - i) {
            let k = n - i - j;
            // Centroid barycentric coordinates for this subtriangle
            let u = (i as f32 + 1.0 / 3.0) / (n as f32 + 1.0);
            let v = (j as f32 + 1.0 / 3.0) / (n as f32 + 1.0);
            let w = (k as f32 + 1.0 / 3.0) / (n as f32 + 1.0);
            process_point(
                (u, v, w),
                &triangle_surface,
                &mesh,
                &view_config,
                &submesh,
                buffers,
            );
        }
    }
}
