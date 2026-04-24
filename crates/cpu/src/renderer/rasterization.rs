//! This module provides functions for rasterizing triangles onto a 2D projection plane.
//! It handles triangle visibility, barycentric interpolation, texture sampling, and
//! per-pixel lighting calculations, supporting various mesh modes for flexible rendering.

use crate::renderer::buffers::{ImageBuffers, PointCloudBuffers};
use crate::renderer::lighting::calculate_lighting;
use crate::renderer::sampling::sample_texture_color_bilinear;
use mesh_projector_core::config::PointCloudConfig;
use mesh_projector_core::config::{ProjectionModel, RenderPassConfig};
use mesh_projector_core::constants::{
    DEFAULT_MISSING_TEXTURE_COLOR, TRIANGLE_AREA_DEGENERACY_THRESHOLD,
};
use mesh_projector_core::geometry::{
    barycentric, calculate_bounding_box, calculate_triangle_area, is_inside_triangle,
    is_triangle_visible,
};
use mesh_projector_core::types::color::Color;
use mesh_projector_core::types::coordinate::Coordinate;
use mesh_projector_core::types::mesh::{Mesh, Scene, ShadingMode};

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

struct DepthNormalization {
    far: f32,
    near: f32,
    range: f32,
}

impl DepthNormalization {
    fn new(far: f32, near: f32) -> Self {
        let range = (far - near).abs().max(1e-6);
        Self { far, near, range }
    }

    #[inline]
    fn to_byte(&self, depth_value: f32) -> u8 {
        let near = self.near.min(self.far);
        let far = self.near.max(self.far);
        let depth_clamped = depth_value.clamp(near, far);
        let depth_val_norm = ((far - depth_clamped) / self.range).clamp(0.0, 1.0);
        (depth_val_norm * 255.0).round() as u8
    }
}

struct ProjectedTriangle {
    screen_points: [(f32, f32); 3],
    depth_values: [f32; 3],
    inv_depths: Option<[f32; 3]>,
    depth_norm: DepthNormalization,
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

fn project_triangle_to_screen(
    triangle_surface: &TriangleSurface,
    render_pass_config: &RenderPassConfig,
) -> Option<ProjectedTriangle> {
    let view_config = &render_pass_config.view;
    let view_mat = &view_config.transformation_matrix;
    let u_axis_ws = Coordinate::new(view_mat[[0, 0]], view_mat[[1, 0]], view_mat[[2, 0]]);
    let v_axis_ws = Coordinate::new(view_mat[[0, 1]], view_mat[[1, 1]], view_mat[[2, 1]]);
    let vd = view_config.view_direction;
    let cube_center = (view_config.max_bound_f32 - 1.0) / 2.0;
    let cube_center_coordinate = Coordinate::new(cube_center, cube_center, cube_center);
    match &view_config.projection {
        ProjectionModel::Orthographic => {
            if !is_triangle_visible(triangle_surface.face_normal_ws, vd) {
                return None;
            }
            let offset_u_on_plane = cube_center_coordinate.dot(&u_axis_ws) - cube_center;
            let offset_v_on_plane = cube_center_coordinate.dot(&v_axis_ws) - cube_center;
            let vertices_proj: [Coordinate; 3] = triangle_surface.vertices_ws.map(|vertex_ws| {
                let proj_x = vertex_ws.dot(&u_axis_ws) - offset_u_on_plane;
                let proj_y = vertex_ws.dot(&v_axis_ws) - offset_v_on_plane;
                let z = vertex_ws.dot(&vd);
                Coordinate::new(proj_x, proj_y, z)
            });
            let max_depth_along_view =
                view_config.max_bound_f32 * (vd.x().max(0.0) + vd.y().max(0.0) + vd.z().max(0.0));
            let min_depth_along_view =
                view_config.max_bound_f32 * (vd.x().min(0.0) + vd.y().min(0.0) + vd.z().min(0.0));
            Some(ProjectedTriangle {
                screen_points: [
                    (vertices_proj[0].x(), vertices_proj[0].y()),
                    (vertices_proj[1].x(), vertices_proj[1].y()),
                    (vertices_proj[2].x(), vertices_proj[2].y()),
                ],
                depth_values: [
                    vertices_proj[0].z(),
                    vertices_proj[1].z(),
                    vertices_proj[2].z(),
                ],
                inv_depths: None,
                depth_norm: DepthNormalization::new(max_depth_along_view, min_depth_along_view),
            })
        }
        ProjectionModel::Perspective(params) => {
            let centroid_ws = (triangle_surface.vertices_ws[0]
                + triangle_surface.vertices_ws[1]
                + triangle_surface.vertices_ws[2])
                * (1.0 / 3.0);
            let view_vec = centroid_ws - params.camera_position_ws;
            if triangle_surface.face_normal_ws.dot(&view_vec) >= 0.0 {
                return None;
            }
            let mut screen_pts: [(f32, f32); 3] = [(0.0, 0.0); 3];
            let mut z_cam_vals: [f32; 3] = [0.0; 3];
            for (i, vertex_ws) in triangle_surface.vertices_ws.iter().enumerate() {
                let to_vertex = *vertex_ws - params.camera_position_ws;
                let x_cam = to_vertex.dot(&u_axis_ws);
                let y_cam = to_vertex.dot(&v_axis_ws);
                let z_cam = to_vertex.dot(&vd);
                if z_cam <= 1e-6 {
                    return None;
                }
                let scale = params.focal_length_px / z_cam;
                let col = scale * x_cam + cube_center;
                let row = scale * y_cam + cube_center;
                screen_pts[i] = (col, row);
                z_cam_vals[i] = z_cam;
            }
            let inv_depths = [
                1.0 / z_cam_vals[0],
                1.0 / z_cam_vals[1],
                1.0 / z_cam_vals[2],
            ];
            Some(ProjectedTriangle {
                screen_points: screen_pts,
                depth_values: z_cam_vals,
                inv_depths: Some(inv_depths),
                depth_norm: DepthNormalization::new(params.z_far, params.z_near),
            })
        }
    }
}

#[inline]
fn compute_depth_and_weights(
    barycentric_coords: (f32, f32, f32),
    projected: &ProjectedTriangle,
) -> (f32, (f32, f32, f32)) {
    let (w0, w1, w2) = barycentric_coords;
    if let Some(inv_depths) = projected.inv_depths {
        let denom = w0 * inv_depths[0] + w1 * inv_depths[1] + w2 * inv_depths[2];
        let iw0 = (w0 * inv_depths[0]) / denom;
        let iw1 = (w1 * inv_depths[1]) / denom;
        let iw2 = (w2 * inv_depths[2]) / denom;
        let depth_value = 1.0 / denom;
        (depth_value, (iw0, iw1, iw2))
    } else {
        let depth_value = w0 * projected.depth_values[0]
            + w1 * projected.depth_values[1]
            + w2 * projected.depth_values[2];
        (depth_value, barycentric_coords)
    }
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
    render_pass_config: &RenderPassConfig,
    buffers: &mut ImageBuffers,
    submesh_index: usize,
    triangle_index: usize, // Local triangle index for geometry
    pixel_offset: (f32, f32),
) {
    let submesh = &mesh.meshes[submesh_index];
    let triangle_surface = match extract_triangle_surface(mesh, submesh_index, triangle_index) {
        Ok(surface) => surface,
        Err(()) => return,
    };
    let Some(projected_triangle) =
        project_triangle_to_screen(&triangle_surface, render_pass_config)
    else {
        return;
    };
    let [p0_proj, p1_proj, p2_proj] = projected_triangle.screen_points;
    let z_buffer_shape = buffers.z_buffer.shape();
    let (rows, cols) = (z_buffer_shape[0], z_buffer_shape[1]);
    let Some((min_row, max_row, min_col, max_col)) =
        calculate_bounding_box(p0_proj, p1_proj, p2_proj, rows, cols)
    else {
        return;
    };
    let alpha_u8 = (submesh.material.opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
    for row in min_row..max_row {
        for col in min_col..max_col {
            let pixel_center = (
                col as f32 + 0.5 + pixel_offset.0,
                row as f32 + 0.5 + pixel_offset.1,
            );
            let Some((w0, w1, w2)) = barycentric(p0_proj, p1_proj, p2_proj, pixel_center) else {
                continue;
            };
            if !is_inside_triangle(w0, w1, w2) {
                continue;
            }
            let (depth_value, interpolated_weights) =
                compute_depth_and_weights((w0, w1, w2), &projected_triangle);
            let current_z = buffers.z_buffer[[row, col]];
            if !(depth_value < current_z) {
                continue;
            }
            let interpolated_point = interpolate_triangle_surface(
                &triangle_surface,
                interpolated_weights,
                &mesh,
                &submesh,
            );
            let final_color_linear =
                calculate_lighting(&interpolated_point, &render_pass_config, &submesh.material);
            let final_color_srgb = final_color_linear.to_srgb8();
            buffers.z_buffer[[row, col]] = depth_value;
            buffers.image[[row, col, 0]] = final_color_srgb[0];
            buffers.image[[row, col, 1]] = final_color_srgb[1];
            buffers.image[[row, col, 2]] = final_color_srgb[2];
            buffers.image[[row, col, 3]] = alpha_u8;
            buffers.occupancy_map[[row, col]] = 255;
            let depth_byte = projected_triangle.depth_norm.to_byte(depth_value);
            buffers.depth_map[[row, col]] = depth_byte;
            let interpolated_normal_vs = interpolated_point
                .normal_ws
                .to_view_space_as_vector(&render_pass_config.view.transformation_matrix);
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
            normal_map[[row, col, 3]] = 255;
        }
    }
}

#[inline]
fn process_point(
    barycentric_coords: (f32, f32, f32),
    triangle_surface: &TriangleSurface,
    mesh: &Scene,
    render_pass_config: &RenderPassConfig,
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
    let final_color =
        calculate_lighting(&interpolated_point, &render_pass_config, &submesh.material).to_srgb();
    // Add point and color to the flatten arrays
    buffers.add_point(interpolated_point.position_ws, final_color);
}

pub fn rasterize_triangle_to_point_cloud(
    mesh: &Scene,
    settings: &PointCloudConfig,
    render_pass_config: &RenderPassConfig,
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
    //      n = (-3 ± sqrt(1 + 8s)) / 2
    //
    // We take the positive root and round up:
    //      n = ceil((-3 + sqrt(1 + 8s)) / 2)
    // Compute subdivision level where n is a triangular number
    let n = {
        let s = triangle_area * settings.point_cloud_density_factor;
        let n = ((-3.0 + (1.0 + 8.0 * s).sqrt()) * 0.5).ceil();
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
                &render_pass_config,
                &submesh,
                buffers,
            );
        }
    }
}
