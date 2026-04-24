//! This module implements the core logic for projecting a 3D mesh onto a 2D image plane
//! according to a specified view configuration. It orchestrates the rasterization of mesh
//! triangles, manages per-pixel buffers (color, depth, normals, occupancy), and applies
//! post-processing such as outlier filtering.

use crate::renderer::buffers::ImageBuffers;
use crate::renderer::filtering::filter_outliers;
use crate::renderer::msaa::{sample_offsets, MsaaAccumulator};
use crate::renderer::rasterization::rasterize_triangle_to_images;
use crate::renderer::ssao::apply_ssao;
use mesh_projector_core::config::{CameraConfig, ProjectionConfig, ProjectionType};
use mesh_projector_core::config::{PerspectiveParameters, ProjectionModel, RenderPassConfig};
use mesh_projector_core::types::coordinate::Coordinate;
use mesh_projector_core::types::mesh::Scene;
use mesh_projector_core::types::mesh::ShadingMode;
use ndarray::{Array2, Array3};

/// Builds a projection model from camera configuration and view parameters.
///
/// For orthographic projection, returns a simple orthographic model.
/// For perspective projection, derives camera position, focal length, and clipping planes
/// to ensure the entire cube (defined by `max_bound_f32`) fits within the viewing frustum.
pub fn build_projection_model(
    camera_config: &CameraConfig,
    view_direction: Coordinate,
    max_bound_f32: f32,
) -> ProjectionModel {
    let cube_center = (max_bound_f32 - 1.0) / 2.0;
    let cube_center_coordinate = Coordinate::new(cube_center, cube_center, cube_center);
    match &camera_config.projection {
        ProjectionType::Orthographic => ProjectionModel::Orthographic,
        ProjectionType::Perspective(perspective_cfg) => {
            let half_fov_rad = 0.5 * perspective_cfg.vertical_fov_degrees.to_radians();
            let tan_half_fov = half_fov_rad.tan();
            let focal_length_px = if tan_half_fov.abs() < 1e-6 {
                max_bound_f32 * 0.5
            } else {
                (max_bound_f32 * 0.5) / tan_half_fov
            };
            let camera_position_ws =
                cube_center_coordinate - view_direction * (cube_center + focal_length_px);
            let z_near = focal_length_px;
            let z_far = focal_length_px + 2.0 * cube_center;
            ProjectionModel::Perspective(PerspectiveParameters {
                camera_position_ws,
                focal_length_px,
                z_near,
                z_far,
            })
        }
    }
}

/// Rasterizes all triangles from all submeshes into the provided buffers.
fn rasterize_all_triangles(
    mesh: &Scene,
    render_pass_config: &RenderPassConfig,
    buffers: &mut ImageBuffers,
    sample_offset: (f32, f32),
) {
    for submesh_index in 0..mesh.meshes.len() {
        let num_faces = mesh.meshes[submesh_index].geometry.triangles.nrows();
        for triangle_index in 0..num_faces {
            rasterize_triangle_to_images(
                mesh,
                render_pass_config,
                buffers,
                submesh_index,
                triangle_index,
                sample_offset,
            );
        }
    }
}

/// Renders without MSAA by rasterizing directly to final buffers.
fn render_without_msaa(
    mesh: &Scene,
    render_pass_config: &RenderPassConfig,
    rows: usize,
    cols: usize,
    background_color: [u8; 4],
) -> ImageBuffers {
    let mut buffers = ImageBuffers::new(rows, cols, background_color);
    rasterize_all_triangles(mesh, render_pass_config, &mut buffers, (0.0, 0.0));
    buffers
}

/// Renders with MSAA by accumulating multiple sub-pixel samples.
fn render_with_msaa(
    mesh: &Scene,
    render_pass_config: &RenderPassConfig,
    rows: usize,
    cols: usize,
    background_color: [u8; 4],
    sample_count: u8,
) -> ImageBuffers {
    let sample_offsets = sample_offsets(sample_count);
    let mut msaa_accum = MsaaAccumulator::new(rows, cols, background_color);
    for &sample_offset in &sample_offsets {
        let mut buffers = ImageBuffers::new(rows, cols, background_color);
        rasterize_all_triangles(mesh, render_pass_config, &mut buffers, sample_offset);
        msaa_accum.accumulate(&buffers);
    }
    msaa_accum.resolve(sample_count)
}

/// Computes a single orthographic projection for the given view configuration.
pub fn compute_projection(
    mesh: &Scene,
    settings: &ProjectionConfig,
    render_pass_config: &RenderPassConfig,
) -> (Array3<u8>, Array2<u8>, Array3<u8>, Array2<u8>, u64) {
    let max_bound: u16 = settings.resolution;
    let rows = max_bound as usize;
    let cols = max_bound as usize;
    let background_color = settings.background.to_srgba8();
    let sample_count = settings.msaa.sample_count.samples();
    let mut buffers = if sample_count == 1 {
        render_without_msaa(mesh, render_pass_config, rows, cols, background_color)
    } else {
        render_with_msaa(
            mesh,
            render_pass_config,
            rows,
            cols,
            background_color,
            sample_count,
        )
    };
    // Apply outlier filtering
    filter_outliers(&mut buffers, &settings, max_bound as f32);
    // Apply SSAO if enabled and compatible with the shading mode
    if render_pass_config.ssao.enabled {
        if mesh.mode == ShadingMode::BlinnPhong {
            apply_ssao(&mut buffers, render_pass_config);
        } else {
            log::warn!(
                "SSAO enabled but shading mode is {:?}. SSAO is only applied with Blinn-Phong lighting.",
                mesh.mode
            );
        }
    }
    // Flip buffers
    buffers.v_flip();
    buffers.h_flip();
    // Return from buffers
    (
        buffers.image,
        buffers.depth_map,
        buffers.normal_map,
        buffers.occupancy_map,
        buffers.outlier_count,
    )
}
