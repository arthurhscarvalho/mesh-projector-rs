//! High-level module for 3D mesh projection and rendering.

pub mod config;
pub mod constants;
mod point_cloud;
mod projection;

use crate::projector::config::{PointCloudConfig, ProjectionConfig};
use crate::renderer::config::{LightingConfig, RenderPassConfig, ViewConfig};
use crate::renderer::geometry::create_view_matrix;
use crate::types::coordinate::Coordinate;
use crate::types::image_batch::ImageBatch;
use crate::types::mesh::Scene;
use constants::NUM_CHANNELS;
use ndarray::Array2;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;

/// Computes cube projections for the given mesh and settings.
///
/// Returns:
/// An ImageBatch containing all projection data.
pub fn project_to_image(
    mesh: &Scene,
    settings: &ProjectionConfig,
    lighting_config: &LightingConfig,
) -> ImageBatch {
    log::info!(
        "Initializing projector. Mode: {:?}, Resolution: {}",
        mesh.mode,
        settings.resolution
    );
    // Setup & Validation
    let max_bound: u16 = settings.resolution;
    let max_bound_f32: f32 = max_bound as f32;
    let max_bound_u: usize = max_bound as usize;
    let rows = max_bound_u;
    let cols = max_bound_u;
    let num_views = settings.view_angles.len();
    let mut render_pass_configs = Vec::with_capacity(num_views);
    // Initialize image batch
    let mut image_batch = ImageBatch::new(num_views, rows, cols, NUM_CHANNELS);
    // Handle empty meshes
    if mesh.meshes.is_empty() {
        log::warn!("Empty mesh (no meshes). Returning empty image batch.");
        return image_batch;
    }
    // Set view configs for cube projection
    for i in 0..num_views {
        // Create orientation from the view angles
        let view_angle = &settings.view_angles[i];
        // Create rotation matrix from the orientation
        let rotation_matrix = view_angle.to_rotation_matrix();
        // Create view direction from the rotation matrix
        let view_direction = Coordinate::new(
            rotation_matrix[[0, 2]],
            rotation_matrix[[1, 2]],
            rotation_matrix[[2, 2]],
        );
        // Create view config for the current view
        let view_config = ViewConfig {
            view_direction,
            transformation_matrix: create_view_matrix(&rotation_matrix),
            max_bound_f32,
        };
        render_pass_configs.push(RenderPassConfig {
            view: view_config,
            lighting: *lighting_config,
        });
    }
    // Compute projections for each view in parallel
    image_batch.par_iter_mut().enumerate().for_each(
        |(i, (mut projection, mut depth_map, mut normal_map, mut occupancy_map))| {
            log::info!("Computing projection {} of {}", i + 1, num_views);
            let render_pass_config = &render_pass_configs[i];
            // Compute projection for this view
            let (proj, depth, normal, occ, outliers) =
                projection::compute_projection(&mesh, &settings, &render_pass_config);
            // Log outlier filtering results if requested
            if settings.filtering_window_radius > 0 {
                log::info!(
                    "Outlier filtering removed {} points for view {}",
                    outliers,
                    i + 1
                );
            }
            // Store results in the image batch
            projection.assign(&proj);
            depth_map.assign(&depth);
            normal_map.assign(&normal);
            occupancy_map.assign(&occ);
        },
    );
    image_batch
}

/// Discretizes a mesh into a point cloud.
///
/// Returns:
/// A tuple containing the points and their colors.
pub fn project_to_point_cloud_by_sampling(
    mesh: &Scene,
    settings: &PointCloudConfig,
    lighting_config: &LightingConfig,
) -> (Array2<f32>, Array2<u8>) {
    point_cloud::discretize_mesh_to_point_cloud(mesh, settings, lighting_config)
}
