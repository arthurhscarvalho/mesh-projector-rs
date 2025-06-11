//! High-level module for 3D mesh projection and rendering.

mod buffers;
pub mod config;
mod constants;
mod filtering;
pub mod geometry;
mod lighting;
mod math;
mod point_cloud;
mod projection;
mod rasterization;
mod sampling;

use crate::mesh_io::save_image_data;
use crate::projector::config::{PointCloudConfig, ProjectionConfig, ViewConfig};
use crate::projector::constants::NUM_CHANNELS;
use crate::projector::math::create_view_matrix;
use crate::types::coordinate::Coordinate;
use crate::types::image_batch::ImageBatch;
use crate::types::mesh::Mesh;
use crate::types::orientation::Orientation;
use ndarray::Array2;
use rayon::prelude::*;
use std::path::Path;

/// Computes cube projections for the given mesh and settings.
///
/// Returns:
/// An ImageBatch containing all projection data.
pub fn compute_cube_projections(mesh: &Mesh, settings: &ProjectionConfig) -> ImageBatch {
    log::info!(
        "Initializing projector. Mode: {:?}, Precision: {}, Filtering: {}",
        mesh.mode,
        settings.precision,
        settings.filtering_window_radius
    );
    // Setup & Validation
    let max_bound: u64 = 1 << settings.precision;
    let max_bound_f64: f64 = max_bound as f64;
    let max_bound_u: usize = max_bound as usize;
    let rows = max_bound_u;
    let cols = max_bound_u;
    // Determine number of views and prepare view configurations
    const NUM_VIEWS: usize = 6;
    let mut view_configs = Vec::with_capacity(NUM_VIEWS);
    // Initialize image batch
    let mut image_batch = ImageBatch::new(NUM_VIEWS, rows, cols, NUM_CHANNELS);
    // Handle empty meshes
    if mesh.submeshes.is_empty() {
        log::warn!("Empty mesh (no submeshes). Returning empty image batch.");
        return image_batch;
    }
    // Set view configs for cube projection
    const PI: f64 = std::f64::consts::PI;
    const VIEW_ANGLES: [[f64; 3]; 6] = [
        [0.0, 0.0, 0.0],       // Front: Cam at +Z, looking -Z. No rotation.
        [0.0, PI, 0.0],        // Back: Cam at -Z, looking +Z. Rotate 180 deg around Y.
        [0.0, PI / 2.0, 0.0],  // Right: Cam at +X, looking -X. Rotate 90 deg around Y.
        [0.0, -PI / 2.0, 0.0], // Left: Cam at -X, looking +X. Rotate -90 deg around Y.
        [PI / 2.0, 0.0, 0.0],  // Top: Cam at +Y, looking -Y. Rotate 90 deg around X.
        [-PI / 2.0, 0.0, 0.0], // Bottom: Cam at -Y, looking +Y. Rotate -90 deg around X.
    ];
    // Set view configs for cube projection
    for i in 0..NUM_VIEWS {
        // Create orientation from the view angles
        let angles = Orientation::from(VIEW_ANGLES[i]);
        // Create rotation matrix from the orientation
        let rotation_matrix = angles.to_rotation_matrix();
        // Create view direction from the rotation matrix
        let view_dir = Coordinate::new(
            rotation_matrix[[0, 2]],
            rotation_matrix[[1, 2]],
            rotation_matrix[[2, 2]],
        );
        // Create view config for the current view
        view_configs.push(ViewConfig {
            view_dir,
            rotation_matrix: rotation_matrix.clone(),
            view_matrix: create_view_matrix(&rotation_matrix),
            max_bound_f64,
            light_source: config::LightSource::Camera,
        });
    }
    // Compute projections for each view in parallel
    image_batch.par_iter_mut().enumerate().for_each(
        |(i, (mut projection, mut depth_map, mut normal_map, mut occupancy_map))| {
            let view_config = &view_configs[i];
            log::info!("Computing projection {} of {}", i + 1, NUM_VIEWS);
            // Compute projection for this view
            let (proj, depth, normal, occ, outliers) =
                projection::compute_projection(&mesh, &settings, view_config);
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

/// Computes projections based on the specified mode (either 6 orthographic views or rotation-based).
///
/// Returns an ImageBatch containing all projection data.
pub fn compute_mesh_projections(mesh: &Mesh, settings: &ProjectionConfig) -> ImageBatch {
    log::info!(
        "Initializing projector. Mode: {:?}, Precision: {}, Filtering: {}",
        mesh.mode,
        settings.precision,
        settings.filtering_window_radius
    );
    // Setup & Validation
    let max_bound: u64 = 1 << settings.precision;
    let max_bound_f64: f64 = max_bound as f64;
    let max_bound_u: usize = max_bound as usize;
    let rows = max_bound_u;
    let cols = max_bound_u;
    // Determine number of views and prepare view configurations
    let rotation_config = &settings.rotation_config;
    let num_views = rotation_config.num_steps;
    let mut view_configs = Vec::with_capacity(num_views);
    // Initialize image batch
    let mut image_batch = ImageBatch::new(num_views, rows, cols, NUM_CHANNELS);
    // Handle empty meshes
    if mesh.submeshes.is_empty() {
        log::warn!("Empty mesh (no submeshes). Returning empty image batch.");
        return image_batch;
    }
    // Create view configs for rotation-based projection
    for i in 0..num_views {
        // Create orientation from the view angles
        let angles = Orientation::new(
            rotation_config.angle_deltas.pitch() * i as f64,
            rotation_config.angle_deltas.yaw() * i as f64,
            rotation_config.angle_deltas.roll() * i as f64,
        );
        // Create rotation matrix from the orientation
        let rotation_matrix = angles.to_rotation_matrix();
        // Create view direction from the rotation matrix
        let view_dir = Coordinate::new(
            rotation_matrix[[0, 2]],
            rotation_matrix[[1, 2]],
            rotation_matrix[[2, 2]],
        );
        // Create view config for the current view
        view_configs.push(ViewConfig {
            view_dir,
            rotation_matrix: rotation_matrix.clone(),
            view_matrix: create_view_matrix(&rotation_matrix),
            max_bound_f64,
            light_source: config::LightSource::Camera,
        });
    }
    // Compute projections for each view in parallel
    image_batch.par_iter_mut().enumerate().for_each(
        |(i, (mut projection, mut depth_map, mut normal_map, mut occupancy_map))| {
            let view_config = &view_configs[i];
            log::info!("Computing projection {} of {}", i + 1, num_views);
            // Compute projection for this view
            let (proj, depth, normal, occ, outliers) =
                projection::compute_projection(&mesh, &settings, view_config);
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

/// Computes rotation-based projections and saves them on-the-fly in parallel.
/// This function does not return any projection data.
pub fn compute_and_save_rotation_projections(
    mesh: &Mesh,
    settings: &ProjectionConfig,
    output_dir: &Path,
) {
    log::info!(
        "Initializing rotation-based projector. Precision: {}, Filtering: {}",
        settings.precision,
        settings.filtering_window_radius
    );
    // Setup & Validation
    let max_bound: u64 = 1 << settings.precision;
    let max_bound_f64: f64 = max_bound as f64;
    // Determine number of views and prepare view configurations
    let rotation_config = &settings.rotation_config;
    let num_steps = rotation_config.num_steps;
    let mut view_configs = Vec::with_capacity(num_steps);
    // Handle empty meshes
    if mesh.submeshes.is_empty() {
        log::warn!("Empty mesh (no submeshes). Skipping projection.");
        return;
    }
    // Create view configs for rotation-based projection
    for i in 0..num_steps {
        // Create orientation from the view angles
        let angles = Orientation::new(
            rotation_config.angle_deltas.pitch() * i as f64,
            rotation_config.angle_deltas.yaw() * i as f64,
            rotation_config.angle_deltas.roll() * i as f64,
        );
        // Create rotation matrix from the orientation
        let rotation_matrix = angles.to_rotation_matrix();
        // Create view direction from the rotation matrix
        let view_dir = Coordinate::new(
            rotation_matrix[[0, 2]],
            rotation_matrix[[1, 2]],
            rotation_matrix[[2, 2]],
        );
        // Create view config for the current view
        view_configs.push(ViewConfig {
            view_dir,
            rotation_matrix: rotation_matrix.clone(),
            view_matrix: create_view_matrix(&rotation_matrix),
            max_bound_f64,
            light_source: config::LightSource::Camera,
        });
    }
    // Compute projections for each view in parallel
    view_configs
        .par_iter()
        .enumerate()
        .for_each(|(i, view_config)| {
            log::info!("Computing projection {} of {}", i + 1, num_steps);
            // Compute projection for this view
            let (proj, _, _, _, outliers) =
                projection::compute_projection(mesh, settings, view_config);
            // Log outlier filtering results if requested
            if settings.filtering_window_radius > 0 {
                log::info!(
                    "Outlier filtering removed {} points for view {}",
                    outliers,
                    i + 1
                );
            }
            // Save the projection data for the current view
            if let Err(e) = save_image_data(&proj, output_dir, i, true) {
                log::error!("Failed to save data for view {}: {}", i, e);
            }
        });
    log::info!("Finished computing and saving all rotation-based projections.");
}

/// Discretizes a mesh into a point cloud.
///
/// Returns:
/// A tuple containing the points and their colors.
pub fn mesh_to_point_cloud(mesh: &Mesh, settings: &PointCloudConfig) -> (Array2<f64>, Array2<u8>) {
    point_cloud::discretize_mesh_to_point_cloud(mesh, settings)
}
