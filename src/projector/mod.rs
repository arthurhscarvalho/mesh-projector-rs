// src/projector/mod.rs
mod buffers;
pub mod config;
mod constants;
mod filtering;
mod geometry;
pub mod io;
mod lighting;
mod math;
pub mod mesh;
mod rasterization;
mod sampling;

use crate::projector::buffers::Buffers;
use crate::projector::config::{ProjectorConfig, ViewConfig};
use crate::projector::constants::NUM_CHANNELS;
use crate::projector::filtering::filter_outliers;
use crate::projector::math::{
    combine_rotations, rotation_matrix_x, rotation_matrix_y, rotation_matrix_z,
};
use mesh::Mesh;
use ndarray::{array, s, Array2, Array3, Array4, Axis};
use rasterization::rasterize_triangle;
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;

/// Creates a 4x4 view matrix from a 3x3 rotation matrix.
/// For orthographic projection, the camera is the projection plane itself.
/// This creates a view matrix that transforms from world space to view space,
/// where the view direction becomes the Z axis.
fn create_view_matrix(rotation_matrix: &Array2<f64>) -> Array2<f64> {
    // In orthographic projection, we don't position the camera at a distance
    // Instead, the projection plane itself acts as the camera
    // The rotation matrix should already be set up to transform the view direction to the Z axis
    let r = rotation_matrix;

    // Create the 4x4 view matrix without translation component
    // The rotation part ensures the view direction is transformed to the Z axis
    array![
        [r[[0, 0]], r[[0, 1]], r[[0, 2]], 0.0],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]], 0.0],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]], 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]
}

/// Computes a single orthographic projection for the given view configuration.
fn compute_single_projection(
    mesh: &Mesh,
    settings: &ProjectorConfig,
    view_config: &ViewConfig,
) -> (Array4<u8>, Array4<u8>, Array4<u8>, Array3<u8>, u64) {
    let max_bound: u64 = 1 << settings.precision;
    let max_bound_f64: f64 = max_bound as f64;
    let max_bound_u: usize = max_bound as usize;
    let rows = max_bound_u;
    let cols = max_bound_u;

    // Initialize buffers for a single view
    let mut buffers = Buffers::new(1, rows, cols, NUM_CHANNELS);

    let mut image_view = buffers.images.index_axis_mut(Axis(0), 0);
    let mut depth_map_view = buffers.depth_maps.index_axis_mut(Axis(0), 0);
    let mut normal_map_view = buffers.normal_maps.index_axis_mut(Axis(0), 0);
    let mut z_buffer_view = buffers.z_buffers.index_axis_mut(Axis(0), 0);

    let mut global_triangle_offset = 0;
    // Rasterize all triangles from all submeshes
    for submesh in &mesh.submeshes {
        let num_faces = submesh.geometry.triangles.nrows();
        for triangle_index in 0..num_faces {
            // triangle_index is local to the submesh
            let face_normal_view = submesh.geometry.face_normals.row(triangle_index);
            let uv_lookup_triangle_id = global_triangle_offset + triangle_index;
            rasterize_triangle(
                &submesh.geometry,
                &submesh.material,
                &mesh.texture,
                mesh.mode,
                triangle_index,        // Local triangle index for geometry
                uv_lookup_triangle_id, // Global triangle ID for UVs
                face_normal_view,
                view_config,
                &mut z_buffer_view.view_mut(),
                &mut image_view.view_mut(),
                &mut depth_map_view.view_mut(),
                &mut normal_map_view.view_mut(),
            );
        }
        global_triangle_offset += num_faces;
    }

    // Apply outlier filtering
    let (_, outlier_count) = filter_outliers(
        &mut buffers.images,
        &buffers.z_buffers,
        settings.filtering_window_radius,
        max_bound_f64,
    );

    // Create occupancy map
    let mut occupancy_map = Array3::<u8>::zeros((1, rows, cols));
    let z_slice = buffers.z_buffers.index_axis(Axis(0), 0);
    for r in 0..rows {
        for c in 0..cols {
            let depth_val = z_slice[[r, c]];
            let is_occupied = if view_config.is_front_view {
                depth_val < f64::INFINITY && depth_val.is_finite()
            } else {
                depth_val > f64::NEG_INFINITY && depth_val.is_finite()
            };
            if is_occupied {
                occupancy_map[[0, r, c]] = 255;
            }
        }
    }

    // Create new 4D arrays with the correct shape
    let mut single_image = Array4::<u8>::zeros((1, rows, cols, NUM_CHANNELS));
    let mut single_depth = Array4::<u8>::zeros((1, rows, cols, NUM_CHANNELS));
    let mut single_normal = Array4::<u8>::zeros((1, rows, cols, NUM_CHANNELS));

    // Copy data from the buffers
    single_image
        .slice_mut(s![0, .., .., ..])
        .assign(&buffers.images.slice(s![0, .., .., ..]));
    single_depth
        .slice_mut(s![0, .., .., ..])
        .assign(&buffers.depth_maps.slice(s![0, .., .., ..]));
    single_normal
        .slice_mut(s![0, .., .., ..])
        .assign(&buffers.normal_maps.slice(s![0, .., .., ..]));

    (
        single_image,
        single_depth,
        single_normal,
        occupancy_map,
        outlier_count[0],
    )
}

/// Computes projections based on the specified mode (either 6 orthographic views or rotation-based).
///
/// Returns:
/// Tuple: (images, depth_maps, normal_maps, occupancy_maps, outlier_counts)
/// where occupancy_maps are RGB images (255 for occupied, 0 for empty).
pub fn compute_mesh_projections(
    mesh: &Mesh,
    settings: &ProjectorConfig,
) -> (Array4<u8>, Array4<u8>, Array4<u8>, Array3<u8>, Vec<u64>) {
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

    // Handle empty meshes
    if mesh.submeshes.is_empty()
        || mesh
            .submeshes
            .iter()
            .all(|sm| sm.geometry.vertices.is_empty() || sm.geometry.triangles.is_empty())
    {
        log::warn!(
            "Empty mesh (no submeshes or no geometry in submeshes). Returning empty buffers."
        );
        let empty_buffers = Buffers::new(1, rows, cols, NUM_CHANNELS);
        let empty_occupancy_map = Array3::<u8>::zeros((1, rows, cols));
        return (
            empty_buffers.images,
            empty_buffers.depth_maps,
            empty_buffers.normal_maps,
            empty_occupancy_map,
            vec![0],
        );
    }

    // Determine number of views and prepare view configurations
    let rotation_config = match &settings.rotation_config {
        Some(config) => config,
        None => {
            log::warn!(
                "Rotation configuration is not provided. Skipping rotation-based projection."
            );
            let empty_buffers = Buffers::new(1, rows, cols, NUM_CHANNELS);
            let empty_occupancy_map = Array3::<u8>::zeros((1, rows, cols));
            return (
                empty_buffers.images,
                empty_buffers.depth_maps,
                empty_buffers.normal_maps,
                empty_occupancy_map,
                vec![0],
            );
        }
    };

    let num_views = rotation_config.num_steps;
    let mut view_configs = Vec::with_capacity(num_views);

    // Create view configs for rotation-based projection
    for i in 0..num_views {
        let angle_x = rotation_config.angle_delta_x * i as f64;
        let angle_y = rotation_config.angle_delta_y * i as f64;
        let angle_z = rotation_config.angle_delta_z * i as f64;

        let rx = rotation_matrix_x(angle_x);
        let ry = rotation_matrix_y(angle_y);
        let rz = rotation_matrix_z(angle_z);
        let rotation_matrix = combine_rotations(&rx, &ry, &rz);

        let view_dir_tuple = (
            rotation_matrix[[0, 2]],
            rotation_matrix[[1, 2]],
            rotation_matrix[[2, 2]],
        );

        view_configs.push(ViewConfig {
            is_front_view: true,
            view_dir: view_dir_tuple,
            rotation_matrix: rotation_matrix.clone(),
            view_matrix: create_view_matrix(&rotation_matrix),
            max_bound_f64,
            lighting_config: config::LightingConfig::default(),
        });
    }

    let settings_arc = Arc::new(settings);
    let mesh_arc = Arc::new(mesh.clone());

    let mut projections = Array4::<u8>::zeros((num_views, rows, cols, NUM_CHANNELS));
    let mut depth_maps = Array4::<u8>::zeros((num_views, rows, cols, NUM_CHANNELS));
    let mut normal_maps = Array4::<u8>::zeros((num_views, rows, cols, NUM_CHANNELS));
    let mut occupancy_maps = Array3::<u8>::zeros((num_views, rows, cols));
    let mut outlier_counts = vec![0; num_views];

    // Compute projections for each view in parallel
    projections
        .axis_iter_mut(Axis(0))
        .zip(depth_maps.axis_iter_mut(Axis(0)))
        .zip(normal_maps.axis_iter_mut(Axis(0)))
        .zip(occupancy_maps.axis_iter_mut(Axis(0)))
        .zip(outlier_counts.iter_mut())
        .collect::<Vec<_>>()
        .into_par_iter()
        .enumerate()
        .for_each(
            |(
                i,
                (
                    (((mut projection, mut depth_map), mut normal_map), mut occupancy_map),
                    outlier_count,
                ),
            )| {
                let view_config = &view_configs[i];
                log::info!("Computing projection {} of {}", i + 1, num_views);

                // Compute projection for this view
                let (img, depth, normal, occ, outlier) =
                    compute_single_projection(&mesh_arc, &settings_arc, view_config);

                // Store results in the 5D array and other containers
                projection.assign(&img);
                depth_map.assign(&depth);
                normal_map.assign(&normal);
                occupancy_map.assign(&occ);
                *outlier_count = outlier;
            },
        );

    (
        projections,
        depth_maps,
        normal_maps,
        occupancy_maps,
        outlier_counts,
    )
}

/// Computes rotation-based projections and saves them on-the-fly in parallel.
/// This function does not return any projection data.
pub fn compute_and_save_rotation_projections(
    mesh: &Mesh,
    settings: &ProjectorConfig,
    output_dir: &Path,
) {
    log::info!(
        "Initializing rotation-based projector. Precision: {}, Filtering: {}",
        settings.precision,
        settings.filtering_window_radius
    );

    let rotation_config = match &settings.rotation_config {
        Some(config) => config,
        None => {
            log::warn!(
                "Rotation configuration is not provided. Skipping rotation-based projection."
            );
            return;
        }
    };

    // Setup & Validation
    let max_bound: u64 = 1 << settings.precision;
    let max_bound_f64: f64 = max_bound as f64;

    // Handle empty meshes - CORRECTED
    if mesh.submeshes.is_empty()
        || mesh
            .submeshes
            .iter()
            .all(|sm| sm.geometry.vertices.is_empty() || sm.geometry.triangles.is_empty())
    {
        log::warn!("Empty mesh (no submeshes or no geometry in submeshes). Skipping projection.");
        return;
    }

    let num_steps = rotation_config.num_steps;
    let mut view_configs = Vec::with_capacity(num_steps);

    // Create view configs for rotation-based projection
    for i in 0..num_steps {
        let angle_x = rotation_config.angle_delta_x * i as f64;
        let angle_y = rotation_config.angle_delta_y * i as f64;
        let angle_z = rotation_config.angle_delta_z * i as f64;

        let rx = rotation_matrix_x(angle_x);
        let ry = rotation_matrix_y(angle_y);
        let rz = rotation_matrix_z(angle_z);
        let rotation_matrix = combine_rotations(&rx, &ry, &rz);

        let view_dir_tuple = (
            rotation_matrix[[0, 2]],
            rotation_matrix[[1, 2]],
            rotation_matrix[[2, 2]],
        );

        view_configs.push(ViewConfig {
            is_front_view: true,
            view_dir: view_dir_tuple,
            rotation_matrix: rotation_matrix.clone(),
            view_matrix: create_view_matrix(&rotation_matrix),
            max_bound_f64,
            lighting_config: config::LightingConfig::default(),
        });
    }

    let mesh_arc = Arc::new(mesh.clone());
    let settings_arc = Arc::new(settings);
    let output_dir_arc = Arc::new(output_dir.to_path_buf());

    view_configs
        .par_iter()
        .enumerate()
        .for_each(|(i, view_config)| {
            log::info!("Computing projection {} of {}", i + 1, num_steps);

            // Mesh is no longer rotated here. Transformation happens in compute_single_projection.
            let current_mesh_for_projection = Arc::clone(&mesh_arc).as_ref().clone();

            let (img, _, _, _, _) =
                compute_single_projection(&current_mesh_for_projection, &settings_arc, view_config);

            // Save the projection data for the current view
            if let Err(e) = io::save_projection_data(&img, &output_dir_arc, i, true) {
                log::error!("Failed to save data for view {}: {}", i, e);
            }
        });

    log::info!("Finished computing and saving all rotation-based projections.");
}
