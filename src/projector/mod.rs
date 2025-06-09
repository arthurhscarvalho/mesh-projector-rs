//! High-level module for 3D mesh projection and rendering.

mod buffers;
pub mod config;
mod constants;
mod filtering;
pub mod geometry;
mod lighting;
mod math;
mod projection;
mod rasterization;
mod sampling;

use crate::mesh_io::save_image_data;
use crate::projector::config::{ProjectorConfig, ViewConfig};
use crate::projector::constants::NUM_CHANNELS;
use crate::projector::math::create_view_matrix;
use crate::types::coordinate::Coordinate;
use crate::types::mesh::Mesh;
use crate::types::orientation::Orientation;
use ndarray::{Array3, Array4, Axis};
use rayon::prelude::*;
use std::path::Path;

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
    // Determine number of views and prepare view configurations
    let rotation_config = &settings.rotation_config;
    let num_views = rotation_config.num_steps;
    let mut view_configs = Vec::with_capacity(num_views);
    // Initialize all arrays
    let mut projections = Array4::<u8>::zeros((num_views, rows, cols, NUM_CHANNELS));
    let mut depth_maps = Array4::<u8>::zeros((num_views, rows, cols, NUM_CHANNELS));
    let mut normal_maps = Array4::<u8>::zeros((num_views, rows, cols, NUM_CHANNELS));
    let mut occupancy_maps = Array3::<u8>::zeros((num_views, rows, cols));
    let mut outlier_counts = vec![0; num_views];
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
        return (
            projections,
            depth_maps,
            normal_maps,
            occupancy_maps,
            outlier_counts,
        );
    }
    // Create view configs for rotation-based projection
    for i in 0..num_views {
        let angles = if num_views == 1 {
            // For a single view, use the deltas as absolute rotation angles
            rotation_config.angle_deltas
        } else {
            // For multiple views (rotational sweep), treat deltas as increments
            Orientation::new(
                rotation_config.angle_delta_x() * i as f64,
                rotation_config.angle_delta_y() * i as f64,
                rotation_config.angle_delta_z() * i as f64,
            )
        };

        let rotation_matrix = angles.to_rotation_matrix();
        let view_dir = Coordinate::new(
            rotation_matrix[[0, 2]],
            rotation_matrix[[1, 2]],
            rotation_matrix[[2, 2]],
        );

        view_configs.push(ViewConfig {
            view_dir,
            rotation_matrix: rotation_matrix.clone(),
            view_matrix: create_view_matrix(&rotation_matrix),
            max_bound_f64,
            light_source: config::LightSource::Camera,
        });
    }
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
                    projection::compute_projection(&mesh, &settings, view_config);
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

    let rotation_config = &settings.rotation_config;

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
        let angles = Orientation::new(
            rotation_config.angle_delta_x() * i as f64,
            rotation_config.angle_delta_y() * i as f64,
            rotation_config.angle_delta_z() * i as f64,
        );

        let rotation_matrix = angles.to_rotation_matrix();
        let view_dir = Coordinate::new(
            rotation_matrix[[0, 2]],
            rotation_matrix[[1, 2]],
            rotation_matrix[[2, 2]],
        );

        view_configs.push(ViewConfig {
            view_dir,
            rotation_matrix: rotation_matrix.clone(),
            view_matrix: create_view_matrix(&rotation_matrix),
            max_bound_f64,
            light_source: config::LightSource::Camera,
        });
    }
    view_configs
        .par_iter()
        .enumerate()
        .for_each(|(i, view_config)| {
            log::info!("Computing projection {} of {}", i + 1, num_steps);
            let (img, _, _, _, _) = projection::compute_projection(&mesh, &settings, view_config);

            // Save the projection data for the current view
            if let Err(e) = save_image_data(&img, output_dir, i, true) {
                log::error!("Failed to save data for view {}: {}", i, e);
            }
        });
    log::info!("Finished computing and saving all rotation-based projections.");
}

/// Discretizes a mesh into a 3D point cloud, maintaining the 3D structure.
/// Instead of projecting the mesh onto a 2D plane, this function samples points
/// across the triangle surfaces in 3D space.
///
/// Returns:
/// - points: Array2<f64> - 3D points in the range [0, max_bound]
/// - colors: Array2<u8> - RGB colors for each point (0-255 range)
pub fn discretize_mesh_to_point_cloud(
    mesh: &Mesh,
    settings: &ProjectorConfig,
) -> (ndarray::Array2<f64>, ndarray::Array2<u8>) {
    log::info!(
        "Initializing 3D mesh discretization. Mode: {:?}, Precision: {}",
        mesh.mode,
        settings.precision
    );
    let max_bound: u64 = 1 << settings.precision;
    let max_bound_f64: f64 = max_bound as f64;
    let density_factor = settings.point_cloud_density_factor.unwrap_or(1.0);
    // Initialize point cloud buffers
    let mut points = Vec::new();
    let mut colors = Vec::new();
    let mut num_points: usize = 0;
    // Calculate total mesh area for density calculation
    let mut total_mesh_area = 0.0;
    for submesh_index in 0..mesh.submeshes.len() {
        let submesh = &mesh.submeshes[submesh_index];
        let num_faces = submesh.geometry.triangles.nrows();
        for triangle_index in 0..num_faces {
            let triangle_data =
                rasterization::extract_triangle_data(&mesh, submesh_index, triangle_index);
            let triangle_area = geometry::calculate_triangle_area(&triangle_data.vertices_ws);
            total_mesh_area += triangle_area;
        }
    }
    // Density factor is the number of points per unit area.
    let density = (density_factor * max_bound_f64.powf(2.0)) / total_mesh_area.max(1e-9);
    // Create a placeholder view config for lighting calculations
    let view_config = ViewConfig {
        view_dir: Coordinate::new(0.0, 0.0, 1.0),
        rotation_matrix: ndarray::Array2::eye(3),
        view_matrix: ndarray::Array2::eye(4),
        max_bound_f64,
        light_source: config::LightSource::Camera,
    };
    // Process each submesh
    for submesh_index in 0..mesh.submeshes.len() {
        let submesh = &mesh.submeshes[submesh_index];
        let num_faces = submesh.geometry.triangles.nrows();
        for triangle_index in 0..num_faces {
            // Extract triangle data
            let triangle_data =
                rasterization::extract_triangle_data(&mesh, submesh_index, triangle_index);
            // Calculate triangle area
            let triangle_area = geometry::calculate_triangle_area(&triangle_data.vertices_ws);
            // Calculate number of subdivisions based on triangle area and density
            let n_subdiv = (triangle_area * density).sqrt().floor() as usize;
            if n_subdiv == 0 {
                continue;
            }
            // Sample points across the triangle surface
            for i in 0..=n_subdiv {
                for j in 0..=(n_subdiv - i) {
                    let u = i as f64 / n_subdiv as f64;
                    let v = j as f64 / n_subdiv as f64;
                    let w = 1.0 - u - v;
                    if u < 0.0 || v < 0.0 || w < 0.0 {
                        continue;
                    }
                    // Use the shared interpolation function
                    let attributes = rasterization::interpolate_triangle_attributes(
                        &triangle_data,
                        (u, v, w),
                        mesh,
                    );
                    // Calculate final color using the shared lighting function
                    let final_color = rasterization::calculate_final_color(
                        &attributes,
                        &view_config,
                        &submesh.material,
                        mesh.mode,
                        &config::LightSource::Camera,
                    );
                    // Convert to sRGB and clamp to u8 range
                    let r_srgb = (final_color[0].powf(1.0 / 2.2) * 255.0).min(255.0) as u8;
                    let g_srgb = (final_color[1].powf(1.0 / 2.2) * 255.0).min(255.0) as u8;
                    let b_srgb = (final_color[2].powf(1.0 / 2.2) * 255.0).min(255.0) as u8;
                    // Add point and color to the flatten arrays
                    let pos_array: [f64; 3] = attributes.position_ws.into();
                    points.push(pos_array[0]);
                    points.push(pos_array[1]);
                    points.push(pos_array[2]);
                    colors.push(r_srgb);
                    colors.push(g_srgb);
                    colors.push(b_srgb);
                    num_points += 1;
                }
            }
        }
    }
    let points_array = nd::Array2::from_shape_vec((num_points, 3), points).unwrap();
    let colors_array = nd::Array2::from_shape_vec((num_points, 3), colors).unwrap();
    (points_array, colors_array)
}
