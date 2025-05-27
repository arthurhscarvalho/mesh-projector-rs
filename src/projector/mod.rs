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
pub mod rotation;
mod sampling;

use crate::projector::buffers::Buffers;
use crate::projector::config::{ProjectorConfig, RotationConfig, ViewConfig};
use crate::projector::constants::{NUM_CHANNELS, NUM_IMAGES};
use crate::projector::filtering::filter_outliers;
use crate::projector::geometry::precompute_triangle_normals;
use crate::projector::io::preprocess_vertices;
use mesh::Mesh;
use ndarray::{s, Array2, Array3, Array4, Axis};
use rasterization::rasterize_triangle;
use rayon::prelude::*;
use std::sync::Arc;

/// Computes a single orthographic projection for the given view configuration.
fn compute_single_projection(
    mut mesh: Mesh,
    settings: &ProjectorConfig,
    view_config: &ViewConfig,
) -> (Array4<u8>, Array4<u8>, Array4<u8>, Array3<u8>, u64) {
    let max_bound: u64 = 1 << settings.precision;
    let max_bound_f64: f64 = max_bound as f64;
    let max_bound_u: usize = max_bound as usize;
    let rows = max_bound_u;
    let cols = max_bound_u;

    // Precompute triangle normals for this rotated mesh
    log::info!("Precomputing triangle normals for current view");
    let triangle_normals = precompute_triangle_normals(
        mesh.geometry.vertices.view(),
        mesh.geometry.triangles.view(),
    );

    // Initialize buffers for a single view
    let mut buffers = Buffers::new(1, rows, cols, NUM_CHANNELS);

    let mut image_view = buffers.images.index_axis_mut(Axis(0), 0);
    let mut depth_map_view = buffers.depth_maps.index_axis_mut(Axis(0), 0);
    let mut normal_map_view = buffers.normal_maps.index_axis_mut(Axis(0), 0);
    let mut z_buffer_view = buffers.z_buffers.index_axis_mut(Axis(0), 0);

    // Rasterize all triangles
    let num_faces = mesh.geometry.triangles.nrows();
    for triangle_index in 0..num_faces {
        rasterize_triangle(
            &mut mesh,
            triangle_index,
            &triangle_normals,
            view_config,
            &mut z_buffer_view.view_mut(),
            &mut image_view.view_mut(),
            &mut depth_map_view.view_mut(),
            &mut normal_map_view.view_mut(),
        );
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
) -> (
    Array4<u8>, // images
    Array4<u8>, // depth_maps
    Array4<u8>, // normal_maps
    Array3<u8>, // occupancy_maps (3-channel)
    Vec<u64>,   // outlier_counts
) {
    log::info!(
        "Initializing projector. Mode: {:?}, Precision: {}, Filtering: {}",
        mesh.mode,
        settings.precision,
        settings.filtering_window_radius
    );
    log::info!(
        "Material Properties: Ka={:?}, Kd={:?}, Ks={:?}, Ns={}, Opacity={}, Illum={}",
        mesh.material.ka,
        mesh.material.kd,
        mesh.material.ks,
        mesh.material.ns,
        mesh.material.opacity,
        mesh.material.illum
    );

    // Setup & Validation
    let max_bound: u64 = 1 << settings.precision;
    let max_bound_f64: f64 = max_bound as f64;
    let max_bound_u: usize = max_bound as usize;
    let rows = max_bound_u;
    let cols = max_bound_u;

    // Handle empty meshes
    if mesh.geometry.vertices.is_empty() || mesh.geometry.triangles.is_empty() {
        log::warn!("Empty points or triangles detected. Returning empty buffers.");
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
    let (num_views, view_configs) = if let Some(rotation_config) = &settings.rotation_config {
        // Rotation-based projection
        let num_steps = rotation_config.num_steps;
        let mut configs = Vec::with_capacity(num_steps);

        // Create view configs for rotation-based projection
        for _ in 0..num_steps {
            // Create view config for front view along Z axis
            configs.push(ViewConfig {
                depth_axis: 2, // Z axis
                is_front_view: true,
                view_dir: (0.0, 0.0, 1.0),
                p_axis: 0, // X axis
                q_axis: 1, // Y axis
                max_bound_f64,
            });
        }
        (num_steps, configs)
    } else {
        // Default 6-view orthographic projection
        let configs: Vec<_> = (0..NUM_IMAGES)
            .map(|i| ViewConfig::new(i, max_bound_f64))
            .collect();
        (NUM_IMAGES, configs)
    };

    // Compute projections for each view
    let mut images = Vec::with_capacity(num_views);
    let mut depth_maps = Vec::with_capacity(num_views);
    let mut normal_maps = Vec::with_capacity(num_views);
    let mut occupancy_maps = Vec::with_capacity(num_views);
    let mut outlier_counts = Vec::with_capacity(num_views);

    // Process each view
    for (i, view_config) in view_configs.iter().enumerate() {
        log::info!("Computing projection {} of {}", i + 1, num_views);

        // Create a rotated mesh for this view
        let mut rotated_mesh = mesh.clone();
        if let Some(rotation_config) = &settings.rotation_config {
            if i > 0 {
                let angle_x = rotation_config.angle_delta_x * i as f64;
                let angle_y = rotation_config.angle_delta_y * i as f64;
                let angle_z = rotation_config.angle_delta_z * i as f64;
                rotation::rotate_mesh(&mut rotated_mesh, angle_x, angle_y, angle_z);
                // Preprocess vertices after rotation to ensure proper scaling
                rotated_mesh.geometry.vertices =
                    preprocess_vertices(rotated_mesh.geometry.vertices, settings.precision as u32);
            }
        }

        let (img, depth, normal, occ, outlier) =
            compute_single_projection(rotated_mesh, settings, view_config);

        images.push(img);
        depth_maps.push(depth);
        normal_maps.push(normal);
        occupancy_maps.push(occ);
        outlier_counts.push(outlier);
    }

    // Stack all views into 4D arrays
    let stacked_images = ndarray::stack(
        Axis(0),
        &images
            .iter()
            .map(|x| x.slice(s![0, .., .., ..]))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let stacked_depth_maps = ndarray::stack(
        Axis(0),
        &depth_maps
            .iter()
            .map(|x| x.slice(s![0, .., .., ..]))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let stacked_normal_maps = ndarray::stack(
        Axis(0),
        &normal_maps
            .iter()
            .map(|x| x.slice(s![0, .., .., ..]))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let stacked_occupancy_maps = ndarray::stack(
        Axis(0),
        &occupancy_maps
            .iter()
            .map(|x| x.slice(s![0, .., ..]))
            .collect::<Vec<_>>(),
    )
    .unwrap();

    (
        stacked_images,
        stacked_depth_maps,
        stacked_normal_maps,
        stacked_occupancy_maps,
        outlier_counts,
    )
}
