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

use buffers::Buffers;
use config::{ProjectorConfig, ViewConfig};
use constants::{NUM_CHANNELS, NUM_IMAGES};
use filtering::filter_outliers;
use geometry::precompute_triangle_normals;
use mesh::Mesh;
use nd::{Array4, Axis};
use rasterization::rasterize_triangle;
use rayon::prelude::*;

/// Computes 6 orthographic projections based on the specified mode.
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
    Array4<u8>, // occupancy_maps (3-channel)
    [u64; NUM_IMAGES],
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
    // Shape definitions for output
    // let shape4d_img = Ix4(NUM_IMAGES, rows, cols, NUM_CHANNELS); // Not strictly needed if creating arrays directly
    // let shape4d_occ = Ix4(NUM_IMAGES, rows, cols, 1); // This was for the old 1-channel map

    // Handle empty meshes
    if mesh.geometry.vertices.is_empty() || mesh.geometry.triangles.is_empty() {
        log::warn!("Empty points or triangles detected. Returning empty buffers.");
        let empty_buffers = Buffers::new(NUM_IMAGES, rows, cols, NUM_CHANNELS);
        let empty_occupancy_map = Array4::<u8>::zeros((NUM_IMAGES, rows, cols, NUM_CHANNELS));
        return (
            empty_buffers.images,
            empty_buffers.depth_maps,
            empty_buffers.normal_maps,
            empty_occupancy_map,
            [0; NUM_IMAGES],
        );
    }
    // Precomputation
    log::info!("Precomputing triangle normals");
    let triangle_normals = precompute_triangle_normals(
        mesh.geometry.vertices.view(),
        mesh.geometry.triangles.view(),
    );
    // Buffer Initialization
    log::info!("Initializing output buffers");
    let mut buffers = Buffers::new(NUM_IMAGES, rows, cols, NUM_CHANNELS);
    // Rasterization (Parallel over Views)
    log::info!("Starting rasterization loop for {} views", NUM_IMAGES);
    let mut z_buffer_views: Vec<_> = buffers.z_buffers.axis_iter_mut(Axis(0)).collect();
    let mut image_views: Vec<_> = buffers.images.axis_iter_mut(Axis(0)).collect();
    let mut depth_map_views: Vec<_> = buffers.depth_maps.axis_iter_mut(Axis(0)).collect();
    let mut normal_map_views: Vec<_> = buffers.normal_maps.axis_iter_mut(Axis(0)).collect();

    z_buffer_views
        .par_iter_mut()
        .zip(image_views.par_iter_mut())
        .zip(depth_map_views.par_iter_mut())
        .zip(normal_map_views.par_iter_mut())
        .enumerate()
        .for_each(
            |(img_layer, (((z_slice, image_slice), depth_slice), normal_slice))| {
                let projection_config_view = ViewConfig::new(img_layer, max_bound_f64);
                let num_faces = mesh.geometry.triangles.nrows();
                for triangle_index in 0..num_faces {
                    rasterize_triangle(
                        mesh,
                        triangle_index,
                        &triangle_normals,
                        &projection_config_view,
                        &mut z_slice.view_mut(),
                        &mut image_slice.view_mut(),
                        &mut depth_slice.view_mut(),
                        &mut normal_slice.view_mut(),
                    );
                }
            },
        );

    // Apply outlier filtering (modifies buffers.images in place)
    // The first returned element (outlier_identification_map) is not directly returned by this function anymore.
    let (_outlier_identification_map, outlier_counts) = filter_outliers(
        &mut buffers.images,
        &buffers.z_buffers,
        settings.filtering_window_radius,
        max_bound_f64,
    );

    // Create occupancy maps (white for occupied, black for empty)
    log::info!("Generating occupancy maps from Z-buffers.");
    let mut occupancy_maps = Array4::<u8>::zeros((NUM_IMAGES, rows, cols, NUM_CHANNELS));
    let mut occupancy_map_views: Vec<_> = occupancy_maps.axis_iter_mut(Axis(0)).collect();
    occupancy_map_views
        .par_iter_mut()
        .enumerate()
        .for_each(|(k, occupancy_slice)| {
            let z_slice = buffers.z_buffers.index_axis(Axis(0), k);
            let is_front_view = k % 2 == 0;
            // Assuming occupancy_slice.dim() gives (slice_rows, slice_cols, slice_channels)
            // and z_slice.dim() gives (slice_rows, slice_cols)
            // These should match rows and cols from the outer scope.
            let (slice_rows, slice_cols, _slice_channels) = occupancy_slice.dim();

            for r in 0..slice_rows {
                for c in 0..slice_cols {
                    let depth_val = z_slice[[r, c]];
                    let is_occupied = if is_front_view {
                        depth_val < f64::INFINITY && depth_val.is_finite()
                    } else {
                        depth_val > f64::NEG_INFINITY && depth_val.is_finite()
                    };

                    if is_occupied {
                        for chan in 0..NUM_CHANNELS {
                            // NUM_CHANNELS should be 3 for RGB
                            occupancy_slice[[r, c, chan]] = 255;
                        }
                    }
                    // Else: remains 0 (black) from initialization
                }
            }
        });

    (
        buffers.images,
        buffers.depth_maps,
        buffers.normal_maps,
        occupancy_maps, // The new 3-channel occupancy map
        outlier_counts,
    )
}
