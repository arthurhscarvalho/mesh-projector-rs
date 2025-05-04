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
use constants::{BACKGROUND_BRIGHTNESS, NUM_CHANNELS, NUM_IMAGES};
use filtering::filter_outliers;
use geometry::precompute_triangle_normals;
use mesh::Mesh;
use nd::{Array4, Axis, Ix4};
use rasterization::rasterize_triangle;
use rayon::prelude::*;

/// Computes 6 orthographic projections based on the specified mode.
///
/// Returns:
/// Tuple: (images: Array4<u8>, occlusion_maps: Array4<u8>, outlier_counts: [u64; NUM_IMAGES])
pub fn compute_mesh_projections(
    mesh: &Mesh,
    settings: &ProjectorConfig,
) -> (Array4<u8>, Array4<u8>, [u64; NUM_IMAGES]) {
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
    let shape4d_img = Ix4(NUM_IMAGES, rows, cols, NUM_CHANNELS);
    let shape4d_occ = Ix4(NUM_IMAGES, rows, cols, 1); // Shape for occlusion maps
                                                      // Handle empty meshes (this check is kept for robustness)
    if mesh.geometry.vertices.is_empty() || mesh.geometry.triangles.is_empty() {
        log::warn!("Empty points or triangles detected. Returning empty buffers.");
        return (
            Array4::from_elem(shape4d_img, BACKGROUND_BRIGHTNESS),
            Array4::zeros(shape4d_occ), // Return empty occlusion map
            [0; NUM_IMAGES],            // Return array of zeros for counts
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
    z_buffer_views
        .par_iter_mut()
        .zip(image_views.par_iter_mut())
        .enumerate()
        .for_each(|(img_layer, (z_slice, image_slice))| {
            let projection_config = ViewConfig::new(img_layer);
            let num_faces = mesh.geometry.triangles.nrows();
            for triangle_index in 0..num_faces {
                rasterize_triangle(
                    mesh,
                    triangle_index,
                    &triangle_normals,
                    &projection_config,
                    &mut z_slice.view_mut(),
                    &mut image_slice.view_mut(),
                );
            }
        });
    // Apply outlier filtering if enabled
    let (occlusion_maps, outlier_counts) = filter_outliers(
        &mut buffers.images,
        &buffers.z_buffers,
        settings.filtering_window_radius,
        max_bound_f64,
    );
    (buffers.images, occlusion_maps, outlier_counts)
}
