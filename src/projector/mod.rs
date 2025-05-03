// src/projector/mod.rs
mod buffers;
mod config;
mod constants;
mod filtering;
mod geometry;
pub mod io;
mod lighting;
mod math;
pub mod mesh;
pub mod projection;
mod rasterization;
mod sampling;

use buffers::Buffers;
use config::ProjectionConfig;
use constants::{BACKGROUND_BRIGHTNESS, NUM_IMAGES};
use filtering::filter_outliers;
use geometry::precompute_triangle_normals;
use mesh::Mesh;
use mesh::MeshMode;
use nd::{Array4, Axis, Ix4};
use projection::ProjectionSettings;
use rasterization::rasterize_triangle;
use rayon::prelude::*;
use rayon::scope;

/// Computes 6 orthographic projections based on the specified mode.
///
/// Returns:
/// Tuple: (images: Array4<u8>, occlusion_maps: Array4<u8>, outlier_counts: [u64; NUM_IMAGES])
pub fn compute_mesh_projections(
    mesh: &Mesh,
    settings: &ProjectionSettings, // Projection settings
) -> (Array4<u8>, Array4<u8>, [u64; NUM_IMAGES]) {
    log::info!(
        "Initializing projector backend. Mode: {:?}, Precision: {}, Filtering: {}",
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

    // --- Setup & Validation ---
    let max_bound: u64 = 1 << settings.precision;
    let max_bound_f64: f64 = max_bound as f64;
    let max_bound_u: usize = max_bound as usize;
    let rows = max_bound_u;
    let cols = max_bound_u;
    let channels = 3; // Assuming RGB output for now

    // Handle potential empty inputs
    let shape4d_img = Ix4(NUM_IMAGES, rows, cols, channels);
    let shape4d_occ = Ix4(NUM_IMAGES, rows, cols, 1); // Shape for occlusion maps
    if mesh.geometry.vertices.is_empty() || mesh.geometry.triangles.is_empty() {
        log::warn!("Empty points or triangles detected. Returning empty buffers.");
        return (
            Array4::from_elem(shape4d_img, BACKGROUND_BRIGHTNESS),
            Array4::zeros(shape4d_occ), // Return empty occlusion map
            [0; NUM_IMAGES],            // Return array of zeros for counts
        );
    }

    // Validate data required by the mode
    if mesh.mode == MeshMode::Full || mesh.mode == MeshMode::ObjTexture {
        if mesh.texture.texture.is_none() {
            // This should ideally be caught by Python checks, but panic defensively
            panic!(
                "Texture is required for mode {:?} but was not provided.",
                mesh.mode
            );
        }
        if mesh.texture.triangle_uvs.is_none() {
            panic!(
                "Triangle UVs are required for mode {:?} but were not provided.",
                mesh.mode
            );
        }
        // Validate UV shape if Some
        if let Some(uvs) = &mesh.texture.triangle_uvs {
            let expected_uv_rows = mesh.geometry.triangles.nrows() * 3;
            if expected_uv_rows != uvs.nrows() {
                panic!(
                    "Inconsistent input: Triangles ({}) * 3 != UVs ({}). Expected {}.",
                    mesh.geometry.triangles.nrows(),
                    uvs.nrows(),
                    expected_uv_rows
                );
            }
        }
        // Log texture info if available
        if mesh.mode == MeshMode::ObjTexture || mesh.mode == MeshMode::Full {
            log::info!(
                "Texture loaded: {}x{}x{}",
                mesh.texture.height_f64,
                mesh.texture.width_f64,
                mesh.texture.channels
            );
        }
    } else {
        // MeshMode::ObjOnly
        // We don't need texture or UVs.
        // Check for vertex colors (rasterizer handles default if None)
        if mesh.texture.vertex_colors.is_none() {
            log::warn!(
                "OBJ_ONLY mode: Vertex colors not provided. Defaulting to white in rasterizer."
            );
        }
    }

    // Validate normals shape if Some
    if let Some(normals) = &mesh.geometry.vertex_normals {
        if mesh.geometry.vertices.nrows() != normals.nrows() {
            panic!(
                "Inconsistent input: Vertices ({}) != Vertex Normals ({}).",
                mesh.geometry.vertices.nrows(),
                normals.nrows()
            );
        }
    } else if mesh.mode != MeshMode::ObjOnly && mesh.material.illum > 0 {
        // Warn if normals are missing but lighting is expected
        log::warn!(
             "Vertex normals not provided for lighting mode {:?} (illum={}). Shading may be flat or incorrect.",
             mesh.mode, mesh.material.illum
         );
        // We will proceed and let the rasterizer handle None normals (e.g., skip lighting or use default normal)
    }

    // --- Precomputation ---
    log::info!("Precomputing triangle normals...");
    let triangle_normals = precompute_triangle_normals(
        mesh.geometry.vertices.view(),
        mesh.geometry.triangles.view(),
    );
    log::info!("...Triangle normal handling finished.");

    // --- Buffer Initialization ---
    log::info!("Initializing output buffers...");
    let mut buffers = Buffers::new(NUM_IMAGES, rows, cols, channels);
    log::info!("...Buffer initialization finished.");

    // --- Rasterization (Parallel over Views) ---
    log::info!(
        "Starting parallel rasterization loop for {} views...",
        NUM_IMAGES
    );

    let mut z_buffer_views: Vec<_> = buffers.z_buffers.axis_iter_mut(Axis(0)).collect();
    let mut image_views: Vec<_> = buffers.images.axis_iter_mut(Axis(0)).collect();

    scope(|_s| {
        z_buffer_views
            .par_iter_mut()
            .zip(image_views.par_iter_mut())
            .enumerate()
            .for_each(|(img_layer, (z_slice, image_slice))| {
                let projection_config = ProjectionConfig::new(img_layer);
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
    });
    log::info!("...Rasterization finished.");

    // Apply outlier filtering if enabled
    let (occlusion_maps, outlier_counts) = if settings.filtering_window_radius > 0 {
        log::info!(
            "Filtering outliers with window radius {}...",
            settings.filtering_window_radius
        );
        filter_outliers(
            &buffers.z_buffers,
            settings.filtering_window_radius,
            max_bound_f64,
        )
    } else {
        log::info!("Outlier filtering disabled.");
        let shape4d_occ = Ix4(NUM_IMAGES, rows, cols, 1);
        (Array4::zeros(shape4d_occ), [0; NUM_IMAGES])
    };

    // --- Apply filtering to images using the occlusion map ---
    if settings.filtering_window_radius > 0 {
        log::info!("Applying filtering to projection images based on occlusion map...");
        // Iterate through images and occlusion maps simultaneously
        buffers
            .images
            .axis_iter_mut(Axis(0))
            .zip(occlusion_maps.axis_iter(Axis(0)))
            .par_bridge() // Parallelize the application
            .for_each(|(mut img_slice, occ_slice)| {
                // img_slice shape: (rows, cols, channels)
                // occ_slice shape: (rows, cols, 1)
                for r in 0..rows {
                    for c in 0..cols {
                        // Check the occlusion map's single channel
                        if occ_slice[[r, c, 0]] == 1 {
                            // If outlier, set all channels of the image pixel to background
                            for chan in 0..channels {
                                img_slice[[r, c, chan]] = BACKGROUND_BRIGHTNESS;
                            }
                        }
                    }
                }
            });
        log::info!("...Filtering applied to images.");
    }

    log::info!("...Processing complete.");

    (buffers.images, occlusion_maps, outlier_counts)
}
