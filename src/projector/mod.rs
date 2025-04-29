// src/projector/mod.rs
mod buffers;
mod config;
mod constants;
mod filtering;
mod geometry;
mod rasterization;
mod texture;

use crate::ProjectionMode;
// Removed unused HashMap
// use std::collections::HashMap;

// Removed unused conditional_log
// use crate::conditional_log;
use buffers::Buffers;
use config::ProjectionConfig;
use constants::{BACKGROUND_BRIGHTNESS, NUM_IMAGES};
use filtering::filter_outliers;
use geometry::precompute_triangle_normals;
// Removed unused Array3, Ix3
use nd::{Array4, ArrayView2, ArrayView3, Axis, Ix4};
use rasterization::{rasterize_triangle, MaterialProperties};
use rayon::prelude::*;
use rayon::scope;
use texture::TextureData;

// Removed unused ArrayView, CowArray
// use ndarray::{ArrayView, CowArray};

/// Computes 6 orthographic projections based on the specified mode.
/// Args:
/// * `points`: Vertex coordinates (N, 3).
/// * `triangles`: Triangle indices (M, 3) using usize.
/// * `vertex_normals`: Optional vertex normal vectors (N, 3).
/// * `vertex_colors`: Optional vertex colors (N, 3).
/// * `texture`: Optional texture image data (H, W, C).
/// * `triangle_uvs`: Optional UV coordinates per face-vertex (M * 3, 2).
/// * `precision`: Log2 of output image dimensions.
/// * `ka`, `kd`, `ks`, `ns`, `opacity`, `illum`: Material properties.
/// * `filtering_window_radius`: Radius for outlier filter (0=disabled).
/// * `mode`: The ProjectionMode enum indicating how to render.
///
/// Returns:
/// Tuple: (images: Array4<u8>, outlier_counts: [u64; NUM_IMAGES])
pub fn compute_mesh_projections(
    points: ArrayView2<'_, f32>,                 // Vertex positions (N, 3)
    triangles: ArrayView2<'_, usize>,            // Triangle indices (M, 3) -> usize
    vertex_normals: Option<ArrayView2<'_, f32>>, // Optional Vertex normals (N, 3)
    vertex_colors: Option<ArrayView2<'_, f32>>,  // Optional Vertex colors (N, 3)
    texture: Option<ArrayView3<'_, u8>>,         // Optional Texture data (H, W, C)
    triangle_uvs: Option<ArrayView2<'_, f32>>,   // Optional UV coords per face-vertex (M*3, 2)
    precision: u32,                              // Log2 of output image dimensions
    ka: [f32; 3],                                // Ambient color
    kd: [f32; 3],                                // Diffuse color
    ks: [f32; 3],                                // Specular color
    ns: f32,                                     // Shininess (specular exponent)
    opacity: f32,                                // Opacity (0.0=transparent, 1.0=opaque)
    illum: i32,                                  // Illumination model (i32)
    filtering_window_radius: u32,                // Radius for outlier filter (0=disabled)
    mode: ProjectionMode,                        // Rendering mode enum
) -> (Array4<u8>, [u64; NUM_IMAGES]) {
    log::info!(
        "Initializing projector backend. Mode: {:?}, Precision: {}, Filtering: {}",
        mode,
        precision,
        filtering_window_radius
    );

    // --- Setup & Validation ---
    let max_bound: u64 = 1 << precision;
    let max_bound_f32: f32 = max_bound as f32;
    let max_bound_u: usize = max_bound as usize;
    let rows = max_bound_u;
    let cols = max_bound_u;
    let channels = 3; // Assuming RGB output for now

    // Handle potential empty inputs
    if points.is_empty() || triangles.is_empty() {
        log::warn!("Empty points or triangles detected. Returning empty buffers.");
        let shape4d = Ix4(NUM_IMAGES, rows, cols, channels);
        return (
            Array4::from_elem(shape4d, BACKGROUND_BRIGHTNESS),
            [0; NUM_IMAGES], // Return array of zeros for counts
        );
    }

    // Validate data required by the mode *if provided*
    if mode == ProjectionMode::Full || mode == ProjectionMode::ObjTexture {
        if texture.is_none() {
            // This should ideally be caught by Python checks, but panic defensively
            panic!(
                "Texture is required for mode {:?} but was not provided.",
                mode
            );
        }
        if triangle_uvs.is_none() {
            panic!(
                "Triangle UVs are required for mode {:?} but were not provided.",
                mode
            );
        }
        // Validate UV shape if Some
        if let Some(uvs) = triangle_uvs {
            let expected_uv_rows = triangles.nrows() * 3;
            if expected_uv_rows != uvs.nrows() {
                panic!(
                    "Inconsistent input: Triangles ({}) * 3 != UVs ({}). Expected {}.",
                    triangles.nrows(),
                    uvs.nrows(),
                    expected_uv_rows
                );
            }
        }
        // Create TextureData only if texture is present
        // TextureData::new might need adjustment if it doesn't already handle ArrayView
        let texture_data = TextureData::new(texture.expect("Texture checked above"));
        log::info!(
            "Texture loaded: {}x{}x{}",
            texture_data.height_f32,
            texture_data.width_f32,
            texture_data.channels
        );
    } else {
        // ProjectionMode::ObjOnly
        // We don't need texture or UVs.
        // Check for vertex colors (rasterizer handles default if None)
        if vertex_colors.is_none() {
            log::warn!(
                "OBJ_ONLY mode: Vertex colors not provided. Defaulting to white in rasterizer."
            );
        }
    }

    // Validate normals shape if Some
    if let Some(normals) = vertex_normals {
        if points.nrows() != normals.nrows() {
            panic!(
                "Inconsistent input: Points ({}) != Vertex Normals ({}).",
                points.nrows(),
                normals.nrows()
            );
        }
    } else if mode != ProjectionMode::ObjOnly && illum > 0 {
        // Warn if normals are missing but lighting is expected
        log::warn!(
             "Vertex normals not provided for lighting mode {:?} (illum={}). Shading may be flat or incorrect.",
             mode, illum
         );
        // We will proceed and let the rasterizer handle None normals (e.g., skip lighting or use default normal)
    }

    // --- Create Material Properties Struct ---
    let material = MaterialProperties {
        ka,
        kd,
        ks,
        ns,
        opacity,
        illum,
    };

    // --- Precomputation ---
    log::info!("Precomputing triangle normals...");
    let triangle_normals = precompute_triangle_normals(points, triangles);
    log::info!("...Triangle normal precomputation finished.");

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
                let num_faces = triangles.nrows();

                for t_idx in 0..num_faces {
                    rasterize_triangle(
                        t_idx,
                        points,
                        triangles,
                        triangle_uvs,
                        &triangle_normals,
                        vertex_normals,
                        vertex_colors,
                        &material,
                        texture,
                        &projection_config,
                        z_slice,
                        image_slice,
                        mode,
                    );
                }
            });
    });

    log::info!("...Rasterization finished.");

    // --- Post-processing (Outlier Filtering) ---
    log::info!("Starting outlier filtering...");
    let outlier_counts = filter_outliers(
        &mut buffers,
        filtering_window_radius,
        rows,
        cols,
        max_bound_f32,
        channels,
        BACKGROUND_BRIGHTNESS,
    );
    log::info!("...Filtering finished.");

    // --- Return Results ---
    log::info!("Projection computation complete.");
    (buffers.images, outlier_counts)
}
