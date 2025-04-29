// src/projector/mod.rs
mod buffers;
mod config;
mod constants;
mod filtering;
mod geometry;
mod lighting;
pub mod material;
mod math;
pub mod mesh;
pub mod projection;
mod rasterization;
mod sampling;
pub mod texture;

use crate::ProjectionMode;
use buffers::Buffers;
use config::ProjectionConfig;
use constants::{BACKGROUND_BRIGHTNESS, NUM_IMAGES};
use filtering::filter_outliers;
use geometry::precompute_triangle_normals;
use material::MaterialProperties;
use mesh::{Mesh, MeshAttributes};
use nd::{Array4, Axis, Ix4};
use projection::ProjectionSettings;
use rasterization::rasterize_triangle;
use rayon::prelude::*;
use rayon::scope;

/// Computes 6 orthographic projections based on the specified mode.
///
/// Returns:
/// Tuple: (images: Array4<u8>, outlier_counts: [u64; NUM_IMAGES])
pub fn compute_mesh_projections<'a>(
    mesh: &Mesh<'a>,                 // Core mesh data
    attributes: &MeshAttributes<'a>, // Optional mesh attributes
    material: &MaterialProperties,   // Material properties
    settings: &ProjectionSettings,   // Projection settings
) -> (Array4<u8>, [u64; NUM_IMAGES]) {
    log::info!(
        "Initializing projector backend. Mode: {:?}, Precision: {}, Filtering: {}",
        settings.mode,
        settings.precision,
        settings.filtering_window_radius
    );

    // --- Setup & Validation ---
    let max_bound: u64 = 1 << settings.precision;
    let max_bound_f32: f32 = max_bound as f32;
    let max_bound_u: usize = max_bound as usize;
    let rows = max_bound_u;
    let cols = max_bound_u;
    let channels = 3; // Assuming RGB output for now

    // Handle potential empty inputs
    if mesh.vertices.is_empty() || mesh.triangles.is_empty() {
        log::warn!("Empty points or triangles detected. Returning empty buffers.");
        let shape4d = Ix4(NUM_IMAGES, rows, cols, channels);
        return (
            Array4::from_elem(shape4d, BACKGROUND_BRIGHTNESS),
            [0; NUM_IMAGES], // Return array of zeros for counts
        );
    }

    // Validate data required by the mode
    if settings.mode == ProjectionMode::Full || settings.mode == ProjectionMode::ObjTexture {
        if attributes.texture.is_none() {
            // This should ideally be caught by Python checks, but panic defensively
            panic!(
                "Texture is required for mode {:?} but was not provided.",
                settings.mode
            );
        }
        if attributes.triangle_uvs.is_none() {
            panic!(
                "Triangle UVs are required for mode {:?} but were not provided.",
                settings.mode
            );
        }
        // Validate UV shape if Some
        if let Some(uvs) = attributes.triangle_uvs {
            let expected_uv_rows = mesh.triangles.nrows() * 3;
            if expected_uv_rows != uvs.nrows() {
                panic!(
                    "Inconsistent input: Triangles ({}) * 3 != UVs ({}). Expected {}.",
                    mesh.triangles.nrows(),
                    uvs.nrows(),
                    expected_uv_rows
                );
            }
        }
        // Log texture info if available
        if let Some(ref texture_data) = attributes.texture {
            log::info!(
                "Texture loaded: {}x{}x{}",
                texture_data.height_f32,
                texture_data.width_f32,
                texture_data.channels
            );
        }
    } else {
        // ProjectionMode::ObjOnly
        // We don't need texture or UVs.
        // Check for vertex colors (rasterizer handles default if None)
        if attributes.vertex_colors.is_none() {
            log::warn!(
                "OBJ_ONLY mode: Vertex colors not provided. Defaulting to white in rasterizer."
            );
        }
    }

    // Validate normals shape if Some
    if let Some(normals) = attributes.vertex_normals {
        if mesh.vertices.nrows() != normals.nrows() {
            panic!(
                "Inconsistent input: Vertices ({}) != Vertex Normals ({}).",
                mesh.vertices.nrows(),
                normals.nrows()
            );
        }
    } else if settings.mode != ProjectionMode::ObjOnly && material.illum > 0 {
        // Warn if normals are missing but lighting is expected
        log::warn!(
             "Vertex normals not provided for lighting mode {:?} (illum={}). Shading may be flat or incorrect.",
             settings.mode, material.illum
         );
        // We will proceed and let the rasterizer handle None normals (e.g., skip lighting or use default normal)
    }

    // --- Precomputation ---
    log::info!("Precomputing triangle normals...");
    let triangle_normals = precompute_triangle_normals(mesh.vertices, mesh.triangles);
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
                let num_faces = mesh.triangles.nrows();

                for t_idx in 0..num_faces {
                    rasterize_triangle(
                        t_idx,
                        mesh.vertices,
                        mesh.triangles,
                        attributes.triangle_uvs,
                        &triangle_normals,
                        attributes.vertex_normals,
                        attributes.vertex_colors,
                        material,
                        attributes.texture.as_ref(),
                        &projection_config,
                        &mut z_slice.view_mut(),
                        &mut image_slice.view_mut(),
                        settings.mode,
                    );
                }
            });
    });
    log::info!("...Rasterization finished.");

    // Apply outlier filtering if enabled
    let outlier_counts = if settings.filtering_window_radius > 0 {
        log::info!(
            "Filtering outliers with window radius {}...",
            settings.filtering_window_radius
        );
        filter_outliers(
            &mut buffers.z_buffers,
            &mut buffers.images,
            settings.filtering_window_radius,
            max_bound_f32,
        )
    } else {
        log::info!("Outlier filtering disabled.");
        [0; NUM_IMAGES]
    };
    log::info!("...Processing complete.");

    (buffers.images, outlier_counts)
}
