// src/projector/mod.rs
mod buffers;
mod config;
mod constants;
mod filtering;
mod geometry;
mod rasterization;
mod texture;

use crate::conditional_log;
use buffers::Buffers;
use config::ProjectionConfig;
use constants::{BACKGROUND_BRIGHTNESS, NUM_IMAGES};
use filtering::filter_outliers;
use geometry::precompute_triangle_normals;
use nd::{Array3, Array4, ArrayView2, ArrayView3, Axis, Ix3, Ix4};
use rasterization::{rasterize_triangle, MaterialProperties};
use rayon::prelude::*;
use rayon::scope;
use texture::TextureData;

/// Computes 6 orthographic projections (front/back for X/Y/Z) of a textured mesh.
///
/// Args:
/// * `points`: Vertex coordinates as a 2D array view (N, 3).
/// * `triangles`: Triangle indices as a 2D array view (M, 3) using usize.
/// * `vertex_normals`: Vertex normal vectors as a 2D array view (N, 3).
/// * `texture`: Texture image data as a 3D array view (H, W, C).
/// * `triangle_uvs`: UV coordinates corresponding to each vertex *within each triangle* (M * 3, 2).
/// * `precision`: A power-of-2 exponent determining the output image resolution.
///                The output images will be `(2^precision) x (2^precision)`.
/// * `ka`: Ambient color.
/// * `kd`: Diffuse color.
/// * `ks`: Specular color.
/// * `ns`: Shininess (specular exponent).
/// * `opacity`: Opacity (0.0=transparent, 1.0=opaque).
/// * `illum`: Illumination model.
/// * `filtering_window_radius`: Radius of the square neighborhood window for outlier filtering.
///                              A value of 0 disables the filtering step.
/// * `logging_enabled`: Flag to enable or disable conditional logging messages.
///
/// Returns:
/// Tuple: (images: Array4<u8>, ocp_maps: Array3<f32>, outlier_counts: [u64; NUM_IMAGES])
pub fn compute_mesh_projections(
    points: ArrayView2<f32>,         // Vertex positions (N, 3)
    triangles: ArrayView2<usize>,    // Triangle indices (M, 3) -> usize
    vertex_normals: ArrayView2<f32>, // Vertex normals (N, 3)
    texture: ArrayView3<u8>,         // Texture data (H, W, C)
    triangle_uvs: ArrayView2<f32>,   // UV coords per face-vertex (M*3, 2)
    precision: u32,                  // Log2 of output image dimensions
    ka: [f32; 3],                    // Ambient color
    kd: [f32; 3],                    // Diffuse color
    ks: [f32; 3],                    // Specular color
    ns: f32,                         // Shininess (specular exponent)
    opacity: f32,                    // Opacity (0.0=transparent, 1.0=opaque)
    illum: u32,                      // Illumination model
    filtering_window_radius: u32,    // Radius for outlier filter (0=disabled)
    logging_enabled: bool,           // Enable/disable logging
) -> (Array4<u8>, Array3<f32>, [u64; NUM_IMAGES]) {
    conditional_log!(
        logging_enabled,
        info,
        "Initializing optimized projector back-end (with vertex normals)"
    );

    // --- Setup & Validation ---
    // Calculate image dimensions based on precision.
    let max_bound: u64 = 1 << precision; // e.g., precision 8 -> max_bound 256
    let max_bound_f32: f32 = max_bound as f32; // Dimension as float for calculations
    let max_bound_u: usize = max_bound as usize; // Dimension as usize for indexing
    let rows = max_bound_u; // Image height
    let cols = max_bound_u; // Image width

    // Wrap texture view in TextureData struct for easier access to properties.
    let texture_data = TextureData::new(texture);
    let channels = texture_data.channels; // Number of color channels

    // Handle potential empty inputs gracefully by returning empty/default buffers.
    if points.is_empty()
        || triangles.is_empty()
        || vertex_normals.is_empty()
        || texture.is_empty()
        || triangle_uvs.is_empty()
    {
        conditional_log!(
            logging_enabled,
            warn,
            "Empty input detected (points, triangles, normals, texture, or uvs). Returning empty buffers."
        );
        // Define shapes for empty arrays.
        let shape4d = Ix4(NUM_IMAGES, rows, cols, channels);
        let shape3d = Ix3(NUM_IMAGES, rows, cols);
        // Return background images, zeroed occlusion maps, and zero outlier counts.
        return (
            Array4::from_elem(shape4d, BACKGROUND_BRIGHTNESS),
            Array3::zeros(shape3d),
            [0; NUM_IMAGES],
        );
    }

    // Validate consistency between triangle count and UV coordinate count.
    // Each triangle has 3 vertices, so we need 3 UV pairs per triangle.
    let expected_uv_rows = triangles.nrows() * 3;
    if expected_uv_rows != triangle_uvs.nrows() {
        // Mismatched input sizes indicate a problem. Panic as in original code.
        // For a library, returning a Result<..., Error> might be preferable.
        panic!(
            "Inconsistent input: Number of triangles ({}) * 3 != number of triangle UVs ({}). Expected {} UV coordinates.",
            triangles.nrows(),
            triangle_uvs.nrows(),
            expected_uv_rows
        );
    }

    // Validate vertex count consistency
    if points.nrows() != vertex_normals.nrows() {
        panic!(
            "Inconsistent input: Number of points ({}) != number of vertex normals ({}).",
            points.nrows(),
            vertex_normals.nrows()
        );
    }

    // --- Create Material Properties Struct ---
    let material = MaterialProperties {
        ka,
        kd,
        ks,
        ns,
        opacity,
        illum,
    }; // Use passed values

    // --- Precomputation ---
    conditional_log!(
        logging_enabled,
        info,
        "Precomputing triangle normals (for culling)..."
    );
    let triangle_normals = precompute_triangle_normals(points, triangles);
    conditional_log!(
        logging_enabled,
        info,
        "...Triangle normal precomputation finished."
    );

    // --- Buffer Initialization ---
    conditional_log!(logging_enabled, info, "Initializing output buffers...");
    let mut buffers = Buffers::new(NUM_IMAGES, rows, cols, channels);
    conditional_log!(logging_enabled, info, "...Buffer initialization finished.");

    // --- Rasterization (Parallel over Views within rayon::scope) ---
    conditional_log!(
        logging_enabled,
        info,
        "Starting parallel rasterization loop for {} views...",
        NUM_IMAGES
    );

    let mut z_buffer_views: Vec<_> = buffers.z_buffers.axis_iter_mut(Axis(0)).collect();
    let mut image_views: Vec<_> = buffers.images.axis_iter_mut(Axis(0)).collect();
    let mut ocp_map_views: Vec<_> = buffers.ocp_maps.axis_iter_mut(Axis(0)).collect();

    // Use rayon::scope
    scope(|_| {
        z_buffer_views
            .par_iter_mut()
            .zip(image_views.par_iter_mut())
            .zip(ocp_map_views.par_iter_mut())
            .enumerate()
            .for_each(|(img_layer, ((z_slice, image_slice), ocp_slice))| {
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
                        &material,
                        Some(&texture_data),
                        &projection_config,
                        z_slice,
                        image_slice,
                        ocp_slice,
                    );
                }
            });
    }); // End of rayon::scope

    conditional_log!(logging_enabled, info, "...Rasterization finished.");

    // --- Post-processing (Outlier Filtering) ---
    conditional_log!(logging_enabled, info, "Starting outlier filtering...");
    // Apply the outlier filter to the generated buffers (modifies images and ocp_maps in place).
    let outlier_counts = filter_outliers(
        &mut buffers,            // Pass mutable reference to the buffers
        filtering_window_radius, // User-specified filter radius
        rows,                    // Image height
        cols,                    // Image width
        max_bound_f32,           // Image dimension as float (for threshold calculation)
        channels,                // Number of color channels
        BACKGROUND_BRIGHTNESS,   // Background color value
        logging_enabled,         // Logging flag
    );
    conditional_log!(logging_enabled, info, "...Filtering finished.");

    // --- Return Results ---
    conditional_log!(logging_enabled, info, "Projection computation complete.");
    // Return the final projected images, occlusion maps, and the counts of filtered pixels per image.
    (buffers.images, buffers.ocp_maps, outlier_counts)
}
