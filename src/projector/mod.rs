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
use rasterization::rasterize_triangle;
use rayon::prelude::*;
use texture::TextureData;

/// Computes 6 orthographic projections (front/back for X/Y/Z) of a textured mesh.
///
/// Args:
/// * `points`: Vertex coordinates as a 2D array view (N, 3), where N is the number of vertices.
/// * `triangles`: Triangle indices as a 2D array view (M, 3), where M is the number of triangles.
///                Each row contains indices into the `points` array defining a triangle.
/// * `texture`: Texture image data as a 3D array view (H, W, C), where H/W are height/width,
///              and C is the number of color channels.
/// * `triangle_uvs`: UV coordinates corresponding to each vertex *within each triangle*.
///                   This is a 2D array view (M * 3, 2). The order must match the expanded
///                   vertex order implicit in `triangles` (i.e., UVs for tri 0 vertex 0,
///                   tri 0 vertex 1, tri 0 vertex 2, tri 1 vertex 0, ...).
/// * `precision`: A power-of-2 exponent determining the output image resolution.
///                The output images will be `(2^precision) x (2^precision)`.
/// * `filtering_window_radius`: Radius of the square neighborhood window for outlier filtering.
///                              A value of 0 disables the filtering step.
/// * `logging_enabled`: Flag to enable or disable conditional logging messages.
///
/// Returns:
/// A tuple containing:
/// * `images`: An `Array4<u8>` holding the 6 rendered projection images (NUM_IMAGES, H, W, C).
/// * `ocp_maps`: An `Array3<f32>` holding the 6 occlusion maps (NUM_IMAGES, H, W),
///               where 1.0 indicates a mesh pixel and 0.0 indicates background.
/// * `freqs`: An array `[u64; NUM_IMAGES]` containing the count of filtered outlier pixels
///            for each of the 6 images.
pub fn compute_mesh_projections(
    points: ArrayView2<f32>,       // Vertex positions (N, 3)
    triangles: ArrayView2<usize>,  // Triangle indices (M, 3)
    texture: ArrayView3<u8>,       // Texture data (H, W, C)
    triangle_uvs: ArrayView2<f32>, // UV coords per face-vertex (M*3, 2)
    precision: u32,                // Log2 of output image dimensions
    filtering_window_radius: u32,  // Radius for outlier filter (0=disabled)
    logging_enabled: bool,         // Enable/disable logging
) -> (Array4<u8>, Array3<f32>, [u64; NUM_IMAGES]) {
    conditional_log!(
        logging_enabled,
        info,
        "Initializing optimized projector back-end"
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
    if points.is_empty() || triangles.is_empty() || texture.is_empty() || triangle_uvs.is_empty() {
        conditional_log!(
            logging_enabled,
            warn,
            "Empty input detected (points, triangles, texture, or uvs). Returning empty buffers."
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

    // --- Precomputation ---
    conditional_log!(logging_enabled, info, "Precomputing triangle normals...");
    // Calculate geometric normals for all triangles once upfront.
    let triangle_normals = precompute_triangle_normals(points, triangles);
    conditional_log!(logging_enabled, info, "...Normal precomputation finished.");

    // --- Buffer Initialization ---
    conditional_log!(logging_enabled, info, "Initializing output buffers...");
    // Create the Buffers struct containing initialized images, z-buffers, and occlusion maps.
    let mut buffers = Buffers::new(NUM_IMAGES, rows, cols, channels);
    conditional_log!(logging_enabled, info, "...Buffer initialization finished.");

    // --- Rasterization (Parallel over Views) ---
    conditional_log!(
        logging_enabled,
        info,
        "Starting parallel rasterization loop for {} views...",
        NUM_IMAGES
    );

    // Prepare mutable views into the buffer arrays for parallel processing.
    // Collecting views into Vecs allows using `par_iter_mut` provided by Rayon.
    let mut z_buffer_views: Vec<_> = buffers.z_buffers.axis_iter_mut(Axis(0)).collect();
    let mut image_views: Vec<_> = buffers.images.axis_iter_mut(Axis(0)).collect();
    let mut ocp_map_views: Vec<_> = buffers.ocp_maps.axis_iter_mut(Axis(0)).collect();

    // Use Rayon's parallel iterators to process each projection view (image layer) concurrently.
    z_buffer_views
        .par_iter_mut() // Parallel mutable iterator over Z-buffer views
        .zip(image_views.par_iter_mut()) // Combine with parallel iterator over image views
        .zip(ocp_map_views.par_iter_mut()) // Combine with parallel iterator over occlusion map views
        .enumerate() // Add index `img_layer` (0 to NUM_IMAGES - 1)
        .for_each(|(img_layer, ((z_slice, image_slice), ocp_slice))| {
            // For each view/layer:
            // 1. Create the specific ProjectionConfig for this view.
            let projection_config = ProjectionConfig::new(img_layer);
            let num_faces = triangles.nrows();

            // 2. Rasterize all triangles onto this view's buffers.
            for t_idx in 0..num_faces {
                rasterize_triangle(
                    t_idx,              // Current triangle index
                    points,             // All vertex positions
                    triangles,          // All triangle indices
                    triangle_uvs,       // All UV coordinates
                    &triangle_normals,  // Precomputed normals
                    &texture_data,      // Texture info
                    &projection_config, // View-specific config
                    z_slice,            // Mutable Z-buffer slice for this view
                    image_slice,        // Mutable image slice for this view
                    ocp_slice,          // Mutable occlusion map slice for this view
                );
            }
        });
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
