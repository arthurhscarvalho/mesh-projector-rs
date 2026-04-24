//! This module provides functions for outlier filtering in projected images,
//! specifically targeting pixels whose depth values significantly deviate from
//! their local neighborhood.

use crate::renderer::buffers::ImageBuffers;
use mesh_projector_core::config::ProjectionConfig;
use mesh_projector_core::constants::FILTERING_NEIGHBORHOOD_FACTOR;
use nd::Array2;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::f32;

/// Finds outlier pixels by comparing depth to neighbors.
/// This implementation uses integral images to compute the neighbourhood sum and count in
/// constant time per pixel, reducing the overall complexity from O(N * w^2) to O(N).
fn find_outliers(
    z_buffer: &Array2<f32>, // Z-buffer
    w: usize,               // Filter window radius
    rows: usize,            // Total rows
    cols: usize,            // Total columns
    filter_threshold: f32,  // Depth difference threshold
) -> Vec<(usize, usize)> {
    if rows <= 2 * w || cols <= 2 * w {
        return Vec::new();
    }

    // Build two integral images: one for the depth sum of valid pixels and one for the count
    // of valid pixels. Using `f32` for the count keeps the math homogeneous.
    let mut integral_sum = Array2::<f32>::zeros((rows, cols));
    let mut integral_count = Array2::<f32>::zeros((rows, cols));

    // First pass: populate integral images.
    for r in 0..rows {
        let mut row_sum = 0.0;
        let mut row_count = 0.0;
        for c in 0..cols {
            let val = z_buffer[[r, c]];
            let is_valid = val.is_finite() && val < f32::INFINITY;
            if is_valid {
                row_sum += val;
                row_count += 1.0;
            }
            let above_sum = if r > 0 { integral_sum[[r - 1, c]] } else { 0.0 };
            let above_cnt = if r > 0 {
                integral_count[[r - 1, c]]
            } else {
                0.0
            };
            integral_sum[[r, c]] = row_sum + above_sum;
            integral_count[[r, c]] = row_count + above_cnt;
        }
    }

    // Helper closure to query a rectangle sum in O(1) using the integral image.
    let rect_stats = |sum_img: &Array2<f32>, r0: usize, c0: usize, r1: usize, c1: usize| -> f32 {
        let mut result = sum_img[[r1, c1]];
        if r0 > 0 {
            result -= sum_img[[r0 - 1, c1]];
        }
        if c0 > 0 {
            result -= sum_img[[r1, c0 - 1]];
        }
        if r0 > 0 && c0 > 0 {
            result += sum_img[[r0 - 1, c0 - 1]];
        }
        result
    };

    // Scan through the inner pixels in parallel to find outliers.
    let inner_rows = (w..rows - w).collect::<Vec<_>>();
    inner_rows
        .into_par_iter()
        .flat_map(|r| {
            let mut row_outliers = Vec::new();
            for c in w..cols - w {
                let center_depth = z_buffer[[r, c]];
                if !center_depth.is_finite() || center_depth == f32::INFINITY {
                    continue;
                }
                // Rectangle bounds for the neighbourhood (inclusive indices)
                let r0 = r - w;
                let c0 = c - w;
                let r1 = r + w;
                let c1 = c + w;

                // Sum and count of depths in the neighbourhood
                let sum_depth = rect_stats(&integral_sum, r0, c0, r1, c1) - center_depth;
                let count_neighbors = rect_stats(&integral_count, r0, c0, r1, c1) - 1.0; // exclude centre

                if count_neighbors > 0.0 {
                    let avg_depth = sum_depth / count_neighbors;
                    if center_depth - avg_depth > filter_threshold {
                        row_outliers.push((r, c));
                    }
                }
            }
            row_outliers
        })
        .collect()
}

/// Filters outlier pixels for all image layers and applies the filtering to the images.
pub fn filter_outliers(buffers: &mut ImageBuffers, settings: &ProjectionConfig, max_bound: f32) {
    let filtering_window_radius = settings.filtering_window_radius;
    // Return outlier counts array
    let rows = buffers.z_buffer.shape()[0];
    let cols = buffers.z_buffer.shape()[1];
    if filtering_window_radius == 0 {
        log::info!("Outlier filtering disabled.");
        return;
    };
    log::info!(
        "Filtering outliers with window radius {}",
        filtering_window_radius
    );
    let w = filtering_window_radius as usize;
    if rows <= 2 * w || cols <= 2 * w {
        return;
    }
    let filter_threshold = FILTERING_NEIGHBORHOOD_FACTOR * max_bound;
    let outlier_coords = find_outliers(&buffers.z_buffer, w, rows, cols, filter_threshold);
    let outlier_count = outlier_coords.len() as u64;
    // Populate the occlusion map and counts sequentially after parallel computation
    if !outlier_coords.is_empty() {
        for &(r, c) in &outlier_coords {
            buffers.occupancy_map[[r, c]] = 0;
        }
    }
    let image_channels = buffers.image.shape()[2];
    let background_color = buffers.background_color;
    let fallback_background = *background_color.last().unwrap_or(&0);
    for r in 0..rows {
        for c in 0..cols {
            // Check the occlusion map's single channel
            if buffers.occupancy_map[[r, c]] == 255 {
                continue;
            }
            // If outlier, set all channels of the image pixel to background
            for chan in 0..image_channels {
                let bg_val = background_color
                    .get(chan)
                    .copied()
                    .unwrap_or(fallback_background);
                buffers.image[[r, c, chan]] = bg_val;
            }
        }
    }
    buffers.outlier_count = outlier_count;
}
