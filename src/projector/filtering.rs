// src/projector/filtering.rs
use crate::projector::constants::{
    BACKGROUND_BRIGHTNESS, FILTERING_NEIGHBORHOOD_FACTOR, NUM_IMAGES,
};
use nd::{Array3, Array4, ArrayView2, ArrayViewMut3, Axis};
use rayon::prelude::*;
use std::f32;

/// Finds outlier pixels by comparing depth to neighbors.
/// Now uses z-buffer validity instead of occlusion map.
fn find_outliers_for_layer(
    z_slice: ArrayView2<f32>, // Z-buffer for the current layer
    w: usize,                 // Filter window radius
    rows: usize,              // Total rows
    cols: usize,              // Total columns
    filter_threshold: f32,    // Depth difference threshold
    is_front_view: bool,      // Flag for view direction
) -> Vec<(usize, usize)> {
    let row_start = w;
    let col_start = w;
    let num_inner_rows = rows.saturating_sub(2 * w);
    let num_inner_cols = cols.saturating_sub(2 * w);

    if num_inner_rows == 0 || num_inner_cols == 0 {
        return Vec::new();
    }

    (0..num_inner_rows)
        .into_par_iter()
        .flat_map(|r_offset| {
            let r = row_start + r_offset;
            let mut row_outlier_coords = Vec::new();

            for c_offset in 0..num_inner_cols {
                let c = col_start + c_offset;

                let center_depth = z_slice[[r, c]];
                // Skip if the center pixel depth is invalid (background).
                if (is_front_view && center_depth == f32::INFINITY)
                    || (!is_front_view && center_depth == f32::NEG_INFINITY)
                {
                    continue;
                }
                // Also skip NaNs just in case
                if !center_depth.is_finite() {
                    continue;
                }

                // Calculate the average depth of *valid* neighbors.
                let mut sum_depth = 0.0;
                let mut count_neighbors = 0.0;
                let r_start_nb = r - w;
                let r_end_nb = r + w + 1;
                let c_start_nb = c - w;
                let c_end_nb = c + w + 1;

                for nr in r_start_nb..r_end_nb {
                    for nc in c_start_nb..c_end_nb {
                        if nr == r && nc == c {
                            continue;
                        }

                        let neighbor_depth = z_slice[[nr, nc]];

                        // Check if neighbor is valid (not background and finite)
                        let is_valid_neighbor = if is_front_view {
                            neighbor_depth < f32::INFINITY && neighbor_depth.is_finite()
                        } else {
                            neighbor_depth > f32::NEG_INFINITY && neighbor_depth.is_finite()
                        };

                        if is_valid_neighbor {
                            sum_depth += neighbor_depth;
                            count_neighbors += 1.0;
                        }
                    }
                }

                // Check for outlier condition
                if count_neighbors > 0.0 {
                    let avg_neighbor_depth = sum_depth / count_neighbors;
                    let depth_diff = center_depth - avg_neighbor_depth;

                    let is_outlier = if is_front_view {
                        depth_diff > filter_threshold
                    } else {
                        depth_diff < -filter_threshold
                    };

                    if is_outlier {
                        row_outlier_coords.push((r, c));
                    }
                }
            }
            row_outlier_coords
        })
        .collect()
}

/// Applies filtering by resetting outlier image pixels to background.
fn apply_outlier_filter(
    outlier_coords: &[(usize, usize)],  // List of outlier coordinates
    mut image_slice: ArrayViewMut3<u8>, // Mutable image view
    channels: usize,                    // Number of channels
    background_color: u8,               // Background color
) {
    if outlier_coords.is_empty() {
        return;
    }
    for &(r, c) in outlier_coords {
        for chan in 0..channels {
            image_slice[[r, c, chan]] = background_color;
        }
    }
}

/// Filters outlier pixels for all image layers. Returns counts of filtered pixels per layer.
pub fn filter_outliers(
    z_buffers: &mut Array3<f32>,  // Mutable z buffers to check
    images: &mut Array4<u8>,      // Mutable images to update
    filtering_window_radius: u32, // Radius 'w' (0 disables)
    max_bound_f32: f32,           // Max dimension for threshold calc
) -> [u64; NUM_IMAGES] {
    // Return outlier counts array
    let mut outlier_counts: [u64; NUM_IMAGES] = [0; NUM_IMAGES];

    if filtering_window_radius == 0 {
        return outlier_counts;
    }

    let w = filtering_window_radius as usize;
    let rows = z_buffers.shape()[1];
    let cols = z_buffers.shape()[2];
    let channels = images.shape()[3];

    if rows <= 2 * w || cols <= 2 * w {
        return outlier_counts;
    }

    let filter_threshold = FILTERING_NEIGHBORHOOD_FACTOR * max_bound_f32;

    for k in 0..NUM_IMAGES {
        let z_slice = z_buffers.index_axis(Axis(0), k);
        let is_front_view = k % 2 == 0;

        let outlier_coords =
            find_outliers_for_layer(z_slice, w, rows, cols, filter_threshold, is_front_view);

        let image_slice = images.index_axis_mut(Axis(0), k);

        apply_outlier_filter(
            &outlier_coords,
            image_slice,
            channels,
            BACKGROUND_BRIGHTNESS,
        );

        outlier_counts[k] = outlier_coords.len() as u64;
    }

    outlier_counts
}
