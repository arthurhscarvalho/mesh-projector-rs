// src/projector/filtering.rs
use crate::projector::buffers::Buffers;
use crate::projector::constants::{
    BACKGROUND_BRIGHTNESS, FILTERING_NEIGHBORHOOD_FACTOR, NUM_CHANNELS,
};
use nd::Array2;
use rayon::prelude::*;
use std::f64;

/// Finds outlier pixels by comparing depth to neighbors.
fn find_outliers(
    z_buffer: &Array2<f64>, // Z-buffer
    w: usize,               // Filter window radius
    rows: usize,            // Total rows
    cols: usize,            // Total columns
    filter_threshold: f64,  // Depth difference threshold
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
                let center_depth = z_buffer[[r, c]];
                // Skip if the center pixel depth is invalid (background).
                if center_depth == f64::INFINITY || center_depth == f64::NEG_INFINITY {
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
                        let neighbor_depth = z_buffer[[nr, nc]];
                        // Check if neighbor is valid (not background and finite)
                        let is_valid_neighbor =
                            neighbor_depth < f64::INFINITY && neighbor_depth.is_finite();
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
                    let is_outlier = depth_diff > filter_threshold;
                    if is_outlier {
                        row_outlier_coords.push((r, c));
                    }
                }
            }
            row_outlier_coords
        })
        .collect()
}

/// Filters outlier pixels for all image layers and applies the filtering to the images.
pub fn filter_outliers(buffers: &mut Buffers, filtering_window_radius: u64, max_bound_f64: f64) {
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
    let filter_threshold = FILTERING_NEIGHBORHOOD_FACTOR * max_bound_f64;
    let outlier_coords = find_outliers(&buffers.z_buffer, w, rows, cols, filter_threshold);
    let outlier_count = outlier_coords.len() as u64;
    // Populate the occlusion map and counts sequentially after parallel computation
    if !outlier_coords.is_empty() {
        for &(r, c) in &outlier_coords {
            buffers.occupancy_map[[r, c]] = 0;
        }
    }
    for r in 0..rows {
        for c in 0..cols {
            // Check the occlusion map's single channel
            if buffers.occupancy_map[[r, c]] == 255 {
                continue;
            }
            // If outlier, set all channels of the image pixel to background
            for chan in 0..NUM_CHANNELS {
                buffers.image[[r, c, chan]] = BACKGROUND_BRIGHTNESS;
            }
        }
    }
    buffers.outlier_count = outlier_count;
}
