// src/projector/filtering.rs
use crate::projector::constants::{
    BACKGROUND_BRIGHTNESS, FILTERING_NEIGHBORHOOD_FACTOR, NUM_CHANNELS, NUM_IMAGES,
};
use nd::{Array3, Array4, ArrayView2, Axis, Ix4};
use rayon::prelude::*;
use std::f64;

/// Finds outlier pixels by comparing depth to neighbors.
/// Now uses z-buffer validity instead of occlusion map.
fn find_outliers_for_layer(
    z_slice: ArrayView2<f64>, // Z-buffer for the current layer
    w: usize,                 // Filter window radius
    rows: usize,              // Total rows
    cols: usize,              // Total columns
    filter_threshold: f64,    // Depth difference threshold
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
                if (is_front_view && center_depth == f64::INFINITY)
                    || (!is_front_view && center_depth == f64::NEG_INFINITY)
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
                            neighbor_depth < f64::INFINITY && neighbor_depth.is_finite()
                        } else {
                            neighbor_depth > f64::NEG_INFINITY && neighbor_depth.is_finite()
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

/// Filters outlier pixels for all image layers and applies the filtering to the images.
/// Returns a tuple containing:
/// 1. An occlusion map (Array4<u8>, shape=[NUM_IMAGES, rows, cols, 1]) where 1 indicates an outlier.
/// 2. Counts of filtered pixels per layer ([u64; NUM_IMAGES]).
pub fn filter_outliers(
    images: &mut Array4<u8>,
    z_buffers: &Array3<f64>,
    filtering_window_radius: u64,
    max_bound_f64: f64,
) -> (Array4<u8>, [u64; NUM_IMAGES]) {
    // Return outlier counts array
    let rows = z_buffers.shape()[1];
    let cols = z_buffers.shape()[2];
    if filtering_window_radius == 0 {
        log::info!("Outlier filtering disabled.");
        let shape4d_occ = Ix4(NUM_IMAGES, rows, cols, 1);
        return (Array4::zeros(shape4d_occ), [0; NUM_IMAGES]);
    };
    log::info!(
        "Filtering outliers with window radius {}",
        filtering_window_radius
    );
    // Initialize occlusion map (NUM_IMAGES, rows, cols, 1 channel)
    let mut outlier_counts: [u64; NUM_IMAGES] = [0; NUM_IMAGES];
    let mut occlusion_maps = Array4::<u8>::zeros(Ix4(NUM_IMAGES, rows, cols, 1));
    let w = filtering_window_radius as usize;
    if rows <= 2 * w || cols <= 2 * w {
        return (occlusion_maps, outlier_counts); // Return empty map and zero counts
    }
    let filter_threshold = FILTERING_NEIGHBORHOOD_FACTOR * max_bound_f64;
    // Collect results from parallel processing
    let results: Vec<_> = (0..NUM_IMAGES)
        .into_par_iter()
        .map(|k| {
            let z_slice = z_buffers.index_axis(Axis(0), k);
            let is_front_view = k % 2 == 0;
            let outlier_coords =
                find_outliers_for_layer(z_slice, w, rows, cols, filter_threshold, is_front_view);
            let count = outlier_coords.len() as u64;
            (k, outlier_coords, count) // Return index, coords, and count
        })
        .collect();
    // Populate the occlusion map and counts sequentially after parallel computation
    for (k, outlier_coords, count) in results {
        outlier_counts[k] = count;
        if !outlier_coords.is_empty() {
            let mut occlusion_slice = occlusion_maps.index_axis_mut(Axis(0), k);
            for &(r, c) in &outlier_coords {
                // Set the single channel to 1 for outliers
                occlusion_slice[[r, c, 0]] = 1;
            }
        }
    }
    // Apply filtering to images using the occlusion map
    images
        .axis_iter_mut(Axis(0))
        .zip(occlusion_maps.axis_iter(Axis(0)))
        .par_bridge() // Parallelize the application
        .for_each(|(mut img_slice, occ_slice)| {
            // img_slice shape: (rows, cols, channels)
            // occ_slice shape: (rows, cols, 1)
            for r in 0..rows {
                for c in 0..cols {
                    // Check the occlusion map's single channel
                    if occ_slice[[r, c, 0]] == 0 {
                        continue;
                    }
                    // If outlier, set all channels of the image pixel to background
                    for chan in 0..NUM_CHANNELS {
                        img_slice[[r, c, chan]] = BACKGROUND_BRIGHTNESS;
                    }
                }
            }
        });
    (occlusion_maps, outlier_counts)
}
