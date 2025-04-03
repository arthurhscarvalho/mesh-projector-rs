// src/projector/filtering.rs
use crate::conditional_log;
use crate::projector::buffers::Buffers;
use crate::projector::constants::{FILTERING_NEIGHBORHOOD_FACTOR, NUM_IMAGES};
use nd::{ArrayView2, ArrayViewMut2, ArrayViewMut3, Axis};
use rayon::prelude::*;

/// Finds outlier pixels for a single image layer by comparing each pixel's depth
/// to the average depth of its neighbors within a window. Uses parallel processing over rows.
/// Only considers pixels that are part of the object (ocp > 0) and have finite depth.
fn find_outliers_for_layer(
    z_slice: ArrayView2<f32>,        // Z-buffer for the current layer
    ocp_orig_slice: ArrayView2<f32>, // *Original* occlusion map (before filtering) for this layer
    w: usize,              // Filter window radius (half-width, window size is (2w+1)x(2w+1))
    rows: usize,           // Total rows in the image
    cols: usize,           // Total columns in the image
    filter_threshold: f32, // Depth difference threshold to classify an outlier
    is_front_view: bool,   // Flag indicating if this is a front view or back view
) -> Vec<(usize, usize)> {
    // Returns list of (row, col) coordinates of outliers
    // Define the inner region where the filter window fits completely.
    // Avoid processing pixels near the border where the window would go out of bounds.
    let row_start = w;
    let col_start = w;
    let num_inner_rows = rows.saturating_sub(2 * w); // Number of rows in the processable inner region
    let num_inner_cols = cols.saturating_sub(2 * w); // Number of columns in the processable inner region

    if num_inner_rows == 0 || num_inner_cols == 0 {
        // Image is too small for the filter window; return empty list.
        return Vec::new();
    }

    // Parallel iteration over the rows of the inner region.
    (0..num_inner_rows)
        .into_par_iter()
        .flat_map(|r_offset| {
            // r_offset is the row index within the inner region
            let r = row_start + r_offset; // Calculate the absolute row index 'r'
            let mut row_outlier_coords = Vec::new(); // Collect outliers found in this row

            // Iterate through columns of the inner region for the current row 'r'.
            for c_offset in 0..num_inner_cols {
                // c_offset is the column index within the inner region
                let c = col_start + c_offset; // Calculate the absolute column index 'c'

                // Skip pixels that are originally background (not part of the mesh).
                // Using ocp_orig_slice ensures we don't base decisions on already filtered neighbors.
                if ocp_orig_slice[[r, c]] == 0.0 {
                    continue;
                }

                // Get the depth of the center pixel.
                let center_depth = z_slice[[r, c]];
                // Skip if the center pixel depth is invalid (Infinity or NaN).
                if !center_depth.is_finite() {
                    continue;
                }

                // Calculate the average depth of valid neighbors within the (2w+1)x(2w+1) window.
                let mut sum_depth = 0.0;
                let mut count_neighbors = 0.0;
                let r_start_nb = r - w; // Neighbor window start row
                let r_end_nb = r + w + 1; // Neighbor window end row (exclusive)
                let c_start_nb = c - w; // Neighbor window start col
                let c_end_nb = c + w + 1; // Neighbor window end col (exclusive)

                // Iterate through the neighborhood window.
                for nr in r_start_nb..r_end_nb {
                    for nc in c_start_nb..c_end_nb {
                        // Skip the center pixel itself.
                        if nr == r && nc == c {
                            continue;
                        }

                        // Check if the neighbor pixel is part of the *original* object.
                        if ocp_orig_slice[[nr, nc]] > 0.0 {
                            let neighbor_depth = z_slice[[nr, nc]];
                            // Include neighbor only if its depth is finite.
                            if neighbor_depth.is_finite() {
                                sum_depth += neighbor_depth;
                                count_neighbors += 1.0;
                            }
                        }
                    }
                }

                // Check for the outlier condition if there were valid neighbors.
                if count_neighbors > 0.0 {
                    let avg_neighbor_depth = sum_depth / count_neighbors;
                    let depth_diff = center_depth - avg_neighbor_depth;

                    // Determine if the pixel is an outlier based on the depth difference and view type.
                    // An outlier is significantly further away (larger depth difference) than its neighbors.
                    // Front view: depth increases away from camera. Outlier if center_depth is much larger than avg. -> depth_diff > threshold
                    // Back view: depth decreases (becomes more negative) away from camera. Outlier if center_depth is much smaller than avg. -> depth_diff < -threshold
                    let is_outlier = if is_front_view {
                        depth_diff > filter_threshold
                    } else {
                        depth_diff < -filter_threshold
                    };

                    if is_outlier {
                        // If it's an outlier, add its coordinates to the list for this row.
                        row_outlier_coords.push((r, c));
                    }
                }
            }
            // Return the list of outliers found in this row (for flat_map).
            row_outlier_coords
        })
        .collect() // Collect results from all parallel row computations into a single Vec.
}

/// Applies the filtering results by resetting outlier pixels in the image and occlusion buffers.
/// Sets outlier pixels to background color and marks them as not occluded.
fn apply_outlier_filter(
    outlier_coords: &[(usize, usize)], // List of (row, col) coordinates identified as outliers
    mut image_slice: ArrayViewMut3<u8>, // Mutable view to the image layer buffer
    mut ocp_slice: ArrayViewMut2<f32>, // Mutable view to the occlusion map layer buffer
    channels: usize,                   // Number of color channels
    background_color: u8,              // Background color value to use for reset pixels
) {
    if outlier_coords.is_empty() {
        return; // No outliers to filter for this layer.
    }
    // Iterate through the identified outlier coordinates.
    for &(r, c) in outlier_coords {
        // Reset the occlusion map value for this pixel to 0.0 (background).
        ocp_slice[[r, c]] = 0.0;
        // Reset the color of the pixel to the background color for all channels.
        for chan in 0..channels {
            image_slice[[r, c, chan]] = background_color;
        }
    }
}

/// Filters outlier pixels for all image layers based on depth differences in a local neighborhood.
/// Outliers are pixels whose depth significantly deviates from the average depth of their neighbors.
/// Modifies the `images` and `ocp_maps` fields of the `buffers` struct in place.
///
/// Returns an array containing the count of filtered outlier pixels for each of the `NUM_IMAGES` layers.
pub fn filter_outliers(
    buffers: &mut Buffers,        // Mutable reference to the output buffers
    filtering_window_radius: u32, // Radius 'w' of the neighborhood window (0 to disable filtering)
    rows: usize,                  // Number of rows in the images/buffers
    cols: usize,                  // Number of columns in the images/buffers
    max_bound_f32: f32,           // Maximum dimension used to calculate the filter threshold
    channels: usize,              // Number of color channels in the image buffer
    background_color: u8,         // Background color value (typically BACKGROUND_BRIGHTNESS)
    logging_enabled: bool,        // Flag to enable/disable conditional logging
) -> [u64; NUM_IMAGES] {
    // Returns counts of outliers per image layer
    let mut outlier_counts: [u64; NUM_IMAGES] = [0; NUM_IMAGES]; // Initialize counts to zero

    // Skip filtering entirely if the radius is zero.
    if filtering_window_radius == 0 {
        conditional_log!(
            logging_enabled,
            info,
            "Filtering skipped: window radius is zero."
        );
        return outlier_counts;
    }

    let w = filtering_window_radius as usize; // Convert radius to usize for indexing

    /*
    Skip filtering if the image dimensions are too small to accommodate the filter window.
    The window size is (2w+1)x(2w+1), so dimensions must be at least 2w+1.
    We check > 2*w which is equivalent to >= 2*w + 1.
     */
    if rows <= 2 * w || cols <= 2 * w {
        conditional_log!(
            logging_enabled,
            warn, // Use warn as this might indicate an issue or unexpected configuration
            "Filtering skipped: Image dimensions ({}x{}) too small for filter window radius {}.",
            rows,
            cols,
            w
        );
        return outlier_counts;
    }

    conditional_log!(
        logging_enabled,
        info,
        "Starting outlier filtering loop (Window Radius: {})",
        w
    );
    // Calculate the depth difference threshold based on the image scale.
    let filter_threshold = FILTERING_NEIGHBORHOOD_FACTOR * max_bound_f32;

    // Process each image layer (view) independently.
    for k in 0..NUM_IMAGES {
        // IMPORTANT: Create a copy of the original occlusion map slice for this layer *before* applying filtering.
        // The outlier detection logic needs to know which neighbors were *originally* part of the object,
        // not influenced by filtering applied to previous pixels/neighbors in the same pass.
        let ocp_orig_slice = buffers.ocp_maps.index_axis(Axis(0), k).to_owned();

        // Get an immutable view of the Z-buffer for this layer.
        let z_slice = buffers.z_buffers.index_axis(Axis(0), k);
        // Determine if the current layer corresponds to a front view (even index).
        let is_front_view = k % 2 == 0;

        // Find all outlier coordinates for the current layer using parallel processing.
        let outlier_coords = find_outliers_for_layer(
            z_slice,               // Z-buffer view
            ocp_orig_slice.view(), // Pass immutable view of the *original* occlusion map
            w,                     // Window radius
            rows,                  // Image rows
            cols,                  // Image columns
            filter_threshold,      // Depth difference threshold
            is_front_view,         // View type flag
        );

        // Get mutable views to the image and occlusion map slices for this layer.
        let image_slice = buffers.images.index_axis_mut(Axis(0), k);
        let ocp_slice = buffers.ocp_maps.index_axis_mut(Axis(0), k);

        // Apply the filter: Reset the identified outlier pixels in the buffers.
        apply_outlier_filter(
            &outlier_coords,  // List of outlier coordinates
            image_slice,      // Mutable image slice
            ocp_slice,        // Mutable occlusion map slice
            channels,         // Number of channels
            background_color, // Background color value
        );

        // Store the number of outliers found and filtered for this layer.
        outlier_counts[k] = outlier_coords.len() as u64;
    }

    conditional_log!(logging_enabled, info, "Filtering finished");
    outlier_counts // Return the array of outlier counts per layer
}
