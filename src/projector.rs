use crate::conditional_log;
use nd::{s, Array3, Array4, ArrayView2, ArrayView3, Axis, Ix3, Ix4};
use rayon::prelude::*;
use std::f64;

// --- barycentric function remains the same ---
/// Helper: computes barycentric coordinates of point (px,py) with respect to triangle vertices (ax,ay), (bx,by), (cx,cy).
/// Returns (u,v,w) such that p = u*a + v*b + w*c and u+v+w = 1.
/// Returns None if the triangle area is nearly zero (degenerate).
#[inline]
fn barycentric(
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    cx: f64,
    cy: f64,
    px: f64,
    py: f64,
) -> Option<(f64, f64, f64)> {
    // Vector approach: v0 = b - a, v1 = c - a, v2 = p - a
    let v0x = bx - ax;
    let v0y = by - ay;
    let v1x = cx - ax;
    let v1y = cy - ay;
    let v2x = px - ax;
    let v2y = py - ay;
    // Calculate dot products
    let d00 = v0x * v0x + v0y * v0y; // dot(v0, v0)
    let d01 = v0x * v1x + v0y * v1y; // dot(v0, v1)
    let d11 = v1x * v1x + v1y * v1y; // dot(v1, v1)
    let d20 = v2x * v0x + v2y * v0y; // dot(v2, v0)
    let d21 = v2x * v1x + v2y * v1y; // dot(v2, v1)
                                     // Calculate denominator (proportional to triangle area * 2)
    let denom = d00 * d11 - d01 * d01;
    // Check for degenerate triangle
    const BARYCENTRIC_EPSILON: f64 = 1e-9;
    if denom.abs() < BARYCENTRIC_EPSILON {
        return None;
    }
    // Calculate barycentric coordinates v and w
    let inv_denom = 1.0 / denom;
    let v = (d11 * d20 - d01 * d21) * inv_denom;
    let w = (d00 * d21 - d01 * d20) * inv_denom;
    // Calculate u from u + v + w = 1
    let u = 1.0 - v - w;
    Some((u, v, w))
}

/// Compute mesh projections onto the 6 faces of a bounding cube using a Z-buffer,
/// per-pixel texture mapping, and back-face culling. Parallelized implementation.
/// Corrected for mutable borrow errors.
///
/// # Arguments
/// * `points`: An (N x 3) array view of 3D point coordinates (f64).
/// * `triangles`: An (M x 3) array view of indices (usize) into `points`.
/// * `texture`: An (TexH x TexW x C) array view representing the texture image.
/// * `triangle_uvs`: An (M*3 x 2) array view of UV coordinates (f64).
/// * `precision`: Log2 of the side length of the output images.
/// * `filtering`: Size of the square neighborhood window for outlier filtering. 0 disables filtering.
/// * `logging_enabled`: Flag to enable/disable conditional logging.
///
/// # Returns
/// A tuple containing:
/// * `images`: An `Array4<u8>` of shape (6, H, W, C).
/// * `ocp_maps`: An `Array3<f64>` of shape (6, H, W).
/// * `freqs`: An array `[u64; 6]` counting filtered outliers per view.
pub fn compute_mesh_projections(
    points: ArrayView2<f64>,
    triangles: ArrayView2<usize>,
    texture: ArrayView3<u8>,
    triangle_uvs: ArrayView2<f64>,
    precision: u32,
    filtering: u32,
    logging_enabled: bool,
) -> (Array4<u8>, Array3<f64>, [u64; 6]) {
    conditional_log!(
        logging_enabled,
        info,
        "Initializing optimized projector back-end"
    );

    // --- Initial Setup (mostly unchanged) ---
    let max_bound: u64 = 1 << precision;
    let max_bound_f64: f64 = max_bound as f64;
    let max_bound_u: usize = max_bound as usize;
    let rows = max_bound_u;
    let cols = max_bound_u;

    let tex_height = texture.shape()[0];
    let tex_width = texture.shape()[1];
    let channels = texture.shape()[2];
    let tex_height_f64 = tex_height as f64;
    let tex_width_f64 = tex_width as f64;

    let num_images: usize = 6;
    let initial_color_val: u8 = 255;
    const BARYCENTRIC_COORD_EPSILON: f64 = -1e-9;

    if points.is_empty() || triangles.is_empty() || texture.is_empty() || triangle_uvs.is_empty() {
        conditional_log!(
            logging_enabled,
            warn,
            "Empty input detected (points, triangles, texture, or uvs)."
        );
        let shape4d = Ix4(num_images, rows, cols, channels);
        let shape3d = Ix3(num_images, rows, cols);
        return (
            Array4::from_elem(shape4d, initial_color_val),
            Array3::zeros(shape3d),
            [0; 6],
        );
    }

    if triangles.nrows() * 3 != triangle_uvs.nrows() {
        panic!(
            "Inconsistent input: Number of triangle vertices ({}) != number of triangle UVs ({}).",
            triangles.nrows() * 3,
            triangle_uvs.nrows()
        );
    }

    let mut images = Array4::from_elem((num_images, rows, cols, channels), initial_color_val);
    let mut ocp_maps = Array3::zeros((num_images, rows, cols));
    let mut z_buffers = Array3::from_elem((num_images, rows, cols), f64::INFINITY);
    for i in (1..num_images).step_by(2) {
        z_buffers.slice_mut(s![i, .., ..]).fill(f64::NEG_INFINITY);
    }

    let projection_plane_axes: [(usize, usize); 3] = [(1, 2), (0, 2), (0, 1)];
    let num_faces = triangles.nrows();

    conditional_log!(
        logging_enabled,
        info,
        "Starting parallel rasterization loop"
    );

    // --- Parallel Rasterization Loop (Corrected) ---
    // Explicitly collect mutable views for each layer BEFORE the parallel loop.
    // These Vecs hold the views; the parallel iterator will borrow them mutably.
    let mut z_buffer_views: Vec<_> = z_buffers.axis_iter_mut(Axis(0)).collect();
    let mut image_views: Vec<_> = images.axis_iter_mut(Axis(0)).collect();
    let mut ocp_map_views: Vec<_> = ocp_maps.axis_iter_mut(Axis(0)).collect();

    // Use Rayon's parallel iterators on the collected views.
    // Zip them together and enumerate to get the img_layer index.
    z_buffer_views
        .par_iter_mut() // Parallel iterator over mutable z-buffer views
        .zip(image_views.par_iter_mut()) // Zip with image views
        .zip(ocp_map_views.par_iter_mut()) // Zip with ocp map views
        .enumerate() // Get the index (img_layer)
        .for_each(|(img_layer, ((z_slice, image_slice), ocp_slice))| {
            // Inside this closure, we have exclusive mutable access to the slices
            // corresponding to the current img_layer.

            let depth_axis = img_layer / 2;
            let view_type = img_layer % 2;
            let is_front_view = view_type == 0;

            let view_dir = match (depth_axis, view_type) {
                (0, 0) => (1.0, 0.0, 0.0),
                (0, 1) => (-1.0, 0.0, 0.0),
                (1, 0) => (0.0, 1.0, 0.0),
                (1, 1) => (0.0, -1.0, 0.0),
                (2, 0) => (0.0, 0.0, 1.0),
                (2, 1) => (0.0, 0.0, -1.0),
                _ => unreachable!(),
            };

            let (p_axis, q_axis) = projection_plane_axes[depth_axis];

            // Loop through all triangles FOR THIS VIEW
            for t_idx in 0..num_faces {
                let i0 = triangles[[t_idx, 0]];
                let i1 = triangles[[t_idx, 1]];
                let i2 = triangles[[t_idx, 2]];

                // Use .get() for potentially slightly safer access if indices *could* be invalid
                // Or stick with indexing assuming valid inputs for performance.
                let p0 = points.row(i0);
                let p1 = points.row(i1);
                let p2 = points.row(i2);

                let v0x = p1[0] - p0[0];
                let v0y = p1[1] - p0[1];
                let v0z = p1[2] - p0[2];
                let v1x = p2[0] - p0[0];
                let v1y = p2[1] - p0[1];
                let v1z = p2[2] - p0[2];
                let normal_x = v0y * v1z - v0z * v1y;
                let normal_y = v0z * v1x - v0x * v1z;
                let normal_z = v0x * v1y - v0y * v1x;

                let dot = normal_x * view_dir.0 + normal_y * view_dir.1 + normal_z * view_dir.2;
                if dot >= 0.0 {
                    continue;
                } // Back-face culling

                let uv_idx_base = t_idx * 3;
                let uv0 = triangle_uvs.row(uv_idx_base);
                let uv1 = triangle_uvs.row(uv_idx_base + 1);
                let uv2 = triangle_uvs.row(uv_idx_base + 2);

                let ax_2d = p0[p_axis];
                let ay_2d = p0[q_axis];
                let bx_2d = p1[p_axis];
                let by_2d = p1[q_axis];
                let cx_2d = p2[p_axis];
                let cy_2d = p2[q_axis];

                let da = p0[depth_axis];
                let db = p1[depth_axis];
                let dc = p2[depth_axis];

                let min_col_f = ax_2d.min(bx_2d.min(cx_2d));
                let max_col_f = ax_2d.max(bx_2d.max(cx_2d));
                let min_row_f = ay_2d.min(by_2d.min(cy_2d));
                let max_row_f = ay_2d.max(by_2d.max(cy_2d));

                let min_col = (min_col_f.floor() as i64).max(0) as usize;
                // Correct max bound clamping: should be less than size
                let max_col = (max_col_f.ceil() as i64).min(cols as i64 - 1).max(0) as usize;
                let min_row = (min_row_f.floor() as i64).max(0) as usize;
                let max_row = (max_row_f.ceil() as i64).min(rows as i64 - 1).max(0) as usize;

                if min_col >= cols || min_row >= rows || min_col > max_col || min_row > max_row {
                    continue;
                }

                // Iterate pixels in bbox
                for r in min_row..=max_row {
                    for c in min_col..=max_col {
                        let px = c as f64 + 0.5;
                        let py = r as f64 + 0.5;

                        if let Some((u, v, w)) =
                            barycentric(ax_2d, ay_2d, bx_2d, by_2d, cx_2d, cy_2d, px, py)
                        {
                            if u >= BARYCENTRIC_COORD_EPSILON
                                && v >= BARYCENTRIC_COORD_EPSILON
                                && w >= BARYCENTRIC_COORD_EPSILON
                            {
                                let interpolated_depth = u * da + v * db + w * dc;

                                // Use the mutable SLICES passed into the closure
                                // Indexing is now relative to the slice (2D for z/ocp, 3D for image)
                                let current_z = z_slice[[r, c]]; // Read from slice
                                let depth_test_passed = if is_front_view {
                                    interpolated_depth < current_z
                                } else {
                                    interpolated_depth > current_z
                                };

                                if depth_test_passed {
                                    z_slice[[r, c]] = interpolated_depth; // Write to slice

                                    let interp_u = u * uv0[0] + v * uv1[0] + w * uv2[0];
                                    let interp_v = u * uv0[1] + v * uv1[1] + w * uv2[1];
                                    let tex_x = (interp_u * tex_width_f64)
                                        .round()
                                        .max(0.0)
                                        .min(tex_width_f64 - 1.0)
                                        as usize;
                                    let tex_y = ((1.0 - interp_v) * tex_height_f64) // Assuming V is inverted
                                        .round()
                                        .max(0.0)
                                        .min(tex_height_f64 - 1.0)
                                        as usize;

                                    // Write to image slice (3D view) and ocp slice (2D view)
                                    for chan in 0..channels {
                                        // image_slice is view (H, W, C)
                                        image_slice[[r, c, chan]] = texture[[tex_y, tex_x, chan]];
                                    }
                                    ocp_slice[[r, c]] = 1.0; // ocp_slice is view (H, W)
                                }
                            }
                        }
                    } // end col loop
                } // end row loop
            } // end triangle loop
        }); // end parallel view loop

    conditional_log!(logging_enabled, info, "Rasterization finished");

    // --- Optional Parallel Outlier Filtering Pass (Revised & Corrected) ---
    let mut freqs: [u64; 6] = [0; 6];
    if filtering > 0 && rows > 2 * filtering as usize && cols > 2 * filtering as usize {
        conditional_log!(logging_enabled, info, "Starting parallel filtering loop");
        let w = filtering as usize;
        let filter_threshold = 0.1 * max_bound_f64;
        let ocp_maps_orig = ocp_maps.clone(); // Still need original state for reads

        // We will parallelize the detection per layer, collect coordinates,
        // then mutate sequentially per layer to update maps and freqs safely.
        // Keep the outer loop sequential over k (layers).
        for k in 0..num_images {
            let is_front_view = k % 2 == 0;
            let row_start = w;
            let col_start = w;
            let num_inner_rows = rows.saturating_sub(2 * w);
            let num_inner_cols = cols.saturating_sub(2 * w);

            // Get immutable slices for reading original/z data for this layer
            let z_slice = z_buffers.index_axis(Axis(0), k);
            let ocp_orig_slice = ocp_maps_orig.index_axis(Axis(0), k);

            // --- Pass 1: Detect Outliers in Parallel (for layer k) ---
            // Iterate rows in parallel, collect outlier coords for this layer
            let outliers_coords: Vec<(usize, usize)> = (0..num_inner_rows)
                .into_par_iter()
                .flat_map(|r_offset| {
                    let r = row_start + r_offset;
                    let mut row_outlier_coords = Vec::new();

                    for c_offset in 0..num_inner_cols {
                        let c = col_start + c_offset;

                        if ocp_orig_slice[[r, c]] == 0.0 {
                            continue;
                        }
                        let center_depth = z_slice[[r, c]];
                        if !center_depth.is_finite() {
                            continue;
                        }

                        let r_start_nb = r - w;
                        let r_end_nb = r + w + 1;
                        let c_start_nb = c - w;
                        let c_end_nb = c + w + 1;
                        let mut sum_depth = 0.0;
                        let mut count_neighbors = 0.0;

                        for nr in r_start_nb..r_end_nb {
                            for nc in c_start_nb..c_end_nb {
                                if nr == r && nc == c {
                                    continue;
                                }
                                if ocp_orig_slice[[nr, nc]] > 0.0 {
                                    // Use original ocp map
                                    let neighbor_depth = z_slice[[nr, nc]]; // Use z buffer
                                    if neighbor_depth.is_finite() {
                                        sum_depth += neighbor_depth;
                                        count_neighbors += 1.0;
                                    }
                                }
                            }
                        }

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
                    } // end inner col loop
                    row_outlier_coords // Return coords found in this row
                })
                .collect(); // Collect all outlier coords for layer k

            // --- Pass 2: Mutate & Count (Sequential for layer k) ---
            // Get mutable slices for the current layer k
            let mut image_slice = images.index_axis_mut(Axis(0), k);
            let mut ocp_slice = ocp_maps.index_axis_mut(Axis(0), k);

            let layer_outliers_count = outliers_coords.len() as u64;
            if layer_outliers_count > 0 {
                // Perform mutation sequentially using the collected coordinates
                for &(r, c) in &outliers_coords {
                    ocp_slice[[r, c]] = 0.0; // Update final occupancy map
                    for chan in 0..channels {
                        image_slice[[r, c, chan]] = initial_color_val; // Update final image
                    }
                }
            }
            // Update frequency count for this layer (safe because k loop is sequential)
            freqs[k] = layer_outliers_count;
        } // End sequential k loop for filtering

        conditional_log!(logging_enabled, info, "Filtering finished");
    } else if filtering > 0 {
        conditional_log!(
            logging_enabled,
            warn,
            "Filtering skipped: Image dimensions too small for filter size."
        );
    }

    (images, ocp_maps, freqs)
}
