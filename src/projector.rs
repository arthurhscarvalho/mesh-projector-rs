use nd::{s, Array2, Array3, Array4};
use std::f64;

use crate::conditional_log;

/// Helper: computes barycentric coordinates of point (px,py) with respect to triangle vertices (ax,ay), (bx,by), (cx,cy).
/// Returns (u,v,w) such that p = u*a + v*b + w*c and u+v+w = 1.
/// Returns None if the triangle area is nearly zero (degenerate).
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
    if denom.abs() < 1e-9 {
        // Use a small epsilon
        return None;
    }
    // Calculate barycentric coordinates v and w
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    // Calculate u from u + v + w = 1
    let u = 1.0 - v - w;
    Some((u, v, w))
}

/// Compute mesh projections onto the 6 faces of a bounding cube using a Z-buffer,
/// per-pixel texture mapping, and back-face culling.
///
/// Projects the mesh orthographically along the X, Y, and Z axes onto the faces
/// of the cube defined by `[0, max_bound] x [0, max_bound] x [0, max_bound]`.
///
/// # Arguments
/// * `points`: An (N x 3) array of 3D point coordinates (f64). Expected to be scaled within the bounding cube.
/// * `triangles`: An (M x 3) array of indices (usize) into `points`, defining M triangular faces.
/// * `texture`: An (TexH x TexW x C) array representing the texture image (e.g., u8 for RGB).
/// * `triangle_uvs`: An (M*3 x 2) array of UV coordinates (f64, range 0.0-1.0), corresponding to each vertex corner of each triangle.
/// * `precision`: Log2 of the side length of the output images (e.g., 8 means 256x256).
/// * `filtering`: Size of the square neighborhood window for outlier filtering (e.g., 1 means 3x3). 0 disables filtering.
///
/// # Returns
/// A tuple containing:
/// * `images`: An `Array4<u8>` of shape (6, H, W, C) holding the 6 projected color images (H=W=2^precision).
/// * `ocp_maps`: An `Array3<f64>` of shape (6, H, W) indicating occupied pixels (1.0) vs background (0.0).
/// * `freqs`: An array `[u64; 6]` counting the number of pixels filtered as outliers for each of the 6 views.
pub fn compute_mesh_projections_textured(
    points: Array2<f64>,       // Use ArrayView for read-only access
    triangles: Array2<usize>,  // Use ArrayView
    texture: Array3<u8>,       // Texture data (e.g., RGB u8)
    triangle_uvs: Array2<f64>, // Per-triangle-corner UVs
    precision: u32,
    filtering: u32,
    logging_enabled: bool,
) -> (Array4<u8>, Array3<f64>, [u64; 6]) {
    conditional_log!(logging_enabled, info, "Inititalizing projector back-end");

    let max_bound: u64 = 1 << precision;
    let max_bound_f64: f64 = max_bound as f64;
    let max_bound_u: usize = max_bound as usize;
    let rows = max_bound_u;
    let cols = max_bound_u;

    // Texture dimensions
    let tex_height = texture.shape()[0];
    let tex_width = texture.shape()[1];
    let channels = texture.shape()[2]; // Get channels from texture
    let tex_height_f64 = tex_height as f64;
    let tex_width_f64 = tex_width as f64;

    let num_images: usize = 6;
    let initial_color_val: u8 = 255; // Background color (white) as u8

    if points.is_empty() || triangles.is_empty() || texture.is_empty() || triangle_uvs.is_empty() {
        conditional_log!(
            logging_enabled,
            warn,
            "Warning: Empty input detected (points, triangles, texture, or uvs)."
        );
        return (
            Array4::from_elem((num_images, rows, cols, channels), initial_color_val),
            Array3::zeros((num_images, rows, cols)),
            [0; 6],
        );
    }

    if triangles.nrows() * 3 != triangle_uvs.nrows() {
        panic!(
            "Inconsistent input: Number of triangle indices ({}) * 3 != number of triangle UVs ({}).",
            triangles.nrows(), triangle_uvs.nrows()
         );
    }

    let mut images = Array4::from_elem((num_images, rows, cols, channels), initial_color_val);
    let mut ocp_maps = Array3::zeros((num_images, rows, cols)); // 0.0 = empty, 1.0 = occupied
    let mut z_buffers = Array3::from_elem((num_images, rows, cols), f64::INFINITY);
    let projection_plane_axes: [(usize, usize); 3] = [(1, 2), (0, 2), (0, 1)];

    conditional_log!(logging_enabled, info, "Starting rasterization loop");

    // --- Rasterization Loop ---
    let num_faces = triangles.nrows();
    for t_idx in 0..num_faces {
        // Get vertex indices for this triangle
        let i0 = triangles[[t_idx, 0]];
        let i1 = triangles[[t_idx, 1]];
        let i2 = triangles[[t_idx, 2]];

        // Get vertex points in 3D
        let p0 = points.row(i0);
        let p1 = points.row(i1);
        let p2 = points.row(i2);

        // Compute triangle face normal using cross product of (p1 - p0) and (p2 - p0)
        let ax = p1[0] - p0[0];
        let ay = p1[1] - p0[1];
        let az = p1[2] - p0[2];
        let bx = p2[0] - p0[0];
        let by = p2[1] - p0[1];
        let bz = p2[2] - p0[2];
        let normal_x = ay * bz - az * by;
        let normal_y = az * bx - ax * bz;
        let normal_z = ax * by - ay * bx;
        // (normal_x, normal_y, normal_z) is the triangle normal.

        // Get the UV coordinates for the 3 corners of this triangle
        let uv_idx0 = t_idx * 3;
        let uv_idx1 = t_idx * 3 + 1;
        let uv_idx2 = t_idx * 3 + 2;
        let uv0 = triangle_uvs.row(uv_idx0);
        let uv1 = triangle_uvs.row(uv_idx1);
        let uv2 = triangle_uvs.row(uv_idx2);

        // Loop over each projection axis (depth axis)
        for depth_axis in 0..3 {
            let (p_axis, q_axis) = projection_plane_axes[depth_axis];

            // 2D coordinates for the triangle vertices in the projection plane
            let ax_2d = p0[p_axis];
            let ay_2d = p0[q_axis];
            let bx_2d = p1[p_axis];
            let by_2d = p1[q_axis];
            let cx_2d = p2[p_axis];
            let cy_2d = p2[q_axis];

            // Depth values (from the current depth axis)
            let da = p0[depth_axis];
            let db = p1[depth_axis];
            let dc = p2[depth_axis];

            // Bounding box in 2D for the triangle
            let min_col =
                (ax_2d.min(bx_2d.min(cx_2d)).floor().max(0.0) as usize).min(cols.saturating_sub(1));
            let max_col = (ax_2d.max(bx_2d.max(cx_2d)).ceil().min(max_bound_f64 - 1e-9) as usize)
                .min(cols.saturating_sub(1));
            let min_row =
                (ay_2d.min(by_2d.min(cy_2d)).floor().max(0.0) as usize).min(rows.saturating_sub(1));
            let max_row = (ay_2d.max(by_2d.max(cy_2d)).ceil().min(max_bound_f64 - 1e-9) as usize)
                .min(rows.saturating_sub(1));

            if min_col > max_col || min_row > max_row {
                continue;
            }

            // Loop over the two view types for each axis
            for view_type in 0..2 {
                let img_layer = 2 * depth_axis + view_type;
                let is_front_view = view_type == 0;

                // Determine view direction for current projection:
                // For depth_axis=0: front view -> (1,0,0), back view -> (-1,0,0)
                // For depth_axis=1: front view -> (0,1,0), back view -> (0,-1,0)
                // For depth_axis=2: front view -> (0,0,1), back view -> (0,0,-1)
                let view_dir = match (depth_axis, view_type) {
                    (0, 0) => (1.0, 0.0, 0.0),
                    (0, 1) => (-1.0, 0.0, 0.0),
                    (1, 0) => (0.0, 1.0, 0.0),
                    (1, 1) => (0.0, -1.0, 0.0),
                    (2, 0) => (0.0, 0.0, 1.0),
                    (2, 1) => (0.0, 0.0, -1.0),
                    _ => unreachable!(),
                };

                // Compute dot product between triangle normal and view direction.
                let dot = normal_x * view_dir.0 + normal_y * view_dir.1 + normal_z * view_dir.2;
                // Back-face culling: if the triangle faces away from the camera, skip rasterization for this view.
                if dot >= 0.0 {
                    continue;
                }

                // Rasterize triangle within its bounding box for this view
                for r in min_row..=max_row {
                    for c in min_col..=max_col {
                        let px = c as f64 + 0.5;
                        let py = r as f64 + 0.5;

                        if let Some((u, v, w)) =
                            barycentric(ax_2d, ay_2d, bx_2d, by_2d, cx_2d, cy_2d, px, py)
                        {
                            if u >= -1e-9 && v >= -1e-9 && w >= -1e-9 {
                                // Inside triangle: compute interpolated depth.
                                let interpolated_depth = u * da + v * db + w * dc;
                                let current_z = z_buffers[[img_layer, r, c]];
                                let depth_test_passed = if is_front_view {
                                    interpolated_depth < current_z
                                } else {
                                    interpolated_depth > current_z
                                };

                                if depth_test_passed {
                                    z_buffers[[img_layer, r, c]] = interpolated_depth;
                                    // Texture Mapping: Interpolate UV coordinates.
                                    let interp_u = u * uv0[0] + v * uv1[0] + w * uv2[0];
                                    let interp_v = u * uv0[1] + v * uv1[1] + w * uv2[1];
                                    let tex_x = (interp_u * tex_width_f64)
                                        .round()
                                        .max(0.0)
                                        .min(tex_width_f64 - 1.0)
                                        as usize;
                                    let tex_y = (interp_v * tex_height_f64)
                                        .round()
                                        .max(0.0)
                                        .min(tex_height_f64 - 1.0)
                                        as usize;

                                    let final_color_view = texture.slice(s![tex_y, tex_x, ..]);
                                    images
                                        .slice_mut(s![img_layer, r, c, ..])
                                        .assign(&final_color_view);
                                    ocp_maps[[img_layer, r, c]] = 1.0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    conditional_log!(
        logging_enabled && filtering > 0,
        info,
        "Starting filtering loop"
    );

    // --- Optional Outlier Filtering Pass (Needs slight adaptation for u8 images) ---
    let mut freqs: [u64; 6] = [0; 6];
    if filtering > 0 {
        let w = filtering as usize;
        let filter_threshold = 0.1 * max_bound_f64;
        let ocp_maps_orig = ocp_maps.clone();

        for k in 0..num_images {
            let is_front_view = k % 2 == 0;
            for r in w..(rows.saturating_sub(w)) {
                for c in w..(cols.saturating_sub(w)) {
                    if ocp_maps_orig[[k, r, c]] == 0.0 {
                        continue;
                    }
                    let r_start = r - w;
                    let r_end = r + w + 1;
                    let c_start = c - w;
                    let c_end = c + w + 1;
                    let z_neighborhood = z_buffers.slice(s![k, r_start..r_end, c_start..c_end]);
                    let ocp_neighborhood =
                        ocp_maps_orig.slice(s![k, r_start..r_end, c_start..c_end]);
                    let mut sum_depth = 0.0;
                    let mut count_neighbors = 0.0;
                    let center_local_idx = (w, w);

                    for nr in 0..(2 * w + 1) {
                        for nc in 0..(2 * w + 1) {
                            if (nr, nc) != center_local_idx && ocp_neighborhood[[nr, nc]] > 0.0 {
                                let neighbor_depth = z_neighborhood[[nr, nc]];
                                if neighbor_depth.is_finite() {
                                    sum_depth += neighbor_depth;
                                    count_neighbors += 1.0;
                                }
                            }
                        }
                    }

                    if count_neighbors > 0.0 {
                        let avg_neighbor_depth = sum_depth / count_neighbors;
                        let current_depth = z_buffers[[k, r, c]];
                        if current_depth.is_finite() {
                            let is_outlier = if is_front_view {
                                current_depth > avg_neighbor_depth + filter_threshold
                            } else {
                                current_depth < avg_neighbor_depth - filter_threshold
                            };
                            if is_outlier {
                                ocp_maps[[k, r, c]] = 0.0;
                                images.slice_mut(s![k, r, c, ..]).fill(initial_color_val);
                                freqs[k] += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    (images, ocp_maps, freqs)
}
