use crate::conditional_log;
use nd::{s, Array2, Array3, Array4, ArrayView2, ArrayView3, Axis, Ix3, Ix4};
use rayon::prelude::*;

#[inline]
fn barycentric(
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
    cx: f32,
    cy: f32,
    px: f32,
    py: f32,
) -> Option<(f32, f32, f32)> {
    let v0x = bx - ax;
    let v0y = by - ay;
    let v1x = cx - ax;
    let v1y = cy - ay;
    let v2x = px - ax;
    let v2y = py - ay;

    let d00 = v0x * v0x + v0y * v0y;
    let d01 = v0x * v1x + v0y * v1y;
    let d11 = v1x * v1x + v1y * v1y;
    let d20 = v2x * v0x + v2y * v0y;
    let d21 = v2x * v1x + v2y * v1y;

    let denom = d00 * d11 - d01 * d01;

    const BARYCENTRIC_EPSILON: f32 = 1e-7;
    if denom.abs() < BARYCENTRIC_EPSILON {
        return None;
    }

    let inv_denom = 1.0f32 / denom;
    let v = (d11 * d20 - d01 * d21) * inv_denom;
    let w = (d00 * d21 - d01 * d20) * inv_denom;

    let u = 1.0f32 - v - w;
    Some((u, v, w))
}

pub fn compute_mesh_projections(
    points: ArrayView2<f32>,
    triangles: ArrayView2<usize>,
    texture: ArrayView3<u8>,
    triangle_uvs: ArrayView2<f32>,
    precision: u32,
    filtering: u32,
    logging_enabled: bool,
) -> (Array4<u8>, Array3<f32>, [u64; 6]) {
    conditional_log!(
        logging_enabled,
        info,
        "Initializing optimized projector back-end"
    );

    let max_bound: u64 = 1 << precision;
    let max_bound_f32: f32 = max_bound as f32;
    let max_bound_u: usize = max_bound as usize;
    let rows = max_bound_u;
    let cols = max_bound_u;

    let tex_height = texture.shape()[0];
    let tex_width = texture.shape()[1];
    let channels = texture.shape()[2];
    let tex_height_f32 = tex_height as f32;
    let tex_width_f32 = tex_width as f32;

    let num_images: usize = 6;
    let initial_color_val: u8 = 255;
    const BARYCENTRIC_COORD_EPSILON: f32 = -1e-7;

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

    let num_faces = triangles.nrows();

    conditional_log!(logging_enabled, info, "Precomputing triangle normals");
    let mut triangle_normals = Array2::<f32>::zeros((num_faces, 3));
    for t_idx in 0..num_faces {
        let i0 = triangles[[t_idx, 0]];
        let i1 = triangles[[t_idx, 1]];
        let i2 = triangles[[t_idx, 2]];

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

        triangle_normals[[t_idx, 0]] = normal_x;
        triangle_normals[[t_idx, 1]] = normal_y;
        triangle_normals[[t_idx, 2]] = normal_z;
    }
    conditional_log!(logging_enabled, info, "Normal precomputation finished");

    let mut images = Array4::from_elem((num_images, rows, cols, channels), initial_color_val);
    let mut ocp_maps = Array3::<f32>::zeros((num_images, rows, cols));
    let mut z_buffers = Array3::from_elem((num_images, rows, cols), f32::INFINITY);
    for i in (1..num_images).step_by(2) {
        z_buffers.slice_mut(s![i, .., ..]).fill(f32::NEG_INFINITY);
    }

    let projection_plane_axes: [(usize, usize); 3] = [(1, 2), (0, 2), (0, 1)];

    conditional_log!(
        logging_enabled,
        info,
        "Starting parallel rasterization loop (f32)"
    );

    let mut z_buffer_views: Vec<_> = z_buffers.axis_iter_mut(Axis(0)).collect();
    let mut image_views: Vec<_> = images.axis_iter_mut(Axis(0)).collect();
    let mut ocp_map_views: Vec<_> = ocp_maps.axis_iter_mut(Axis(0)).collect();

    z_buffer_views
        .par_iter_mut()
        .zip(image_views.par_iter_mut())
        .zip(ocp_map_views.par_iter_mut())
        .enumerate()
        .for_each(|(img_layer, ((z_slice, image_slice), ocp_slice))| {
            let depth_axis = img_layer / 2;
            let view_type = img_layer % 2;
            let is_front_view = view_type == 0;

            let view_dir = match (depth_axis, view_type) {
                (0, 0) => (1.0f32, 0.0f32, 0.0f32),
                (0, 1) => (-1.0f32, 0.0f32, 0.0f32),
                (1, 0) => (0.0f32, 1.0f32, 0.0f32),
                (1, 1) => (0.0f32, -1.0f32, 0.0f32),
                (2, 0) => (0.0f32, 0.0f32, 1.0f32),
                (2, 1) => (0.0f32, 0.0f32, -1.0f32),
                _ => unreachable!(),
            };

            let (p_axis, q_axis) = projection_plane_axes[depth_axis];

            for t_idx in 0..num_faces {
                let normal = triangle_normals.row(t_idx);
                let dot = normal[0] * view_dir.0 + normal[1] * view_dir.1 + normal[2] * view_dir.2;
                if dot >= 0.0f32 {
                    continue;
                }

                let i0 = triangles[[t_idx, 0]];
                let i1 = triangles[[t_idx, 1]];
                let i2 = triangles[[t_idx, 2]];

                let p0 = points.row(i0);
                let p1 = points.row(i1);
                let p2 = points.row(i2);

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
                let max_col = (max_col_f.ceil() as i64).min(cols as i64 - 1).max(0) as usize;
                let min_row = (min_row_f.floor() as i64).max(0) as usize;
                let max_row = (max_row_f.ceil() as i64).min(rows as i64 - 1).max(0) as usize;

                if min_col >= cols || min_row >= rows || min_col > max_col || min_row > max_row {
                    continue;
                }

                for r in min_row..=max_row {
                    for c in min_col..=max_col {
                        let px = c as f32 + 0.5f32;
                        let py = r as f32 + 0.5f32;

                        if let Some((u, v, w)) =
                            barycentric(ax_2d, ay_2d, bx_2d, by_2d, cx_2d, cy_2d, px, py)
                        {
                            if u >= BARYCENTRIC_COORD_EPSILON
                                && v >= BARYCENTRIC_COORD_EPSILON
                                && w >= BARYCENTRIC_COORD_EPSILON
                            {
                                let interpolated_depth = u * da + v * db + w * dc;

                                let current_z = z_slice[[r, c]];
                                let depth_test_passed = if is_front_view {
                                    interpolated_depth < current_z
                                } else {
                                    interpolated_depth > current_z
                                };

                                if depth_test_passed {
                                    z_slice[[r, c]] = interpolated_depth;

                                    let interp_u = u * uv0[0] + v * uv1[0] + w * uv2[0];
                                    let interp_v = u * uv0[1] + v * uv1[1] + w * uv2[1];
                                    let tex_x = (interp_u * tex_width_f32)
                                        .round()
                                        .max(0.0f32)
                                        .min(tex_width_f32 - 1.0f32)
                                        as usize;
                                    let tex_y = ((1.0f32 - interp_v) * tex_height_f32)
                                        .round()
                                        .max(0.0f32)
                                        .min(tex_height_f32 - 1.0f32)
                                        as usize;

                                    for chan in 0..channels {
                                        image_slice[[r, c, chan]] = texture[[tex_y, tex_x, chan]];
                                    }
                                    ocp_slice[[r, c]] = 1.0f32;
                                }
                            }
                        }
                    }
                }
            }
        });

    conditional_log!(logging_enabled, info, "Rasterization finished");

    let mut freqs: [u64; 6] = [0; 6];
    if filtering > 0 && rows > 2 * filtering as usize && cols > 2 * filtering as usize {
        conditional_log!(
            logging_enabled,
            info,
            "Starting parallel filtering loop (f32)"
        );
        let w = filtering as usize;

        let filter_threshold = 0.1 * max_bound_f32;

        for k in 0..num_images {
            let ocp_orig_slice = ocp_maps.index_axis(Axis(0), k).to_owned();

            let is_front_view = k % 2 == 0;
            let row_start = w;
            let col_start = w;
            let num_inner_rows = rows.saturating_sub(2 * w);
            let num_inner_cols = cols.saturating_sub(2 * w);

            let z_slice = z_buffers.index_axis(Axis(0), k);

            let outliers_coords: Vec<(usize, usize)> = (0..num_inner_rows)
                .into_par_iter()
                .flat_map(|r_offset| {
                    let r = row_start + r_offset;
                    let mut row_outlier_coords = Vec::new();

                    for c_offset in 0..num_inner_cols {
                        let c = col_start + c_offset;

                        if ocp_orig_slice[[r, c]] == 0.0f32 {
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
                        let mut sum_depth = 0.0f32;
                        let mut count_neighbors = 0.0f32;

                        for nr in r_start_nb..r_end_nb {
                            for nc in c_start_nb..c_end_nb {
                                if nr == r && nc == c {
                                    continue;
                                }

                                if ocp_orig_slice[[nr, nc]] > 0.0f32 {
                                    let neighbor_depth = z_slice[[nr, nc]];
                                    if neighbor_depth.is_finite() {
                                        sum_depth += neighbor_depth;
                                        count_neighbors += 1.0f32;
                                    }
                                }
                            }
                        }

                        if count_neighbors > 0.0f32 {
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
                .collect();

            let mut image_slice = images.index_axis_mut(Axis(0), k);
            let mut ocp_slice = ocp_maps.index_axis_mut(Axis(0), k);

            let layer_outliers_count = outliers_coords.len() as u64;
            if layer_outliers_count > 0 {
                for &(r, c) in &outliers_coords {
                    ocp_slice[[r, c]] = 0.0f32;
                    for chan in 0..channels {
                        image_slice[[r, c, chan]] = initial_color_val;
                    }
                }
            }
            freqs[k] = layer_outliers_count;
        }

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
