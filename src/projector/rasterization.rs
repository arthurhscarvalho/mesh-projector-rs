// src/projector/rasterization.rs
use crate::projector::config::ProjectionConfig;
use crate::projector::geometry::{
    barycentric, calculate_bounding_box, is_inside_triangle, is_triangle_visible,
};
use crate::projector::texture::TextureData;
use nd::{s, Array2, ArrayView2, ArrayViewMut, ArrayViewMut2, ArrayViewMut3};
use std::f32;

/// Interpolates texture coordinates using barycentric coordinates and samples the texture color
/// using bilinear filtering.
/// Assumes standard UV mapping where (0,0) is top-left and (1,1) is bottom-right.
/// Handles clamping coordinates to the texture boundaries.
#[inline]
fn sample_texture_color_bilinear(
    interp_u_coord: f32, // Perspective-correct interpolated U coordinate [0, 1]
    interp_v_coord: f32, // Perspective-correct interpolated V coordinate [0, 1]
    texture_data: &TextureData, // Contains texture view and dimensions
    target_color: &mut ArrayViewMut<u8, nd::Ix1>, // Mutable slice to write the sampled color (length = channels)
) {
    // --- Bilinear Filtering Implementation ---

    // Calculate floating-point texture coordinates (px, py)
    // Flip V coordinate: (0,0) in UV is top-left, (0,0) in image array is top-left.
    let tex_xf = interp_u_coord * texture_data.width_f32;
    let tex_yf = (1.0 - interp_v_coord) * texture_data.height_f32;

    // Calculate the integer coordinates of the top-left texel (x0, y0)
    // Ensure coordinates stay within valid sampling range [0, dim - 1.0001] to avoid issues at the boundary
    let x0f = tex_xf.max(0.0).min(texture_data.width_f32 - 1.0001);
    let y0f = tex_yf.max(0.0).min(texture_data.height_f32 - 1.0001);
    let x0 = x0f as usize;
    let y0 = y0f as usize;

    // Calculate the integer coordinates of the bottom-right texel (x1, y1)
    // x1 and y1 are guaranteed to be within bounds [0, width/height - 1] because x0/y0 are clamped <= dim - 1.0001
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    // Calculate the fractional parts (weights) for interpolation
    let wx = x0f - x0 as f32; // Weight for interpolation between x0 and x1
    let wy = y0f - y0 as f32; // Weight for interpolation between y0 and y1

    // Sample the four neighboring texels
    let tex = &texture_data.texture;
    let c = texture_data.channels;
    for chan in 0..c {
        let c00 = tex[[y0, x0, chan]] as f32; // Top-left
        let c10 = tex[[y0, x1, chan]] as f32; // Top-right
        let c01 = tex[[y1, x0, chan]] as f32; // Bottom-left
        let c11 = tex[[y1, x1, chan]] as f32; // Bottom-right

        // Perform bilinear interpolation
        let interpolated_color = c00 * (1.0 - wx) * (1.0 - wy) // Top-left influence
                               + c10 * wx * (1.0 - wy)         // Top-right influence
                               + c01 * (1.0 - wx) * wy         // Bottom-left influence
                               + c11 * wx * wy; // Bottom-right influence

        // Clamp the result to [0, 255] and convert back to u8
        target_color[chan] = interpolated_color.round().max(0.0).min(255.0) as u8;
    }
}

/// Rasterizes a single triangle onto the specified image slices (color, depth, occlusion).
/// Performs back-face culling, projects vertices, iterates pixels within the bounding box,
/// performs depth testing, and updates buffers for visible fragments inside the triangle.
pub fn rasterize_triangle(
    t_idx: usize,                         // Index of the current triangle being processed
    points: ArrayView2<f32>,              // All vertex 3D positions (N, 3)
    triangles: ArrayView2<usize>,         // Triangle vertex indices (M, 3) defining faces
    triangle_uvs: ArrayView2<f32>,        // UV coordinates per triangle vertex (M*3, 2)
    triangle_normals: &Array2<f32>,       // Precomputed geometric normals per triangle (M, 3)
    texture_data: &TextureData,           // Texture image and properties
    projection_config: &ProjectionConfig, // Configuration for the current projection view
    z_slice: &mut ArrayViewMut2<f32>,     // Mutable view to the Z-buffer for this view
    image_slice: &mut ArrayViewMut3<u8>,  // Mutable view to the color image buffer for this view
    ocp_slice: &mut ArrayViewMut2<f32>,   // Mutable view to the occlusion map for this view
) {
    // 1. Back-face Culling: Check if the triangle is visible from the current view direction.
    let normal = triangle_normals.row(t_idx);
    if !is_triangle_visible(normal[0], normal[1], normal[2], projection_config.view_dir) {
        return; // Cull the triangle if it's facing away from the view.
    }

    // 2. Get Vertex Data: Retrieve 3D positions and UV coordinates for the triangle's vertices.
    let tri_indices = triangles.row(t_idx);
    let i0 = tri_indices[0];
    let i1 = tri_indices[1];
    let i2 = tri_indices[2];

    let p0 = points.row(i0); // 3D position of vertex 0
    let p1 = points.row(i1); // 3D position of vertex 1
    let p2 = points.row(i2); // 3D position of vertex 2

    // --- Perspective Correct Interpolation Setup ---
    // Extract depth (z) coordinates and compute 1/z for perspective correction.
    // Add a small epsilon to avoid division by zero or issues very close to the projection plane.
    let z0 = p0[projection_config.depth_axis];
    let z1 = p1[projection_config.depth_axis];
    let z2 = p2[projection_config.depth_axis];
    let inv_z0 = 1.0 / (z0 + 1e-6);
    let inv_z1 = 1.0 / (z1 + 1e-6);
    let inv_z2 = 1.0 / (z2 + 1e-6);

    // UV coordinates are often stored per-face-vertex, so index calculation is t_idx * 3 + vertex_index.
    let uv_idx_base = t_idx * 3;
    let uv0 = triangle_uvs.row(uv_idx_base); // UV coords for vertex 0 of this triangle instance
    let uv1 = triangle_uvs.row(uv_idx_base + 1); // UV coords for vertex 1
    let uv2 = triangle_uvs.row(uv_idx_base + 2); // UV coords for vertex 2

    // Pre-calculate UV coordinates divided by Z for perspective-correct interpolation.
    let uv0_over_z = (uv0[0] * inv_z0, uv0[1] * inv_z0);
    let uv1_over_z = (uv1[0] * inv_z1, uv1[1] * inv_z1);
    let uv2_over_z = (uv2[0] * inv_z2, uv2[1] * inv_z2);
    // --- End Perspective Correct Interpolation Setup ---

    // 3. Project Vertices to 2D Plane: Extract the P and Q axis coordinates based on the projection config.
    // These correspond to the (column, row) coordinates in the output image buffer.
    let p0_proj = (p0[projection_config.p_axis], p0[projection_config.q_axis]);
    let p1_proj = (p1[projection_config.p_axis], p1[projection_config.q_axis]);
    let p2_proj = (p2[projection_config.p_axis], p2[projection_config.q_axis]);

    // 4. Calculate Bounding Box: Determine the min/max pixel coordinates covering the projected triangle.
    let (rows, cols) = (z_slice.shape()[0], z_slice.shape()[1]);
    let Some((min_row, max_row, min_col, max_col)) =
        calculate_bounding_box(p0_proj, p1_proj, p2_proj, rows, cols)
    else {
        return; // Triangle is completely off-screen or projects to zero area on screen.
    };

    // 5. Iterate Pixels in Bounding Box: Loop through each pixel potentially covered by the triangle.
    for r in min_row..=max_row {
        // Iterate through rows (q_axis)
        for c in min_col..=max_col {
            // Iterate through columns (p_axis)
            // Use pixel center coordinates for barycentric calculation for better accuracy.
            let px = c as f32 + 0.5; // Center x-coordinate (p_axis)
            let py = r as f32 + 0.5; // Center y-coordinate (q_axis)

            // 6. Calculate Barycentric Coordinates: Determine if the pixel center (px, py) lies inside the projected triangle.
            if let Some((u, v, w)) = barycentric(p0_proj, p1_proj, p2_proj, (px, py)) {
                // Check if the coordinates are valid (all >= 0, handling floating point tolerance).
                if is_inside_triangle(u, v, w) {
                    // --- Perspective Correct Interpolation ---
                    // Interpolate 1/z across the triangle.
                    let interpolated_inv_z = u * inv_z0 + v * inv_z1 + w * inv_z2;
                    // Calculate the perspective-correct depth Z.
                    let interpolated_depth = 1.0 / interpolated_inv_z;

                    // Interpolate uv/z across the triangle.
                    let interpolated_u_over_z =
                        u * uv0_over_z.0 + v * uv1_over_z.0 + w * uv2_over_z.0;
                    let interpolated_v_over_z =
                        u * uv0_over_z.1 + v * uv1_over_z.1 + w * uv2_over_z.1;

                    // Calculate the perspective-correct UV coordinates.
                    let interp_u_coord = interpolated_u_over_z * interpolated_depth;
                    let interp_v_coord = interpolated_v_over_z * interpolated_depth;
                    // --- End Perspective Correct Interpolation ---

                    // 8. Depth Test: Compare the interpolated depth with the value currently in the Z-buffer.
                    let current_z = z_slice[[r, c]];
                    let depth_test_passed = if projection_config.is_front_view {
                        // For front views, smaller depth values are closer.
                        // Initial Z is +Infinity.
                        interpolated_depth < current_z
                    } else {
                        // For back views, larger depth values (less negative or more positive if crossing origin) are closer.
                        // Initial Z is -Infinity.
                        interpolated_depth > current_z
                    };

                    if depth_test_passed {
                        // 9. Update Buffers: If the depth test passes, this fragment is visible.
                        // Update the Z-buffer with the new closer depth.
                        z_slice[[r, c]] = interpolated_depth;
                        // Mark this pixel as covered by the mesh in the occlusion map.
                        ocp_slice[[r, c]] = 1.0;

                        // 10. Sample Texture and Update Color Buffer:
                        // Get a mutable slice for the color channels of the current pixel.
                        let mut color_pixel_slice = image_slice.slice_mut(s![r, c, ..]);
                        // Interpolate UVs and sample the texture color using bilinear filtering.
                        sample_texture_color_bilinear(
                            interp_u_coord,
                            interp_v_coord,
                            texture_data,
                            &mut color_pixel_slice,
                        );
                    }
                }
            }
        }
    }
}
