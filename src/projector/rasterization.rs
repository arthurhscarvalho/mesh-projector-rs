// src/projector/rasterization.rs
use crate::projector::config::ProjectionConfig;
use crate::projector::geometry::{
    barycentric, calculate_bounding_box, is_inside_triangle, is_triangle_visible,
};
use crate::projector::texture::TextureData;
use nd::{s, Array2, ArrayView, ArrayView2, ArrayViewMut, ArrayViewMut2, ArrayViewMut3};
use std::f32;

/// Interpolates texture coordinates using barycentric coordinates and samples the texture color.
/// Assumes standard UV mapping where (0,0) is top-left and (1,1) is bottom-right.
/// Handles clamping coordinates to the texture boundaries.
#[inline]
fn sample_texture_color(
    u: f32,                                       // Barycentric coordinate corresponding to uv0
    v: f32,                                       // Barycentric coordinate corresponding to uv1
    w: f32,                                       // Barycentric coordinate corresponding to uv2
    uv0: ArrayView<f32, nd::Ix1>, // Texture coordinates (u, v) for the first vertex of the triangle
    uv1: ArrayView<f32, nd::Ix1>, // Texture coordinates (u, v) for the second vertex
    uv2: ArrayView<f32, nd::Ix1>, // Texture coordinates (u, v) for the third vertex
    texture_data: &TextureData,   // Contains texture view and dimensions
    target_color: &mut ArrayViewMut<u8, nd::Ix1>, // Mutable slice to write the sampled color (length = channels)
) {
    // Linearly interpolate the UV coordinates using the barycentric coordinates.
    let interp_u_coord = u * uv0[0] + v * uv1[0] + w * uv2[0];
    let interp_v_coord = u * uv0[1] + v * uv1[1] + w * uv2[1];

    // Convert interpolated UV coordinates [0, 1] to texture pixel coordinates [0, width-1], [0, height-1].
    // We sample at the nearest pixel center using round().
    // Clamp coordinates to ensure they stay within the valid texture bounds [0, dim-1].
    let tex_x = (interp_u_coord * texture_data.width_f32) // Scale U to texture width
        .round() // Find nearest pixel center
        .max(0.0) // Clamp minimum to 0
        .min(texture_data.width_f32 - 1.0) // Clamp maximum to width-1
        as usize; // Convert to integer index

    // The V texture coordinate is often inverted (0 is top, 1 is bottom in UV, but 0 is top row in image array).
    // We apply (1.0 - V) to flip it before scaling and clamping.
    let tex_y = ((1.0 - interp_v_coord) * texture_data.height_f32) // Scale flipped V to texture height
        .round() // Find nearest pixel center
        .max(0.0) // Clamp minimum to 0
        .min(texture_data.height_f32 - 1.0) // Clamp maximum to height-1
        as usize; // Convert to integer index

    // Sample the color from the texture at the calculated (tex_y, tex_x) coordinates
    // for all channels and write it to the target color buffer slice.
    for chan in 0..texture_data.channels {
        target_color[chan] = texture_data.texture[[tex_y, tex_x, chan]];
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

    // UV coordinates are often stored per-face-vertex, so index calculation is t_idx * 3 + vertex_index.
    let uv_idx_base = t_idx * 3;
    let uv0 = triangle_uvs.row(uv_idx_base); // UV coords for vertex 0 of this triangle instance
    let uv1 = triangle_uvs.row(uv_idx_base + 1); // UV coords for vertex 1
    let uv2 = triangle_uvs.row(uv_idx_base + 2); // UV coords for vertex 2

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
                    // 7. Interpolate Depth: Calculate the depth value at this pixel using barycentric coordinates.
                    let interpolated_depth = u * p0[projection_config.depth_axis]
                        + v * p1[projection_config.depth_axis]
                        + w * p2[projection_config.depth_axis];

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
                        // Interpolate UVs and sample the texture color.
                        sample_texture_color(
                            u,
                            v,
                            w,
                            uv0,
                            uv1,
                            uv2,
                            texture_data,
                            &mut color_pixel_slice,
                        );
                    }
                }
            }
        }
    }
}
