// src/projector/rasterization.rs
use crate::projector::config::ProjectionConfig;
use crate::projector::geometry::{
    barycentric, calculate_bounding_box, dot_product, is_inside_triangle, is_triangle_visible,
    reflect, vec_scale_ret,
};
use crate::projector::texture::TextureData;
use nd::{Array2, ArrayView2, ArrayViewMut2, ArrayViewMut3};
use std::f32;

/// Interpolates texture coordinates using barycentric coordinates and samples the texture color
/// using bilinear filtering. Returns the sampled color as a `Vec<f32>` in [0, 1] range.
/// Assumes standard UV mapping where (0,0) is top-left and (1,1) is bottom-right.
/// Handles clamping coordinates to the texture boundaries.
#[inline]
fn sample_texture_color_bilinear(
    interp_u_coord: f32, // Perspective-correct interpolated U coordinate [0, 1]
    interp_v_coord: f32, // Perspective-correct interpolated V coordinate [0, 1]
    texture_data: &TextureData, // Contains texture view and dimensions
) -> Vec<f32> {
    let tex_xf = interp_u_coord * texture_data.width_f32;
    // Flip V coordinate for standard UV mapping (0,0 at top-left)
    let tex_yf = (1.0 - interp_v_coord) * texture_data.height_f32;

    // Clamp coordinates to valid texture range, accounting for potential floating point inaccuracies
    let x0f = tex_xf.max(0.0).min(texture_data.width_f32 - 1.0001);
    let y0f = tex_yf.max(0.0).min(texture_data.height_f32 - 1.0001);
    let x0 = x0f as usize;
    let y0 = y0f as usize;

    // Ensure bottom-right texel coordinates are within bounds
    let x1 = (x0 + 1).min((texture_data.width_f32 - 1.0) as usize);
    let y1 = (y0 + 1).min((texture_data.height_f32 - 1.0) as usize);

    // Calculate interpolation weights
    let wx = x0f - x0 as f32;
    let wy = y0f - y0 as f32;

    let tex = &texture_data.texture;
    let c = texture_data.channels;
    let mut sampled_color = vec![0.0f32; c];

    for chan in 0..c {
        let c00 = tex[[y0, x0, chan]] as f32; // Top-left
        let c10 = tex[[y0, x1, chan]] as f32; // Top-right
        let c01 = tex[[y1, x0, chan]] as f32; // Bottom-left
        let c11 = tex[[y1, x1, chan]] as f32; // Bottom-right

        let interpolated_channel = c00 * (1.0 - wx) * (1.0 - wy)
            + c10 * wx * (1.0 - wy)
            + c01 * (1.0 - wx) * wy
            + c11 * wx * wy;

        sampled_color[chan] = interpolated_channel / 255.0; // Normalize u8 to f32 [0, 1]
    }
    sampled_color
}

/// Calculates the final color of a fragment based on material, lighting, and texture.
/// Assumes inputs (material colors, texture) are sRGB and performs calculations in linear space.
#[inline]
fn calculate_fragment_color(
    material: &MaterialProperties,
    map_kd: Option<&TextureData>,
    interp_u_coord: f32,
    interp_v_coord: f32,
    normal: &[f32; 3],
    light_dir: &[f32; 3],
    view_dir: &[f32; 3],
    light_color: &[f32; 3],
) -> [f32; 3] {
    // Returns final *linear* color before alpha blending
    // --- Linearize Material Colors (assuming input Ka/Kd/Ks are sRGB) ---
    let linear_ka = [
        material.ka[0].powf(2.2),
        material.ka[1].powf(2.2),
        material.ka[2].powf(2.2),
    ];
    let linear_kd = [
        material.kd[0].powf(2.2),
        material.kd[1].powf(2.2),
        material.kd[2].powf(2.2),
    ];
    let linear_ks = [
        material.ks[0].powf(2.2),
        material.ks[1].powf(2.2),
        material.ks[2].powf(2.2),
    ];

    // --- Calculate Effective Diffuse Color (in linear space) ---
    let effective_diffuse_linear = if let Some(tex_data) = map_kd {
        let sampled_rgb = sample_texture_color_bilinear(interp_u_coord, interp_v_coord, tex_data);
        let mut modulated_color = [0.0f32; 3];
        if sampled_rgb.len() >= 3 {
            for i in 0..3 {
                let linear_sampled = sampled_rgb[i].powf(2.2); // Linearize texture sample
                modulated_color[i] = linear_sampled * linear_kd[i]; // Modulate by linear Kd
            }
        } else if sampled_rgb.len() == 1 {
            // Handle grayscale texture
            let linear_sampled = sampled_rgb[0].powf(2.2);
            for i in 0..3 {
                modulated_color[i] = linear_sampled * linear_kd[i];
            }
        } else {
            modulated_color = linear_kd; // Fallback to linear Kd if texture is invalid
        }
        modulated_color
    } else {
        // If no texture, use linearized material diffuse color directly
        linear_kd
    };

    // --- Illumination Model Logic (Using LINEAR colors) ---
    let mut final_color_linear = [0.0f32; 3];
    match material.illum {
        0 => {
            // Color on and Ambient off
            final_color_linear = effective_diffuse_linear;
        }
        1 => {
            // Color on and Ambient on
            for i in 0..3 {
                final_color_linear[i] = linear_ka[i] + effective_diffuse_linear[i];
            }
        }
        2 => {
            // Highlight on (Phong)

            // Ambient Component
            let global_ambient_light = [0.25, 0.25, 0.25]; // Standard ambient factor
            let ambient_term = [
                global_ambient_light[0] * linear_ka[0],
                global_ambient_light[1] * linear_ka[1],
                global_ambient_light[2] * linear_ka[2],
            ];

            // Diffuse Component
            let dot_nl = dot_product(normal, light_dir).max(0.0);
            let diffuse_term = vec_scale_ret(&effective_diffuse_linear, dot_nl * light_color[0]);

            // Specular Component
            let mut reflected_light_dir = [0.0; 3];
            let neg_light_dir = [-light_dir[0], -light_dir[1], -light_dir[2]];
            reflect(&neg_light_dir, normal, &mut reflected_light_dir);
            let dot_rv = dot_product(&reflected_light_dir, view_dir).max(0.0);
            let specular_intensity = if material.ns > 0.0 {
                dot_rv.powf(material.ns)
            } else {
                0.0
            };
            let specular_term = vec_scale_ret(&linear_ks, specular_intensity * light_color[0]);

            // Combine components
            for i in 0..3 {
                final_color_linear[i] = ambient_term[i] + diffuse_term[i] + specular_term[i];
            }
        }
        _ => {
            // Default/Unsupported: Use effective_diffuse color only
            final_color_linear = effective_diffuse_linear;
        }
    }

    // --- Reinhard Tone Mapping --- Apply tone mapping to the linear color
    let mut tonemapped_color = [0.0f32; 3];
    for i in 0..3 {
        tonemapped_color[i] = final_color_linear[i] / (1.0 + final_color_linear[i]);
    }

    tonemapped_color // Return the tone-mapped linear color
}

/// Rasterizes a single triangle onto the specified image slices (color, depth, occlusion).
/// Performs back-face culling, projects vertices, iterates pixels within the bounding box,
/// performs depth testing, Phong shading, and updates buffers for visible fragments.
pub fn rasterize_triangle(
    t_idx: usize,                         // Index of the current triangle being processed
    points: ArrayView2<f32>,              // All vertex 3D positions (N, 3)
    triangles: ArrayView2<usize>,         // Triangle vertex indices (M, 3) defining faces
    triangle_uvs: ArrayView2<f32>,        // UV coordinates per triangle vertex (M*3, 2)
    triangle_normals: &Array2<f32>,       // Precomputed geometric normals per triangle (M, 3)
    material: &MaterialProperties,        // Material properties for this triangle
    map_kd: Option<&TextureData>,         // Optional diffuse texture map
    projection_config: &ProjectionConfig, // Configuration for the current projection view
    z_slice: &mut ArrayViewMut2<f32>,     // Mutable view to the Z-buffer for this view
    image_slice: &mut ArrayViewMut3<u8>,  // Mutable view to the color image buffer for this view
    ocp_slice: &mut ArrayViewMut2<f32>,   // Mutable view to the occlusion map for this view
) {
    // 1. Back-face Culling: Check if the triangle is visible from the current view direction.
    let normal_view = triangle_normals.row(t_idx);
    let mut normal_f32: [f32; 3] = [normal_view[0], normal_view[1], normal_view[2]];
    // --- Normalize the normal vector for lighting calculations ---
    let mag_sq = dot_product(&normal_f32, &normal_f32);
    if mag_sq > 1e-9 {
        // Avoid division by zero or near-zero for degenerate triangles
        let inv_mag = 1.0 / mag_sq.sqrt();
        for i in 0..3 {
            normal_f32[i] *= inv_mag;
        }
    } else {
        // Handle degenerate triangle normal (e.g., set to a default like [0,0,1] or skip lighting)
        // For now, let it be [0,0,0], which will result in zero diffuse/specular
        normal_f32 = [0.0, 0.0, 0.0];
    }
    // -----------------------------------------------------------

    // Use the *unnormalized* normal for back-face culling (magnitude doesn't affect sign)
    if !is_triangle_visible(
        normal_view[0],
        normal_view[1],
        normal_view[2],
        projection_config.view_dir,
    ) {
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

    // --- Lighting Setup (Simple) ---
    // Use a fixed diagonal light source instead of view direction -> Change to use view direction
    /*
    let fixed_light_dir = [1.0f32, 1.0f32, 1.0f32];
    let light_mag_sq = fixed_light_dir[0] * fixed_light_dir[0]
        + fixed_light_dir[1] * fixed_light_dir[1]
        + fixed_light_dir[2] * fixed_light_dir[2];
    let inv_light_mag = 1.0 / light_mag_sq.sqrt();
    let light_dir = [
        fixed_light_dir[0] * inv_light_mag,
        fixed_light_dir[1] * inv_light_mag,
        fixed_light_dir[2] * inv_light_mag,
    ]; // Normalized fixed light direction
    */
    // Light direction L should point FROM surface point TO light source.
    // If light source is at the viewpoint, L is opposite of projection_config.view_dir.
    let light_dir_tuple = (
        -projection_config.view_dir.0,
        -projection_config.view_dir.1,
        -projection_config.view_dir.2,
    );
    let light_dir = [light_dir_tuple.0, light_dir_tuple.1, light_dir_tuple.2]; // Already normalized
    let light_color = [1.0, 1.0, 1.0]; // Reverted light intensity back to 1.0 (white light)
                                       // View direction (vector FROM surface point TO camera/eye)
                                       // Note: For orthographic, this is constant, opposite of the projection direction.

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
                        // --- ADDED: Check for full transparency ---
                        // If opacity is effectively zero, skip shading and writing for this fragment
                        if material.opacity < 1e-6 {
                            // Use a small epsilon
                            return; // Don't draw anything for this fragment
                        }
                        // --- END CHECK ---

                        // 9. Update Buffers: If the depth test passes AND fragment is not fully transparent.
                        // Update the Z-buffer with the new closer depth.
                        z_slice[[r, c]] = interpolated_depth;
                        // Mark this pixel as covered by the mesh in the occlusion map.
                        // NOTE: Even partially transparent objects contribute to occlusion map here.
                        ocp_slice[[r, c]] = 1.0;

                        // Calculate Fragment Color using the dedicated function
                        let tonemapped_linear_color = calculate_fragment_color(
                            material,
                            map_kd,
                            interp_u_coord,
                            interp_v_coord,
                            &normal_f32,
                            &light_dir,
                            &light_dir, // Use light_dir for view_dir (orthographic, light at camera)
                            &light_color,
                        );

                        // --- Alpha Blending ---
                        // Use the result from calculate_fragment_color for blending
                        let src_alpha = material.opacity;
                        let one_minus_src_alpha = 1.0 - src_alpha;
                        let background_color = [1.0f32, 1.0f32, 1.0f32]; // Assuming white background

                        let mut blended_color_f32 = [0.0f32; 3];
                        for i in 0..3 {
                            blended_color_f32[i] = (tonemapped_linear_color[i] * src_alpha)
                                + (background_color[i] * one_minus_src_alpha);
                        }

                        // Clamp final blended color before converting to u8
                        // (Clamping might still be needed if alpha blending produces values slightly outside [0,1])
                        for i in 0..3 {
                            blended_color_f32[i] = blended_color_f32[i].clamp(0.0, 1.0);
                        }

                        // Convert final blended f32 color [0, 1] back to u8 [0, 255] for image buffer
                        let image_channels = image_slice.shape()[2];
                        for chan in 0..image_channels.min(3) {
                            // Write R, G, B from blended color
                            // Convert linear color back to sRGB before storing as u8
                            let srgb_val = blended_color_f32[chan].powf(1.0 / 2.2);
                            // Write the linear, blended, clamped value directly
                            image_slice[[r, c, chan]] = (srgb_val * 255.0) as u8;
                        }
                        // Handle Alpha channel if present - Write the original fragment opacity
                        // Alpha is typically linear, no need to convert.
                        if image_channels == 4 {
                            image_slice[[r, c, 3]] = (src_alpha * 255.0) as u8;
                        }
                    }
                }
            }
        }
    }
}

// Simple struct to hold material properties
#[derive(Debug, Clone, Copy)]
pub struct MaterialProperties {
    pub ka: [f32; 3], // Ambient color
    pub kd: [f32; 3], // Diffuse color
    pub ks: [f32; 3], // Specular color
    pub ns: f32,      // Specular exponent
    pub opacity: f32, // Opacity (0.0=transparent, 1.0=opaque)
    pub illum: u32,   // Illumination model
}
