// src/projector/rasterization.rs
use crate::projector::config::ProjectionConfig;
use crate::projector::geometry::{
    barycentric, calculate_bounding_box, is_inside_triangle, is_triangle_visible,
};
use crate::projector::texture::TextureData;
use nd::{Array2, ArrayView2, ArrayViewMut2, ArrayViewMut3};
use std::f32;

/// Interpolates texture coordinates using barycentric coordinates and samples the texture color
/// using bilinear filtering. Returns the sampled color as [f32; channels].
/// Assumes standard UV mapping where (0,0) is top-left and (1,1) is bottom-right.
/// Handles clamping coordinates to the texture boundaries.
#[inline]
fn sample_texture_color_bilinear(
    interp_u_coord: f32, // Perspective-correct interpolated U coordinate [0, 1]
    interp_v_coord: f32, // Perspective-correct interpolated V coordinate [0, 1]
    texture_data: &TextureData, // Contains texture view and dimensions
) -> Vec<f32> {
    // Return type changed
    // --- Bilinear Filtering Implementation ---

    // Calculate floating-point texture coordinates (px, py)
    let tex_xf = interp_u_coord * texture_data.width_f32;
    let tex_yf = (1.0 - interp_v_coord) * texture_data.height_f32;

    // Calculate the integer coordinates of the top-left texel (x0, y0)
    let x0f = tex_xf.max(0.0).min(texture_data.width_f32 - 1.0001);
    let y0f = tex_yf.max(0.0).min(texture_data.height_f32 - 1.0001);
    let x0 = x0f as usize;
    let y0 = y0f as usize;

    // Calculate the integer coordinates of the bottom-right texel (x1, y1)
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    // Calculate the fractional parts (weights) for interpolation
    let wx = x0f - x0 as f32;
    let wy = y0f - y0 as f32;

    // Sample the four neighboring texels
    let tex = &texture_data.texture;
    let c = texture_data.channels;
    let mut sampled_color = vec![0.0f32; c]; // Allocate vector for result

    for chan in 0..c {
        let c00 = tex[[y0, x0, chan]] as f32; // Top-left
        let c10 = tex[[y0, x1, chan]] as f32; // Top-right
        let c01 = tex[[y1, x0, chan]] as f32; // Bottom-left
        let c11 = tex[[y1, x1, chan]] as f32; // Bottom-right

        // Perform bilinear interpolation
        let interpolated_channel = c00 * (1.0 - wx) * (1.0 - wy) // Top-left influence
                               + c10 * wx * (1.0 - wy)         // Top-right influence
                               + c01 * (1.0 - wx) * wy         // Bottom-left influence
                               + c11 * wx * wy; // Bottom-right influence

        sampled_color[chan] = interpolated_channel / 255.0; // Normalize to [0, 1] range
    }
    sampled_color // Return the vec
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
    let mut normal: [f32; 3] = [normal_view[0], normal_view[1], normal_view[2]];
    // --- Normalize the normal vector for lighting calculations ---
    let mag_sq = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
    if mag_sq > 1e-9 {
        // Avoid division by zero or near-zero for degenerate triangles
        let inv_mag = 1.0 / mag_sq.sqrt();
        normal[0] *= inv_mag;
        normal[1] *= inv_mag;
        normal[2] *= inv_mag;
    } else {
        // Handle degenerate triangle normal (e.g., set to a default like [0,0,1] or skip lighting)
        // For now, let it be [0,0,0], which will result in zero diffuse/specular
        normal = [0.0, 0.0, 0.0];
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
    // Use a fixed diagonal light source instead of view direction
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
       // let light_dir_tuple = projection_config.view_dir; // Old: Use view direction as light direction
       // let light_dir = [light_dir_tuple.0, light_dir_tuple.1, light_dir_tuple.2]; // Old: Already normalized
    let light_color = [1.0, 1.0, 1.0]; // White light
                                       // View direction (vector FROM surface point TO camera/eye)
                                       // Note: For orthographic, this is constant, opposite of the projection direction.
    let view_dir_tuple = (
        -projection_config.view_dir.0,
        -projection_config.view_dir.1,
        -projection_config.view_dir.2,
    );
    let view_dir = [view_dir_tuple.0, view_dir_tuple.1, view_dir_tuple.2]; // Already normalized
                                                                           // --- End Lighting Setup ---

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

                        // --- Restore Original Shading Logic ---

                        // 10. Shading Calculation (Based on Illumination Model)
                        let mut final_color_f32 = [0.0f32; 3]; // Use f32 for calculations

                        // Calculate effective diffuse color: Texture * Kd (if texture), or just Kd.
                        let effective_diffuse = if let Some(tex_data) = map_kd {
                            let sampled_rgb = sample_texture_color_bilinear(
                                interp_u_coord,
                                interp_v_coord,
                                tex_data,
                            );

                            // --- DEBUG LOGGING for Texture Sampling ---
                            if r % 100 == 0 && c % 100 == 0 {
                                // Log occasionally
                                log::debug!(
                                    "t_idx: {}, r: {}, c: {}, uv_in: ({:.3}, {:.3}), sampled_rgb: {:?}",
                                    t_idx, r, c, interp_u_coord, interp_v_coord, sampled_rgb
                                );
                            }
                            // --- END DEBUG LOGGING ---

                            // Modulate texture color by Kd
                            let mut modulated_color = [0.0f32; 3];
                            if sampled_rgb.len() >= 3 {
                                for i in 0..3 {
                                    modulated_color[i] = sampled_rgb[i] * material.kd[i];
                                }
                            } else if sampled_rgb.len() == 1 {
                                // Handle grayscale texture
                                for i in 0..3 {
                                    modulated_color[i] = sampled_rgb[0] * material.kd[i];
                                }
                            } else {
                                // Fallback
                                modulated_color = material.kd; // Use Kd if texture is invalid
                            }
                            modulated_color // This is Texture * Kd
                        } else {
                            // If no texture, use material diffuse color directly
                            material.kd
                        };

                        // --- Illumination Model Logic (Using effective_diffuse) ---
                        match material.illum {
                            0 => {
                                // Color on and Ambient off
                                final_color_f32 = effective_diffuse;
                            }
                            1 => {
                                // Color on and Ambient on
                                // final_color = Ka + effective_diffuse
                                for i in 0..3 {
                                    final_color_f32[i] = material.ka[i] + effective_diffuse[i];
                                }
                            }
                            2 => {
                                // Highlight on (Phong)
                                // FinalColor = GlobalAmbient*Ka + LightColor*(Diffuse + Specular)

                                // Ambient Component (using Ka and a global ambient factor)
                                let global_ambient_light = [0.25, 0.25, 0.25]; // Increased ambient factor
                                let ambient_term = [
                                    global_ambient_light[0] * material.ka[0],
                                    global_ambient_light[1] * material.ka[1],
                                    global_ambient_light[2] * material.ka[2],
                                ];

                                // Diffuse Component (use effective_diffuse, apply lighting)
                                let dot_nl = dot_product(&normal, &light_dir).max(0.0);
                                let diffuse_term =
                                    vec_scale_ret(&effective_diffuse, dot_nl * light_color[0]); // Use effective_diffuse

                                // Specular Component (LightColor * Ks * pow(max(0, R.V), Ns))
                                let mut reflected_light_dir = [0.0; 3];
                                reflect(
                                    &vec_scale_ret(&light_dir, -1.0),
                                    &normal,
                                    &mut reflected_light_dir,
                                ); // Use NORMALIZED normal for reflection
                                let dot_rv = dot_product(&reflected_light_dir, &view_dir).max(0.0);
                                let specular_intensity = if material.ns > 0.0 {
                                    dot_rv.powf(material.ns)
                                } else {
                                    0.0
                                }; // Handle Ns=0
                                let specular_term = vec_scale_ret(
                                    &material.ks,
                                    specular_intensity * light_color[0],
                                ); // Assuming white light [1,1,1]

                                // Combine components (Ambient + Diffuse + Specular)
                                for i in 0..3 {
                                    final_color_f32[i] = ambient_term[i]  // Ambient term
                                                      + diffuse_term[i] // Diffuse term
                                                      + specular_term[i]; // Specular term
                                }
                            }
                            _ => {
                                // Default/Unsupported: Use effective_diffuse color only
                                final_color_f32 = effective_diffuse;
                            }
                        }

                        // Apply Opacity (Modulate final color by opacity)
                        vec_scale_inplace(&mut final_color_f32, material.opacity);

                        // Clamp final color *after* opacity modulation and before converting to u8
                        for i in 0..3 {
                            final_color_f32[i] = final_color_f32[i].clamp(0.0, 1.0);
                        }

                        // Convert final f32 color [0, 1] back to u8 [0, 255] for image buffer
                        let image_channels = image_slice.shape()[2];
                        for chan in 0..image_channels.min(3) {
                            // Write R, G, B
                            image_slice[[r, c, chan]] = (final_color_f32[chan] * 255.0) as u8;
                        }
                        // Handle Alpha channel if present
                        if image_channels == 4 {
                            image_slice[[r, c, 3]] = (material.opacity * 255.0) as u8;
                        }
                        // --- END ORIGINAL SHADING LOGIC ---
                    }
                }
            }
        }
    }
}

// Helper function needed for reflect call above (ndarray doesn't have simple vec scale returning new vec)
#[inline]
fn vec_scale_ret(a: &[f32], s: f32) -> Vec<f32> {
    a.iter().map(|x| x * s).collect()
}

// --- Helper functions for basic vector math ---
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn vec_sub(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..out.len() {
        out[i] = a[i] - b[i];
    }
}

#[inline]
fn vec_scale_inplace(a: &mut [f32], s: f32) {
    // New inplace scaling function
    for i in 0..a.len() {
        a[i] *= s;
    }
}

#[inline]
fn reflect(neg_l: &[f32], n: &[f32], out: &mut [f32]) {
    let dot_nl = dot_product(n, neg_l);
    // Use a temporary variable to avoid mutable borrow conflict
    let mut temp = [0.0f32; 3]; // Assuming 3D vectors
    vec_scale_inplace(&mut temp, 2.0 * dot_nl); // temp = 2 * dot(N, -L) * N
    vec_sub(neg_l, &temp, out); // out = neg_l - temp
}

// Simple struct to hold material properties (replace with proper MTL parsing later)
#[derive(Debug, Clone, Copy)] // Add derive traits for easier use
pub struct MaterialProperties {
    pub ka: [f32; 3], // Ambient color
    pub kd: [f32; 3], // Diffuse color
    pub ks: [f32; 3], // Specular color
    pub ns: f32,      // Specular exponent
    pub opacity: f32, // Opacity (0.0=transparent, 1.0=opaque)
    pub illum: u32,   // Illumination model
}
