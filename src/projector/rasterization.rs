// src/projector/rasterization.rs
use crate::projector::config::ProjectionConfig;
use crate::projector::geometry::{
    barycentric, calculate_bounding_box, dot_product, is_inside_triangle, is_triangle_visible,
    reflect,
};
use crate::ProjectionMode;
use nd::{Array2, ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3};
use std::f32;

/// Samples the texture color using bilinear filtering.
/// Assumes texture view is valid.
#[inline]
fn sample_texture_color_bilinear(
    interp_u_coord: f32,
    interp_v_coord: f32,
    texture: ArrayView3<u8>, // Pass texture view directly
) -> [f32; 3] {
    // Return fixed-size array [R, G, B] in [0,1] range
    let tex_height = texture.shape()[0];
    let tex_width = texture.shape()[1];
    let channels = texture.shape()[2]; // Should typically be 3

    let width_f32 = tex_width as f32;
    let height_f32 = tex_height as f32;

    let tex_xf = interp_u_coord * width_f32;
    // Assuming V=0 is bottom in input UVs based on Python preprocessing flip logic
    let tex_yf = interp_v_coord * height_f32;

    let x0f = tex_xf.max(0.0).min(width_f32 - 1.0001);
    let y0f = tex_yf.max(0.0).min(height_f32 - 1.0001);
    let x0 = x0f as usize;
    let y0 = y0f as usize;

    let x1 = (x0 + 1).min(tex_width - 1);
    let y1 = (y0 + 1).min(tex_height - 1);

    let wx = x0f - x0 as f32;
    let wy = y0f - y0 as f32;

    let mut sampled_color = [0.0f32; 3];

    // Assume 3 channels (RGB)
    if channels >= 3 {
        for chan in 0..3 {
            let c00 = texture[[y0, x0, chan]] as f32; // Bottom-left in texture space
            let c10 = texture[[y0, x1, chan]] as f32; // Bottom-right
            let c01 = texture[[y1, x0, chan]] as f32; // Top-left
            let c11 = texture[[y1, x1, chan]] as f32; // Top-right

            // Bilinear interpolation formula might need checking based on coord system
            let interpolated_channel = c00 * (1.0 - wx) * (1.0 - wy)
                + c10 * wx * (1.0 - wy)
                + c01 * (1.0 - wx) * wy
                + c11 * wx * wy;

            sampled_color[chan] = interpolated_channel / 255.0; // Normalize u8 to f32 [0, 1]
        }
    } else {
        // Fallback for grayscale or unexpected channel count? Default to gray.
        let gray_val = if channels == 1 {
            texture[[y0, x0, 0]] as f32 / 255.0
        } else {
            0.5
        };
        sampled_color = [gray_val; 3];
    }
    sampled_color
}

/// Calculates the final linear color based on base color, material, lighting, and normal.
#[inline]
fn calculate_lighting(
    material: &MaterialProperties,
    linear_base_color: [f32; 3],
    normal: [f32; 3],
    light_dir: [f32; 3],
    view_dir: [f32; 3],
    light_color: [f32; 3],
) -> [f32; 3] {
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

    let effective_diffuse_linear = [
        linear_base_color[0] * linear_kd[0],
        linear_base_color[1] * linear_kd[1],
        linear_base_color[2] * linear_kd[2],
    ];

    let mut final_color_linear = [0.0f32; 3];
    match material.illum {
        0 => {
            // Base color only
            final_color_linear = linear_base_color;
        }
        1 => {
            // Ambient + Diffuse (Original simple model)
            let ambient_term = linear_ka;
            let dot_nl = dot_product(&normal, &light_dir).max(0.0);
            let diffuse_term = [
                effective_diffuse_linear[0] * dot_nl * light_color[0],
                effective_diffuse_linear[1] * dot_nl * light_color[1],
                effective_diffuse_linear[2] * dot_nl * light_color[2],
            ];
            for i in 0..3 {
                final_color_linear[i] = ambient_term[i] + diffuse_term[i];
            }
        }
        2 | _ => {
            // Illum 2+ (Revised Phong model variation)
            // Ambient term: Modulate Ka by a global ambient factor
            let global_ambient = [0.2, 0.2, 0.2]; // Example low global ambient light
            let ambient_term = [
                linear_ka[0] * global_ambient[0],
                linear_ka[1] * global_ambient[1],
                linear_ka[2] * global_ambient[2],
            ];

            // Diffuse term (without light color initially)
            let dot_nl = dot_product(&normal, &light_dir).max(0.0);
            let diffuse_term = [
                effective_diffuse_linear[0] * dot_nl,
                effective_diffuse_linear[1] * dot_nl,
                effective_diffuse_linear[2] * dot_nl,
            ];

            // Specular term (without light color initially)
            let mut reflected_light_dir = [0.0; 3];
            let neg_light_dir = [-light_dir[0], -light_dir[1], -light_dir[2]];
            reflect(&neg_light_dir, &normal, &mut reflected_light_dir);
            normalize_vec3(&mut reflected_light_dir);
            let dot_rv = dot_product(&reflected_light_dir, &view_dir).max(0.0);
            let specular_intensity = if material.ns > 0.0 {
                dot_rv.powf(material.ns)
            } else {
                0.0
            };
            let specular_term = [
                linear_ks[0] * specular_intensity,
                linear_ks[1] * specular_intensity,
                linear_ks[2] * specular_intensity,
            ];

            // Combine: Final = Ambient + LightColor * (Diffuse + Specular)
            for i in 0..3 {
                final_color_linear[i] =
                    ambient_term[i] + light_color[i] * (diffuse_term[i] + specular_term[i]);
            }
        }
    }
    final_color_linear
}

/// Rasterizes a single triangle based on the projection mode.
#[allow(clippy::too_many_arguments)]
pub fn rasterize_triangle(
    t_idx: usize,                            // Index of the current triangle
    points: ArrayView2<f32>,                 // All vertex 3D positions (N, 3)
    triangles: ArrayView2<usize>,            // Triangle vertex indices (M, 3)
    triangle_uvs: Option<ArrayView2<f32>>,   // Optional UV coords per tri vertex (M*3, 2)
    triangle_normals: &Array2<f32>,          // Precomputed face normals (M, 3) -> For Culling
    vertex_normals: Option<ArrayView2<f32>>, // Optional vertex normals (N, 3) -> For Shading
    vertex_colors: Option<ArrayView2<f32>>,  // Optional vertex colors (N, 3) -> For Shading
    material: &MaterialProperties,           // Material properties
    texture: Option<ArrayView3<u8>>,         // Optional diffuse texture map view
    projection_config: &ProjectionConfig,    // Configuration for the current projection view
    z_slice: &mut ArrayViewMut2<f32>,        // Mutable view to the Z-buffer slice
    image_slice: &mut ArrayViewMut3<u8>,     // Mutable view to the color image buffer slice
    mode: ProjectionMode,                    // Rendering mode
) {
    // 1. Back-face Culling
    let geometric_normal_view = triangle_normals.row(t_idx);
    let face_normal = [
        geometric_normal_view[0],
        geometric_normal_view[1],
        geometric_normal_view[2],
    ];
    if !is_triangle_visible(
        face_normal[0],
        face_normal[1],
        face_normal[2],
        projection_config.view_dir,
    ) {
        return;
    }

    // 2. Get Vertex Data Indices & Positions
    let tri_indices = triangles.row(t_idx);
    let i0 = tri_indices[0];
    let i1 = tri_indices[1];
    let i2 = tri_indices[2];
    let p0 = points.row(i0);
    let p1 = points.row(i1);
    let p2 = points.row(i2);

    // --- Setup Data for Interpolation ---
    let z0 = p0[projection_config.depth_axis];
    let z1 = p1[projection_config.depth_axis];
    let z2 = p2[projection_config.depth_axis];
    // Add small epsilon to avoid division by zero, check if necessary
    let inv_z0 = 1.0 / (z0.abs() + 1e-6) * z0.signum();
    let inv_z1 = 1.0 / (z1.abs() + 1e-6) * z1.signum();
    let inv_z2 = 1.0 / (z2.abs() + 1e-6) * z2.signum();

    // Perspective-correct attributes (attribute / z)
    let mut n0_over_z = (0.0, 0.0, 0.0);
    let mut n1_over_z = (0.0, 0.0, 0.0);
    let mut n2_over_z = (0.0, 0.0, 0.0);
    let mut uv0_over_z = (0.0, 0.0);
    let mut uv1_over_z = (0.0, 0.0);
    let mut uv2_over_z = (0.0, 0.0);
    // Default vertex color is white
    let mut c0_over_z = (inv_z0, inv_z0, inv_z0);
    let mut c1_over_z = (inv_z1, inv_z1, inv_z1);
    let mut c2_over_z = (inv_z2, inv_z2, inv_z2);
    let mut has_normals = false;

    // Setup normals if available
    if let Some(v_normals) = vertex_normals {
        let n0v = v_normals.row(i0);
        let n1v = v_normals.row(i1);
        let n2v = v_normals.row(i2);
        n0_over_z = (n0v[0] * inv_z0, n0v[1] * inv_z0, n0v[2] * inv_z0);
        n1_over_z = (n1v[0] * inv_z1, n1v[1] * inv_z1, n1v[2] * inv_z1);
        n2_over_z = (n2v[0] * inv_z2, n2v[1] * inv_z2, n2v[2] * inv_z2);
        has_normals = true;
    }

    // Setup UVs if available (required for texture modes)
    if mode == ProjectionMode::Full || mode == ProjectionMode::ObjTexture {
        if let Some(uvs) = triangle_uvs {
            let uv_idx_base = t_idx * 3;
            let uv0 = uvs.row(uv_idx_base);
            let uv1 = uvs.row(uv_idx_base + 1);
            let uv2 = uvs.row(uv_idx_base + 2);
            uv0_over_z = (uv0[0] * inv_z0, uv0[1] * inv_z0);
            uv1_over_z = (uv1[0] * inv_z1, uv1[1] * inv_z1);
            uv2_over_z = (uv2[0] * inv_z2, uv2[1] * inv_z2);
        } else {
            // Should have been caught earlier, but default to avoid panic
            uv0_over_z = (0.0, 0.0);
            uv1_over_z = (0.0, 0.0);
            uv2_over_z = (0.0, 0.0);
        }
    }

    // Setup vertex colors if available (required for ObjOnly mode)
    if mode == ProjectionMode::ObjOnly {
        if let Some(v_colors) = vertex_colors {
            let c0v = v_colors.row(i0);
            let c1v = v_colors.row(i1);
            let c2v = v_colors.row(i2);
            // Assume vertex colors are sRGB [0,1], store linear/z for interpolation
            c0_over_z = (
                c0v[0].powf(2.2) * inv_z0,
                c0v[1].powf(2.2) * inv_z0,
                c0v[2].powf(2.2) * inv_z0,
            );
            c1_over_z = (
                c1v[0].powf(2.2) * inv_z1,
                c1v[1].powf(2.2) * inv_z1,
                c1v[2].powf(2.2) * inv_z1,
            );
            c2_over_z = (
                c2v[0].powf(2.2) * inv_z2,
                c2v[1].powf(2.2) * inv_z2,
                c2v[2].powf(2.2) * inv_z2,
            );
        }
        // If None, the default white setup remains.
    }

    // 3. Project Vertices to 2D
    let p0_proj = (p0[projection_config.p_axis], p0[projection_config.q_axis]);
    let p1_proj = (p1[projection_config.p_axis], p1[projection_config.q_axis]);
    let p2_proj = (p2[projection_config.p_axis], p2[projection_config.q_axis]);

    // 4. Calculate Bounding Box
    let (rows, cols) = (z_slice.shape()[0], z_slice.shape()[1]);
    let Some((min_row, max_row, min_col, max_col)) =
        calculate_bounding_box(p0_proj, p1_proj, p2_proj, rows, cols)
    else {
        return;
    };

    // 5. Pixel Loop
    for row in min_row..max_row {
        for col in min_col..max_col {
            let pixel_center = (col as f32 + 0.5, row as f32 + 0.5);

            // 6. Barycentric Coordinates FIRST
            let Some((w0, w1, w2)) = barycentric(p0_proj, p1_proj, p2_proj, pixel_center) else {
                continue;
            };

            // Check if inside using Barycentric coordinates
            if is_inside_triangle(w0, w1, w2) {
                // 7. Perspective Correct Interpolation
                let interpolated_inv_z = w0 * inv_z0 + w1 * inv_z1 + w2 * inv_z2;
                if interpolated_inv_z == 0.0 {
                    continue;
                }
                let interpolated_z = 1.0 / interpolated_inv_z;

                // 8. Depth Test
                let current_z = z_slice[[row, col]];
                let depth_test_passed = if projection_config.is_front_view {
                    interpolated_z < current_z
                } else {
                    interpolated_z > current_z
                };

                if depth_test_passed {
                    // --- Interpolate Attributes ---
                    let interpolated_normal = if has_normals {
                        let bw0 = w0;
                        let bw1 = w1;
                        let bw2 = w2;
                        let nx = (bw0 * n0_over_z.0 + bw1 * n1_over_z.0 + bw2 * n2_over_z.0)
                            * interpolated_z;
                        let ny = (bw0 * n0_over_z.1 + bw1 * n1_over_z.1 + bw2 * n2_over_z.1)
                            * interpolated_z;
                        let nz = (bw0 * n0_over_z.2 + bw1 * n1_over_z.2 + bw2 * n2_over_z.2)
                            * interpolated_z;
                        let mut norm = [nx, ny, nz];
                        normalize_vec3(&mut norm);
                        norm
                    } else {
                        face_normal
                    };

                    let (interpolated_u, interpolated_v) =
                        if mode == ProjectionMode::Full || mode == ProjectionMode::ObjTexture {
                            let u = (w0 * uv0_over_z.0 + w1 * uv1_over_z.0 + w2 * uv2_over_z.0)
                                * interpolated_z;
                            let v = (w0 * uv0_over_z.1 + w1 * uv1_over_z.1 + w2 * uv2_over_z.1)
                                * interpolated_z;
                            (u, v)
                        } else {
                            (0.0, 0.0)
                        };

                    let linear_base_color = match mode {
                        ProjectionMode::ObjOnly => {
                            let r = (w0 * c0_over_z.0 + w1 * c1_over_z.0 + w2 * c2_over_z.0)
                                * interpolated_z;
                            let g = (w0 * c0_over_z.1 + w1 * c1_over_z.1 + w2 * c2_over_z.1)
                                * interpolated_z;
                            let b = (w0 * c0_over_z.2 + w1 * c1_over_z.2 + w2 * c2_over_z.2)
                                * interpolated_z;
                            [r.max(0.0), g.max(0.0), b.max(0.0)]
                        }
                        ProjectionMode::ObjTexture | ProjectionMode::Full => {
                            if let Some(tex) = texture {
                                let sampled_srgb = sample_texture_color_bilinear(
                                    interpolated_u,
                                    interpolated_v,
                                    tex,
                                );
                                [
                                    sampled_srgb[0].powf(2.2),
                                    sampled_srgb[1].powf(2.2),
                                    sampled_srgb[2].powf(2.2),
                                ]
                            } else {
                                [1.0, 0.0, 1.0]
                            }
                        }
                    };

                    // --- Calculate Final Color (Lighting) ---
                    let view_dir_tuple = projection_config.view_dir;
                    let view_dir_array = [view_dir_tuple.0, view_dir_tuple.1, view_dir_tuple.2];
                    let light_dir_array =
                        [-view_dir_array[0], -view_dir_array[1], -view_dir_array[2]];
                    let light_color = [1.0, 1.0, 1.0];

                    let final_linear_color = calculate_lighting(
                        material,
                        linear_base_color,
                        interpolated_normal,
                        light_dir_array,
                        view_dir_array,
                        light_color,
                    );

                    // --- Gamma Correction (Linear -> sRGB) & Clamping ---
                    let final_color_srgb = [
                        final_linear_color[0].powf(1.0 / 2.2).max(0.0).min(1.0),
                        final_linear_color[1].powf(1.0 / 2.2).max(0.0).min(1.0),
                        final_linear_color[2].powf(1.0 / 2.2).max(0.0).min(1.0),
                    ];

                    // 9. Update Buffers
                    z_slice[[row, col]] = interpolated_z;
                    image_slice[[row, col, 0]] = (final_color_srgb[0] * 255.0) as u8;
                    image_slice[[row, col, 1]] = (final_color_srgb[1] * 255.0) as u8;
                    image_slice[[row, col, 2]] = (final_color_srgb[2] * 255.0) as u8;
                }
            }
        }
    }
}

/// Normalizes a 3D vector in place. Handles zero vector case.
#[inline]
fn normalize_vec3(v: &mut [f32; 3]) {
    let norm_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if norm_sq > 1e-9 {
        // Use a threshold slightly larger than 0 for robustness
        let norm = norm_sq.sqrt();
        v[0] /= norm;
        v[1] /= norm;
        v[2] /= norm;
    } else {
        // Handle zero vector case (e.g., set to a default direction or leave as zero)
        v[0] = 0.0;
        v[1] = 0.0;
        v[2] = 1.0; // Example: Default to pointing up Z-axis
    }
}

/// Material properties matching typical MTL definitions.
#[derive(Debug, Clone)]
pub struct MaterialProperties {
    pub ka: [f32; 3], // Ambient color
    pub kd: [f32; 3], // Diffuse color
    pub ks: [f32; 3], // Specular color
    pub ns: f32,      // Specular exponent
    #[allow(dead_code)]
    pub opacity: f32, // Opacity (0.0=transparent, 1.0=opaque) - Currently unused in shader
    pub illum: i32,   // Illumination model
}
