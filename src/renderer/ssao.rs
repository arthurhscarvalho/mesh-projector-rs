//! Screen-space ambient occlusion (SSAO) post-processing utilities.
//!
//! The implementation follows the classic SSAO technique using hemisphere sampling in
//! view space, leveraging the resolved depth and normal buffers to compute an occlusion
//! factor that darkens the final color image.

use crate::renderer::buffers::ImageBuffers;
use crate::renderer::config::{ProjectionModel, RenderPassConfig, ViewConfig};
use crate::types::color::{srgb_byte_to_linear, Color};
use ndarray::{Array2, ArrayView2};
use std::f32::consts::PI;

/// Primary entry point: compute SSAO from the supplied buffers and darken the color image.
pub fn apply_ssao(buffers: &mut ImageBuffers, render_pass_config: &RenderPassConfig) {
    let config = render_pass_config.ssao;
    if !config.enabled {
        return;
    }

    let mut occlusion = compute_occlusion_map(buffers, render_pass_config);
    if config.blur_radius > 0 {
        occlusion = bilateral_blur_occlusion_map(&occlusion.view(), config.blur_radius, 0.5);
    }
    darken_image(buffers, &occlusion.view(), config.intensity);
}

/// Computes an occlusion map in the range `[0, 1]` where `0` means fully lit
/// and `1` means fully occluded.
fn compute_occlusion_map(
    buffers: &ImageBuffers,
    render_pass_config: &RenderPassConfig,
) -> Array2<f32> {
    let config = render_pass_config.ssao;
    let (rows, cols) = buffers.z_buffer.dim();
    let mut occ_map = Array2::<f32>::zeros((rows, cols));

    if config.num_samples == 0 || config.sample_radius <= 0.0 || config.max_distance <= 0.0 {
        return occ_map;
    }

    // Generate hemisphere sample vectors
    let sample_vectors = generate_poisson_hemisphere_samples(config.num_samples);

    for r in 0..rows {
        for c in 0..cols {
            // Skip background pixels
            if buffers.occupancy_map[[r, c]] == 0 {
                continue;
            }

            let depth = buffers.z_buffer[[r, c]];
            if !depth.is_finite() || depth == f32::INFINITY {
                continue;
            }

            let Some(view_pos) =
                reconstruct_view_position(r, c, depth, rows, cols, &render_pass_config.view)
            else {
                continue;
            };

            // Decode surface normal from normal map (convert from [0,255] to [-1,1])
            let normal = decode_normal_from_buffer(buffers, r, c);
            if vec3_length_squared(&normal) < 1e-6 {
                continue; // Skip invalid normals
            }
            let normal = vec3_normalize(&normal);

            // Create tangent-bitangent-normal matrix for hemisphere orientation
            let (tangent, bitangent) = create_tangent_basis(&normal);

            // Generate per-pixel random rotation to break up repetition patterns
            let random_rotation = generate_pixel_noise(r, c);

            let mut occlusion = 0.0;
            let mut valid_samples = 0;

            // Sample points in hemisphere around this pixel
            for sample_vec in &sample_vectors {
                // Apply random rotation to sample
                let rotated_sample = rotate_sample_2d(sample_vec, random_rotation);

                // Transform sample from hemisphere space to view space
                let sample_view = transform_sample_to_view_space(
                    &rotated_sample,
                    &view_pos,
                    &normal,
                    &tangent,
                    &bitangent,
                    config.sample_radius,
                );

                if sample_view[2] <= 0.0 {
                    continue;
                }

                // Project sample back to screen space
                let Some((sample_r, sample_c)) =
                    view_to_screen_space(&sample_view, rows, cols, &render_pass_config.view)
                else {
                    continue;
                };

                // Check if sample is within screen bounds
                if sample_r < 0
                    || sample_r >= rows as isize
                    || sample_c < 0
                    || sample_c >= cols as isize
                {
                    continue;
                }

                let screen_r = sample_r as usize;
                let screen_c = sample_c as usize;

                // Skip unoccupied pixels in sample location
                if buffers.occupancy_map[[screen_r, screen_c]] == 0 {
                    continue;
                }

                let sample_depth = buffers.z_buffer[[screen_r, screen_c]];
                if !sample_depth.is_finite() || sample_depth == f32::INFINITY {
                    continue;
                }

                // Check for occlusion with bias and distance attenuation
                let distance = vec3_length(&vec3_sub(sample_view, view_pos));
                if distance > config.max_distance {
                    continue;
                }

                // The sample is occluded if there's geometry closer to the camera
                if sample_depth < sample_view[2] - config.bias {
                    // Apply distance-based attenuation
                    let normalized_dist = distance / config.max_distance;
                    let attenuation = 1.0 - (normalized_dist * normalized_dist); // Quadratic falloff

                    occlusion += attenuation.max(0.0);
                }

                valid_samples += 1;
            }

            // Normalize occlusion by number of valid samples
            if valid_samples > 0 {
                let raw_occlusion = occlusion / valid_samples as f32;
                occ_map[[r, c]] = raw_occlusion.clamp(0.0, 1.0);
            }
        }
    }
    occ_map
}

/// Generates hemisphere sample vectors using Poisson disk sampling for better distribution.
fn generate_poisson_hemisphere_samples(num_samples: usize) -> Vec<[f32; 3]> {
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let i_f = i as f32;

        // Use a better distribution with stratified sampling
        let u1 = (i_f + 0.5) / num_samples as f32;
        let u2 = (i_f * 0.618_034) % 1.0; // Golden ratio for phi

        // Generate points on hemisphere with cosine-weighted distribution
        let cos_theta = u1.sqrt();
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi = 2.0 * PI * u2;

        // Scale samples to be biased toward the normal (more realistic occlusion)
        let scale = 0.1 + 0.9 * u1; // Bias samples toward surface

        samples.push([
            sin_theta * phi.cos() * scale,
            sin_theta * phi.sin() * scale,
            cos_theta * scale,
        ]);
    }

    samples
}

/// Generates per-pixel noise for randomizing sample orientations.
fn generate_pixel_noise(row: usize, col: usize) -> f32 {
    // Simple hash-based noise generation for consistent per-pixel randomness
    let mut hash = (row as u32).wrapping_mul(73856093) ^ (col as u32).wrapping_mul(19349663);
    hash = hash.wrapping_mul(83492791);
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x7feb352d);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x846ca68b);
    hash ^= hash >> 16;

    // Convert to float in [0, 2π]
    (hash as f32 / u32::MAX as f32) * 2.0 * PI
}

/// Rotates a sample vector around the Z-axis by the given angle.
fn rotate_sample_2d(sample: &[f32; 3], angle: f32) -> [f32; 3] {
    let cos_angle = angle.cos();
    let sin_angle = angle.sin();

    [
        sample[0] * cos_angle - sample[1] * sin_angle,
        sample[0] * sin_angle + sample[1] * cos_angle,
        sample[2],
    ]
}

/// Reconstructs view space position from screen coordinates and depth.
fn reconstruct_view_position(
    row: usize,
    col: usize,
    depth: f32,
    rows: usize,
    cols: usize,
    view: &ViewConfig,
) -> Option<[f32; 3]> {
    if !depth.is_finite() || depth <= 0.0 {
        return None;
    }
    let row_f = row as f32 + 0.5;
    let col_f = col as f32 + 0.5;

    match &view.projection {
        ProjectionModel::Orthographic => {
            // Convert screen coordinates to NDC [-1, 1]
            let ndc_x = (col_f / cols as f32) * 2.0 - 1.0;
            let ndc_y = (row_f / rows as f32) * 2.0 - 1.0;
            Some([ndc_x, ndc_y, depth])
        }
        ProjectionModel::Perspective(params) => {
            let cube_center = (view.max_bound_f32 - 1.0) * 0.5;
            let x_cam = (col_f - cube_center) * depth / params.focal_length_px;
            let y_cam = (row_f - cube_center) * depth / params.focal_length_px;
            Some([x_cam, y_cam, depth])
        }
    }
}

/// Decodes normal from the normal map buffer (converts from [0,255] to [-1,1] range).
fn decode_normal_from_buffer(buffers: &ImageBuffers, row: usize, col: usize) -> [f32; 3] {
    let nx = (buffers.normal_map[[row, col, 0]] as f32 / 255.0) * 2.0 - 1.0;
    let ny = (buffers.normal_map[[row, col, 1]] as f32 / 255.0) * 2.0 - 1.0;
    let nz = (buffers.normal_map[[row, col, 2]] as f32 / 255.0) * 2.0 - 1.0;
    [nx, ny, nz]
}

/// Normalizes a vector.
fn vec3_normalize(v: &[f32; 3]) -> [f32; 3] {
    let len = vec3_length(v);
    if len > f32::EPSILON {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0] // Default to up vector
    }
}

/// Creates a tangent basis for orienting the hemisphere sampling.
fn create_tangent_basis(normal: &[f32; 3]) -> ([f32; 3], [f32; 3]) {
    // Choose an arbitrary vector not parallel to the normal
    let up = if normal[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };

    // Compute tangent as cross product of normal and up
    let tangent = vec3_cross(normal, &up);
    let tangent = vec3_normalize(&tangent);

    // Compute bitangent as cross product of normal and tangent
    let bitangent = vec3_cross(normal, &tangent);

    (tangent, bitangent)
}

/// Cross product of two 3D vectors.
fn vec3_cross(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Transforms a hemisphere sample to view space using TBN matrix.
fn transform_sample_to_view_space(
    sample: &[f32; 3],
    view_pos: &[f32; 3],
    normal: &[f32; 3],
    tangent: &[f32; 3],
    bitangent: &[f32; 3],
    radius: f32,
) -> [f32; 3] {
    // Scale sample by radius
    let scaled_sample = [sample[0] * radius, sample[1] * radius, sample[2] * radius];

    // Transform to view space using TBN matrix
    let world_sample = [
        tangent[0] * scaled_sample[0]
            + bitangent[0] * scaled_sample[1]
            + normal[0] * scaled_sample[2],
        tangent[1] * scaled_sample[0]
            + bitangent[1] * scaled_sample[1]
            + normal[1] * scaled_sample[2],
        tangent[2] * scaled_sample[0]
            + bitangent[2] * scaled_sample[1]
            + normal[2] * scaled_sample[2],
    ];

    // Add to view position
    [
        view_pos[0] + world_sample[0],
        view_pos[1] + world_sample[1],
        view_pos[2] + world_sample[2],
    ]
}

/// Projects view space position back to screen space coordinates.
fn view_to_screen_space(
    view_pos: &[f32; 3],
    rows: usize,
    cols: usize,
    view: &ViewConfig,
) -> Option<(isize, isize)> {
    match &view.projection {
        ProjectionModel::Orthographic => {
            // Convert from view space (assumed NDC) to screen coordinates
            let screen_x = ((view_pos[0] + 1.0) * 0.5 * cols as f32).round() as isize;
            let screen_y = ((view_pos[1] + 1.0) * 0.5 * rows as f32).round() as isize;
            Some((screen_y, screen_x))
        }
        ProjectionModel::Perspective(params) => {
            if view_pos[2] <= 0.0 {
                return None;
            }
            let cube_center = (view.max_bound_f32 - 1.0) * 0.5;
            let scale = params.focal_length_px / view_pos[2];
            let col = scale * view_pos[0] + cube_center;
            let row = scale * view_pos[1] + cube_center;
            Some((row.round() as isize, col.round() as isize))
        }
    }
}

/// Helper function to compute vector length.
fn vec3_length(v: &[f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Helper function to compute vector length squared.
fn vec3_length_squared(v: &[f32; 3]) -> f32 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

/// Helper function for vector subtraction.
fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Applies a bilateral blur that preserves edges while reducing noise.
fn bilateral_blur_occlusion_map(
    src: &ArrayView2<f32>,
    radius: usize,
    sigma_color: f32,
) -> Array2<f32> {
    if radius == 0 {
        return src.to_owned();
    }

    let (rows, cols) = src.dim();
    let mut dst = Array2::<f32>::zeros((rows, cols));
    let rad = radius as isize;
    let sigma_color_sq = sigma_color * sigma_color;

    for r in 0..rows as isize {
        for c in 0..cols as isize {
            let center_val = src[[r as usize, c as usize]];
            let mut sum = 0.0;
            let mut weight_sum = 0.0;

            for dy in -rad..=rad {
                let rr = r + dy;
                if rr < 0 || rr >= rows as isize {
                    continue;
                }

                for dx in -rad..=rad {
                    let cc = c + dx;
                    if cc < 0 || cc >= cols as isize {
                        continue;
                    }

                    let neighbor_val = src[[rr as usize, cc as usize]];

                    // Spatial weight (Gaussian)
                    let spatial_dist_sq = (dx * dx + dy * dy) as f32;
                    let spatial_weight =
                        (-spatial_dist_sq / (2.0 * radius as f32 * radius as f32)).exp();

                    // Color weight (based on value difference)
                    let color_diff = (center_val - neighbor_val).abs();
                    let color_weight = (-color_diff * color_diff / (2.0 * sigma_color_sq)).exp();

                    let weight = spatial_weight * color_weight;
                    sum += neighbor_val * weight;
                    weight_sum += weight;
                }
            }

            dst[[r as usize, c as usize]] = if weight_sum > 0.0 {
                sum / weight_sum
            } else {
                center_val
            };
        }
    }
    dst
}

/// Darkens the colour buffer in-place based on the computed occlusion map.
fn darken_image(buffers: &mut ImageBuffers, occlusion: &ArrayView2<f32>, intensity: f32) {
    let (rows, cols) = occlusion.dim();
    let color_channels = buffers.image.shape()[2].min(3);
    for r in 0..rows {
        for c in 0..cols {
            // Skip unoccupied pixels to preserve the background color.
            if buffers.occupancy_map[[r, c]] == 0 {
                continue;
            }
            let occ = (occlusion[[r, c]] * intensity).clamp(0.0, 1.0);
            let factor = 1.0 - occ;

            // Fetch existing colour, convert to linear, apply factor.
            let mut lin = [0.0; 3];
            for ch in 0..color_channels {
                lin[ch] = srgb_byte_to_linear(buffers.image[[r, c, ch]]);
                lin[ch] *= factor;
            }
            let color = Color::new(lin[0], lin[1], lin[2]);
            let srgb = color.to_srgb8();
            for ch in 0..color_channels {
                buffers.image[[r, c, ch]] = srgb[ch];
            }
        }
    }
}
