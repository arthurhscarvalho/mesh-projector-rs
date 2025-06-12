//! Provides utilities for sampling colors from textures, including bilinear filtering,
//! to support accurate texture mapping in the projector module.

use crate::types::color::Color;
use ndarray::ArrayView3;

/// Samples the texture color using bilinear filtering.
/// Assumes texture data is valid.
pub fn sample_texture_color_bilinear(
    interp_u_coord: f64,
    interp_v_coord: f64,
    texture: ArrayView3<u8>,
    channels: usize,
    height_f64: f64,
    width_f64: f64,
) -> Color {
    // Return fixed-size array [R, G, B] in [0,1] range
    let tex_xf = interp_u_coord * width_f64;
    // Small fudge factor to ensure that, due to floating-point rounding,
    // you never sample outside the valid range of the texture array
    // when doing bilinear interpolation
    const FUDGE_FACTOR: f64 = 1.0001;
    // Get texture coordinates
    let tex_yf = interp_v_coord * height_f64;
    let x0f = tex_xf.max(0.0).min(width_f64 - FUDGE_FACTOR);
    let y0f = tex_yf.max(0.0).min(height_f64 - FUDGE_FACTOR);
    let x0 = x0f as usize;
    let y0 = y0f as usize;
    let x1 = (x0 + 1).min(width_f64 as usize - 1);
    let y1 = (y0 + 1).min(height_f64 as usize - 1);
    let wx = x0f - x0 as f64;
    let wy = y0f - y0 as f64;
    // Initialize sampled color
    let mut sampled_color = [0.0; 3];
    // If 3 channels (RGB)
    if channels >= 3 {
        for chan in 0..3 {
            let c00 = texture[[y0, x0, chan]] as f64; // Bottom-left in texture space
            let c10 = texture[[y0, x1, chan]] as f64; // Bottom-right
            let c01 = texture[[y1, x0, chan]] as f64; // Top-left
            let c11 = texture[[y1, x1, chan]] as f64; // Top-right
            let interpolated_channel = c00 * (1.0 - wx) * (1.0 - wy)
                + c10 * wx * (1.0 - wy)
                + c01 * (1.0 - wx) * wy
                + c11 * wx * wy;
            sampled_color[chan] = interpolated_channel / 255.0; // Normalize u8 to f64 [0, 1]
        }
    } else {
        panic!("Expected 3 channels (RGB), got {}", channels);
    }
    Color::from_srgb(sampled_color[0], sampled_color[1], sampled_color[2])
}
