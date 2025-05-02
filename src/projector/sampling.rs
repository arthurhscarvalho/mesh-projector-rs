// src/projector/sampling.rs
//! Texture and color sampling functions for the projector.

use crate::projector::texture::TextureData;

/// Samples the texture color using bilinear filtering.
/// Assumes texture data is valid.
pub fn sample_texture_color_bilinear(
    interp_u_coord: f64,
    interp_v_coord: f64,
    texture_data: &TextureData<'_>, // Pass TextureData instead of raw array view
) -> [f64; 3] {
    // Return fixed-size array [R, G, B] in [0,1] range
    let texture = texture_data.texture;
    let channels = texture_data.channels;
    let height_f64 = texture_data.height_f64;
    let width_f64 = texture_data.width_f64;

    let tex_xf = interp_u_coord * width_f64;
    // Assuming V=0 is bottom in input UVs based on Python preprocessing flip logic
    let tex_yf = interp_v_coord * height_f64;

    let x0f = tex_xf.max(0.0).min(width_f64 - 1.0001);
    let y0f = tex_yf.max(0.0).min(height_f64 - 1.0001);
    let x0 = x0f as usize;
    let y0 = y0f as usize;

    let x1 = (x0 + 1).min(width_f64 as usize - 1);
    let y1 = (y0 + 1).min(height_f64 as usize - 1);

    let wx = x0f - x0 as f64;
    let wy = y0f - y0 as f64;

    let mut sampled_color = [0.0f64; 3];

    // Assume 3 channels (RGB)
    if channels >= 3 {
        for chan in 0..3 {
            let c00 = texture[[y0, x0, chan]] as f64; // Bottom-left in texture space
            let c10 = texture[[y0, x1, chan]] as f64; // Bottom-right
            let c01 = texture[[y1, x0, chan]] as f64; // Top-left
            let c11 = texture[[y1, x1, chan]] as f64; // Top-right

            // Bilinear interpolation formula might need checking based on coord system
            let interpolated_channel = c00 * (1.0 - wx) * (1.0 - wy)
                + c10 * wx * (1.0 - wy)
                + c01 * (1.0 - wx) * wy
                + c11 * wx * wy;

            sampled_color[chan] = interpolated_channel / 255.0; // Normalize u8 to f64 [0, 1]
        }
    } else {
        // Fallback for grayscale or unexpected channel count? Default to gray.
        let gray_val = if channels == 1 {
            texture[[y0, x0, 0]] as f64 / 255.0
        } else {
            0.5
        };
        sampled_color = [gray_val; 3];
    }
    sampled_color
}
