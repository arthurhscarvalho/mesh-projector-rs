//! Provides utilities for sampling colors from textures, including bilinear filtering,
//! to support accurate texture mapping in the projector module.

use mesh_projector_core::types::color::Color;
use ndarray::ArrayView3;

/// Samples the texture color using bilinear filtering.
/// Assumes texture data is valid.
pub fn sample_texture_color_bilinear(
    interp_u_coord: f32,
    interp_v_coord: f32,
    texture: ArrayView3<u8>,
    channels: usize,
    height_f32: f32,
    width_f32: f32,
) -> Color {
    // Return fixed-size array [R, G, B] in [0,1] range
    let tex_xf = interp_u_coord * width_f32;
    // Small fudge factor to ensure that, due to floating-point rounding,
    // you never sample outside the valid range of the texture array
    // when doing bilinear interpolation
    const FUDGE_FACTOR: f32 = 1.0001;
    // Get texture coordinates
    let tex_yf = interp_v_coord * height_f32;
    let x0f = tex_xf.max(0.0).min(width_f32 - FUDGE_FACTOR);
    let y0f = tex_yf.max(0.0).min(height_f32 - FUDGE_FACTOR);
    let x0 = x0f as usize;
    let y0 = y0f as usize;
    let x1 = (x0 + 1).min(width_f32 as usize - 1);
    let y1 = (y0 + 1).min(height_f32 as usize - 1);
    let wx = x0f - x0 as f32;
    let wy = y0f - y0 as f32;
    // Initialize sampled color
    let mut sampled_color = [0.0; 3];
    // If 3 channels (RGB)
    if channels >= 3 {
        // Fetch neighboring texels once and convert to linear space via LUT.
        let c00 = [
            texture[[y0, x0, 0]],
            texture[[y0, x0, 1]],
            texture[[y0, x0, 2]],
        ];
        let c10 = [
            texture[[y0, x1, 0]],
            texture[[y0, x1, 1]],
            texture[[y0, x1, 2]],
        ];
        let c01 = [
            texture[[y1, x0, 0]],
            texture[[y1, x0, 1]],
            texture[[y1, x0, 2]],
        ];
        let c11 = [
            texture[[y1, x1, 0]],
            texture[[y1, x1, 1]],
            texture[[y1, x1, 2]],
        ];

        for chan in 0..3 {
            let c00_lin = mesh_projector_core::types::color::srgb_byte_to_linear(c00[chan]);
            let c10_lin = mesh_projector_core::types::color::srgb_byte_to_linear(c10[chan]);
            let c01_lin = mesh_projector_core::types::color::srgb_byte_to_linear(c01[chan]);
            let c11_lin = mesh_projector_core::types::color::srgb_byte_to_linear(c11[chan]);

            let interpolated_lin = c00_lin * (1.0 - wx) * (1.0 - wy)
                + c10_lin * wx * (1.0 - wy)
                + c01_lin * (1.0 - wx) * wy
                + c11_lin * wx * wy;
            sampled_color[chan] = interpolated_lin;
        }
    } else {
        panic!("Expected 3 channels (RGB), got {}", channels);
    }
    Color::new(sampled_color[0], sampled_color[1], sampled_color[2])
}
