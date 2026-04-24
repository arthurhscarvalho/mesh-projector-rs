//! Multi-Sample Anti-Aliasing utilities: sample patterns, accumulation and resolve.

use crate::renderer::buffers::ImageBuffers;
use mesh_projector_core::constants::NORMAL_CHANNELS;
use mesh_projector_core::types::color::{srgb_byte_to_linear, Color};
use ndarray::{Array2, Array3};

/// Returns a list of sub-pixel sample offsets for the requested MSAA level.
/// 1× returns a single (0,0) offset (i.e., no MSAA).
pub fn sample_offsets(sample_count: u8) -> Vec<(f32, f32)> {
    match sample_count {
        1 => vec![(0.0, 0.0)],
        2 => vec![(-0.25, -0.25), (0.25, 0.25)],
        4 => vec![(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)],
        8 => vec![
            (-0.375, -0.125),
            (-0.125, -0.375),
            (0.125, -0.125),
            (0.375, -0.375),
            (-0.375, 0.375),
            (-0.125, 0.125),
            (0.125, 0.375),
            (0.375, 0.125),
        ],
        _ => panic!("Unsupported MSAA sample count: {}", sample_count),
    }
}

/// Helper struct accumulating per-sample buffers and resolving them into a final resolved buffer.
pub struct MsaaAccumulator {
    rows: usize,
    cols: usize,
    color_accum: Array3<f32>,  // Premultiplied linear RGB accumulation
    alpha_accum: Array2<f32>,  // Accumulated alpha coverage
    normal_accum: Array3<f32>, // Sum of encoded normals (weighted by alpha)
    depth_min: Array2<u8>,     // Minimum depth per pixel (normalised 0-255)
    z_buffer_min: Array2<f32>, // Minimum raw z depth per pixel
    occupancy: Array2<u8>,     // OR of occupancy per pixel
    background_color: [u8; 4],
}

impl MsaaAccumulator {
    /// Construct a new accumulator for the given image dimensions.
    pub fn new(rows: usize, cols: usize, background_color: [u8; 4]) -> Self {
        Self {
            rows,
            cols,
            color_accum: Array3::<f32>::zeros((rows, cols, 3)),
            alpha_accum: Array2::<f32>::zeros((rows, cols)),
            normal_accum: Array3::<f32>::zeros((rows, cols, NORMAL_CHANNELS)),
            depth_min: Array2::<u8>::from_elem((rows, cols), u8::MAX),
            z_buffer_min: Array2::<f32>::from_elem((rows, cols), f32::INFINITY),
            occupancy: Array2::<u8>::zeros((rows, cols)),
            background_color,
        }
    }

    /// Accumulate one rendered sample pass into the accumulator.
    pub fn accumulate(&mut self, sample_buffers: &ImageBuffers) {
        let img = sample_buffers.image.view();
        let normal = sample_buffers.normal_map.view();
        for r in 0..self.rows {
            for c in 0..self.cols {
                // Occupancy OR
                if sample_buffers.occupancy_map[[r, c]] == 255 {
                    self.occupancy[[r, c]] = 255;
                }
                let alpha = img[[r, c, 3]] as f32 / 255.0;
                self.alpha_accum[[r, c]] += alpha;
                // Colour: convert sRGB byte to linear space and accumulate premultiplied by alpha.
                for ch in 0..3 {
                    let srgb_b = img[[r, c, ch]];
                    let lin = srgb_byte_to_linear(srgb_b);
                    self.color_accum[[r, c, ch]] += lin * alpha;
                }
                // Normal: accumulate encoded values weighted by alpha to avoid background bias.
                for ch in 0..NORMAL_CHANNELS {
                    self.normal_accum[[r, c, ch]] += normal[[r, c, ch]] as f32 * alpha;
                }
                // Depth: keep the minimum (closest) sample.
                let depth_b = sample_buffers.depth_map[[r, c]];
                if depth_b < self.depth_min[[r, c]] {
                    self.depth_min[[r, c]] = depth_b;
                }
                // Z-buffer: keep the minimum raw depth.
                let zb = sample_buffers.z_buffer[[r, c]];
                if zb < self.z_buffer_min[[r, c]] {
                    self.z_buffer_min[[r, c]] = zb;
                }
            }
        }
    }

    /// Resolve accumulated data into a final `ImageBuffers`.
    pub fn resolve(self, sample_count: u8) -> ImageBuffers {
        let inv_samples = 1.0 / sample_count as f32;
        let mut final_buf = ImageBuffers::new(self.rows, self.cols, self.background_color);
        for r in 0..self.rows {
            for c in 0..self.cols {
                // Alpha coverage across samples
                let alpha_total = self.alpha_accum[[r, c]];
                let alpha_avg = (alpha_total * inv_samples).clamp(0.0, 1.0);
                let color_srgb = if alpha_total > 0.0 {
                    let mesh_color_linear = [
                        self.color_accum[[r, c, 0]] / alpha_total,
                        self.color_accum[[r, c, 1]] / alpha_total,
                        self.color_accum[[r, c, 2]] / alpha_total,
                    ];
                    Color::new(
                        mesh_color_linear[0],
                        mesh_color_linear[1],
                        mesh_color_linear[2],
                    )
                    .to_srgb8()
                } else {
                    [
                        self.background_color[0],
                        self.background_color[1],
                        self.background_color[2],
                    ]
                };
                for ch in 0..3 {
                    final_buf.image[[r, c, ch]] = color_srgb[ch];
                }
                let alpha_u8 = (alpha_avg * 255.0).round().clamp(0.0, 255.0) as u8;
                final_buf.image[[r, c, 3]] = alpha_u8;
                // Depth & z
                final_buf.depth_map[[r, c]] = self.depth_min[[r, c]];
                final_buf.z_buffer[[r, c]] = self.z_buffer_min[[r, c]];
                // Normal: average encoded bytes where coverage exists, otherwise leave background (0).
                let normal_denom = if alpha_total > 0.0 { alpha_total } else { 1.0 };
                for ch in 0..NORMAL_CHANNELS {
                    let avg = (self.normal_accum[[r, c, ch]] / normal_denom)
                        .round()
                        .clamp(0.0, 255.0);
                    final_buf.normal_map[[r, c, ch]] = avg as u8;
                }
                final_buf.normal_map[[r, c, 3]] = 255;
                // Occupancy
                final_buf.occupancy_map[[r, c]] = self.occupancy[[r, c]];
            }
        }
        final_buf
    }
}
