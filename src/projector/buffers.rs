// src/projector/buffers.rs
use crate::projector::constants::BACKGROUND_BRIGHTNESS;
use ndarray::{Array2, Array3};
use std::f64;

/// Holds the output image buffers generated during the projection process.
pub struct Buffers {
    /// The rendered color image (H, W, C).
    pub image: Array3<u8>,
    /// The rendered depth map (H, W, C).
    pub depth_map: Array3<u8>,
    /// The rendered normal map (H, W, C).
    pub normal_map: Array3<u8>,
    /// The occupancy map (H, W), with 255 for occupied, 0 for empty.
    pub occupancy_map: Array2<u8>,
    /// Z-buffer (raw depth values) (H, W). Stores the depth of the closest fragment.
    pub z_buffer: Array2<f64>,
    /// The number of outliers filtered out.
    pub outlier_count: u64,
}

impl Buffers {
    /// Creates new buffers sized according to the specified dimensions.
    /// Initializes the buffers appropriately for the projection process.
    pub fn new(rows: usize, cols: usize, channels: usize) -> Self {
        Buffers {
            image: Array3::from_elem((rows, cols, channels), BACKGROUND_BRIGHTNESS),
            depth_map: Array3::zeros((rows, cols, channels)),
            normal_map: Array3::zeros((rows, cols, channels)),
            occupancy_map: Array2::zeros((rows, cols)),
            z_buffer: Array2::from_elem((rows, cols), f64::INFINITY),
            outlier_count: 0,
        }
    }
}
