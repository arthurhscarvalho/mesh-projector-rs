// src/projector/buffers.rs
use crate::projector::constants::BACKGROUND_BRIGHTNESS;
use ndarray::{Array3, Array4};
use std::f64;

/// Holds the output image buffers generated during the projection process.
#[derive(Debug)]
pub struct Buffers {
    /// The rendered color images (NUM_IMAGES, H, W, C). Initialized to `BACKGROUND_BRIGHTNESS`.
    pub images: Array4<u8>,
    /// Z-buffers (raw depth values) (NUM_IMAGES, H, W). Stores the depth of the closest fragment.
    pub z_buffers: Array3<f64>,
    /// The rendered depth maps (NUM_IMAGES, H, W, C), scaled to `u8`. Initialized to 0.
    pub depth_maps: Array4<u8>,
    /// The rendered normal maps (NUM_IMAGES, H, W, C), with components scaled to `u8`. Initialized to 0.
    pub normal_maps: Array4<u8>,
}

impl Buffers {
    /// Creates new buffers sized according to the specified dimensions and number of images.
    /// Initializes the buffers appropriately for the projection process.
    pub fn new(num_images: usize, rows: usize, cols: usize, channels: usize) -> Self {
        Buffers {
            images: Array4::from_elem((num_images, rows, cols, channels), BACKGROUND_BRIGHTNESS),
            z_buffers: Array3::from_elem((num_images, rows, cols), f64::INFINITY),
            depth_maps: Array4::from_elem((num_images, rows, cols, channels), 0u8),
            normal_maps: Array4::from_elem((num_images, rows, cols, channels), 0u8),
        }
    }
}
