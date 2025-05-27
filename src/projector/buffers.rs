// src/projector/buffers.rs
use crate::projector::constants::{BACKGROUND_BRIGHTNESS, NUM_IMAGES};
use ndarray::{s, Array3, Array4};
use std::f64;

/// Holds the output image buffers generated during the projection process:
/// the color images, Z-buffers, depth maps, and normal maps for each of the `NUM_IMAGES` views.
#[derive(Debug)]
pub struct Buffers {
    /// The rendered color images (NUM_IMAGES, H, W, C). Initialized to `BACKGROUND_BRIGHTNESS`.
    pub images: Array4<u8>,
    /// Z-buffers (raw depth values) (NUM_IMAGES, H, W). Stores the depth of the closest fragment.
    /// Initialized to +/- INFINITY depending on the view direction for correct depth testing.
    pub z_buffers: Array3<f64>,
    /// The rendered depth maps (NUM_IMAGES, H, W, C), scaled to `u8`. Initialized to 0.
    pub depth_maps: Array4<u8>,
    /// The rendered normal maps (NUM_IMAGES, H, W, C), with components scaled to `u8`. Initialized to 0.
    pub normal_maps: Array4<u8>,
}

impl Buffers {
    /// Creates new buffers sized according to the specified dimensions and number of images.
    /// Initializes the buffers appropriately for the projection process:
    /// - Images are set to `BACKGROUND_BRIGHTNESS`.
    /// - Z-buffers are initialized to +INFINITY for front views and -INFINITY for back views.
    ///   This ensures the first depth test always passes correctly based on view direction
    ///   (front views check `depth < current_z`, back views check `depth > current_z`).
    /// - Depth maps and Normal maps are initialized to 0.
    pub fn new(num_images: usize, rows: usize, cols: usize, channels: usize) -> Self {
        Buffers {
            images: Array4::from_elem((num_images, rows, cols, channels), BACKGROUND_BRIGHTNESS),
            z_buffers: {
                let mut z = Array3::from_elem((num_images, rows, cols), f64::INFINITY);
                // Initialize back views (odd indices: 1, 3, 5) with NEG_INFINITY
                // For rotation-based views, we only use front views (all +INFINITY)
                if num_images == NUM_IMAGES {
                    for i in (1..num_images).step_by(2) {
                        z.slice_mut(s![i, .., ..]).fill(f64::NEG_INFINITY);
                    }
                }
                z
            },
            depth_maps: Array4::from_elem((num_images, rows, cols, channels), 0u8),
            normal_maps: Array4::from_elem((num_images, rows, cols, channels), 0u8),
        }
    }
}
