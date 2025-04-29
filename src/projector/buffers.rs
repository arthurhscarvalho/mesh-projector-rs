// src/projector/buffers.rs
use crate::projector::constants::{BACKGROUND_BRIGHTNESS, NUM_IMAGES};
use nd::{s, Array3, Array4};
use std::f32;

/// Holds the output image buffers generated during the projection process:
/// the color images and Z-buffers for each of the `NUM_IMAGES` views.
#[derive(Debug)]
pub struct Buffers {
    /// The rendered color images (NUM_IMAGES, H, W, C). Initialized to `BACKGROUND_BRIGHTNESS`.
    pub images: Array4<u8>,
    /// Z-buffers (depth maps) (NUM_IMAGES, H, W). Stores the depth of the closest fragment.
    /// Initialized to +/- INFINITY depending on the view direction for correct depth testing.
    pub z_buffers: Array3<f32>,
}

impl Buffers {
    /// Creates new buffers sized according to the specified dimensions and number of images.
    /// Initializes the buffers appropriately for the projection process:
    /// - Images are set to `BACKGROUND_BRIGHTNESS`.
    /// - Z-buffers are initialized to +INFINITY for front views and -INFINITY for back views.
    ///   This ensures the first depth test always passes correctly based on view direction
    ///   (front views check `depth < current_z`, back views check `depth > current_z`).
    pub fn new(num_images: usize, rows: usize, cols: usize, channels: usize) -> Self {
        // Ensure consistency if NUM_IMAGES constant is used elsewhere
        assert_eq!(
            num_images, NUM_IMAGES,
            "Inconsistent number of images requested in Buffers::new"
        );

        Buffers {
            images: Array4::from_elem((num_images, rows, cols, channels), BACKGROUND_BRIGHTNESS),
            z_buffers: {
                let mut z = Array3::from_elem((num_images, rows, cols), f32::INFINITY);
                // Initialize back views (odd indices: 1, 3, 5) with NEG_INFINITY
                for i in (1..num_images).step_by(2) {
                    z.slice_mut(s![i, .., ..]).fill(f32::NEG_INFINITY);
                }
                z
            },
        }
    }
}
