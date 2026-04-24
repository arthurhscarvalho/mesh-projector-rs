//! Defines and manages the core data buffers used during 3D mesh projection.
//!
//! This module provides the `Buffers` struct, which encapsulates all intermediate and output
//! arrays required for rendering a projection view, including color images, depth maps,
//! normal maps, occupancy masks, z-buffers, and outlier statistics. These buffers are
//! initialized, updated, and accessed throughout the projection pipeline to accumulate
//! per-pixel results and support post-processing operations.

use mesh_projector_core::constants::{IMAGE_CHANNELS, NORMAL_CHANNELS};
use mesh_projector_core::types::color::Color;
use mesh_projector_core::types::coordinate::Coordinate;
use ndarray::{Array2, Array3, Axis};
use std::f32;

/// Holds buffers generated during the projection process.
pub struct ImageBuffers {
    /// The rendered color image (H, W, C).
    pub image: Array3<u8>,
    /// The rendered depth map (H, W, C).
    pub depth_map: Array2<u8>,
    /// The rendered normal map (H, W, C).
    pub normal_map: Array3<u8>,
    /// The occupancy map (H, W), with 255 for occupied, 0 for empty.
    pub occupancy_map: Array2<u8>,
    /// Z-buffer (raw depth values) (H, W). Stores the depth of the closest fragment.
    pub z_buffer: Array2<f32>,
    /// The number of outliers filtered out.
    pub outlier_count: u64,
    /// Background color stored as sRGBA8.
    pub background_color: [u8; 4],
}

impl ImageBuffers {
    /// Creates new buffers sized according to the specified dimensions.
    /// Initializes the buffers appropriately for the projection process.
    pub fn new(rows: usize, cols: usize, background_color: [u8; 4]) -> Self {
        ImageBuffers {
            image: Array3::from_shape_fn((rows, cols, IMAGE_CHANNELS), |(_, _, ch)| {
                *background_color.get(ch).unwrap_or(&0)
            }),
            depth_map: Array2::zeros((rows, cols)),
            normal_map: Array3::from_shape_fn((rows, cols, NORMAL_CHANNELS), |(_, _, ch)| {
                if ch == 3 {
                    255
                } else {
                    0
                }
            }),
            occupancy_map: Array2::zeros((rows, cols)),
            z_buffer: Array2::from_elem((rows, cols), f32::INFINITY),
            outlier_count: 0,
            background_color,
        }
    }

    /// Vertically flip all buffers in-place.
    pub fn v_flip(&mut self) {
        self.image.invert_axis(Axis(0));
        self.depth_map.invert_axis(Axis(0));
        self.normal_map.invert_axis(Axis(0));
        self.occupancy_map.invert_axis(Axis(0));
        self.z_buffer.invert_axis(Axis(0));
    }

    /// Horizontally flip all buffers in-place.
    pub fn h_flip(&mut self) {
        self.image.invert_axis(Axis(1));
        self.depth_map.invert_axis(Axis(1));
        self.normal_map.invert_axis(Axis(1));
        self.occupancy_map.invert_axis(Axis(1));
        self.z_buffer.invert_axis(Axis(1));
    }
}

pub struct PointCloudBuffers {
    pub points: Vec<f32>,
    pub colors: Vec<u8>,
}

impl PointCloudBuffers {
    pub fn new() -> Self {
        PointCloudBuffers {
            points: Vec::new(),
            colors: Vec::new(),
        }
    }

    pub fn add_point(&mut self, xyz: Coordinate, color: Color) {
        self.points.push(xyz.x());
        self.points.push(xyz.y());
        self.points.push(xyz.z());
        self.colors.push((color.r() * 255.0) as u8);
        self.colors.push((color.g() * 255.0) as u8);
        self.colors.push((color.b() * 255.0) as u8);
    }

    pub fn to_array(self) -> (Array2<f32>, Array2<u8>) {
        let num_points = self.points.len() / 3;
        let points_array = Array2::from_shape_vec((num_points, 3), self.points).unwrap();
        let colors_array = Array2::from_shape_vec((num_points, 3), self.colors).unwrap();
        (points_array, colors_array)
    }
}
