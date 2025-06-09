//! This module defines the `Orientation` struct for representing 3D orientations: pitch, yaw, and roll.

use ndarray::{array, Array2};

/// Represents a 3D orientation using Euler angles.
///
/// This struct is used to represent orientations in 3D space using
/// pitch (rotation around X-axis), yaw (rotation around Y-axis),
/// and roll (rotation around Z-axis) angles in radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Orientation(pub [f64; 3]);

impl Orientation {
    /// Creates a new `Orientation` from three f64 components.
    /// # Arguments
    /// * `pitch` - Rotation around X-axis (radians)
    /// * `yaw` - Rotation around Y-axis (radians)
    /// * `roll` - Rotation around Z-axis (radians)
    pub fn new(pitch: f64, yaw: f64, roll: f64) -> Self {
        Orientation([pitch, yaw, roll])
    }

    /// Gets the pitch angle (rotation around X-axis) in radians.
    pub fn pitch(&self) -> f64 {
        self.0[0]
    }

    /// Gets the yaw angle (rotation around Y-axis) in radians.
    pub fn yaw(&self) -> f64 {
        self.0[1]
    }

    /// Gets the roll angle (rotation around Z-axis) in radians.
    pub fn roll(&self) -> f64 {
        self.0[2]
    }

    /// Sets the pitch angle (rotation around X-axis) in radians.
    pub fn set_pitch(&mut self, pitch: f64) {
        self.0[0] = pitch;
    }

    /// Sets the yaw angle (rotation around Y-axis) in radians.
    pub fn set_yaw(&mut self, yaw: f64) {
        self.0[1] = yaw;
    }

    /// Sets the roll angle (rotation around Z-axis) in radians.
    pub fn set_roll(&mut self, roll: f64) {
        self.0[2] = roll;
    }

    /// Creates a 3x3 rotation matrix for rotation around the X-axis (pitch).
    pub fn rotation_matrix_x(&self) -> Array2<f64> {
        let cos_a = self.pitch().cos();
        let sin_a = self.pitch().sin();
        array![[1.0, 0.0, 0.0], [0.0, cos_a, -sin_a], [0.0, sin_a, cos_a]]
    }

    /// Creates a 3x3 rotation matrix for rotation around the Y-axis (yaw).
    pub fn rotation_matrix_y(&self) -> Array2<f64> {
        let cos_a = self.yaw().cos();
        let sin_a = self.yaw().sin();
        array![[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]]
    }

    /// Creates a 3x3 rotation matrix for rotation around the Z-axis (roll).
    pub fn rotation_matrix_z(&self) -> Array2<f64> {
        let cos_a = self.roll().cos();
        let sin_a = self.roll().sin();
        array![[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]]
    }

    /// Creates a combined 3x3 rotation matrix from the orientation.
    /// The rotation order is Z * Y * X (roll * yaw * pitch).
    pub fn to_rotation_matrix(&self) -> Array2<f64> {
        let rx = self.rotation_matrix_x();
        let ry = self.rotation_matrix_y();
        let rz = self.rotation_matrix_z();
        rz.dot(&ry).dot(&rx)
    }
}

impl From<[f64; 3]> for Orientation {
    fn from(arr: [f64; 3]) -> Self {
        Orientation(arr)
    }
}

impl From<Orientation> for [f64; 3] {
    fn from(val: Orientation) -> Self {
        val.0
    }
}

impl From<(f64, f64, f64)> for Orientation {
    fn from(tuple: (f64, f64, f64)) -> Self {
        Orientation([tuple.0, tuple.1, tuple.2])
    }
}

impl From<Orientation> for (f64, f64, f64) {
    fn from(val: Orientation) -> Self {
        (val.0[0], val.0[1], val.0[2])
    }
}
