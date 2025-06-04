// src/projector/math.rs
//! Vector math operations for 3D geometry calculations.
//! Provides basic operations like dot product, vector scaling, subtraction, reflection, etc.

use ndarray::{array, Array2};

/// Calculates the dot product of two vector slices.
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Normalizes a 3D vector in-place.
pub fn normalize_vec3(v: &mut [f64; 3]) {
    let length_squared = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if length_squared > 0.0 {
        let inv_length = 1.0 / length_squared.sqrt();
        v[0] *= inv_length;
        v[1] *= inv_length;
        v[2] *= inv_length;
    }
}

/// Creates a 3x3 rotation matrix for rotation around the X-axis.
pub fn rotation_matrix_x(angle_rad: f64) -> Array2<f64> {
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    array![[1.0, 0.0, 0.0], [0.0, cos_a, -sin_a], [0.0, sin_a, cos_a]]
}

/// Creates a 3x3 rotation matrix for rotation around the Y-axis.
pub fn rotation_matrix_y(angle_rad: f64) -> Array2<f64> {
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    array![[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]]
}

/// Creates a 3x3 rotation matrix for rotation around the Z-axis.
pub fn rotation_matrix_z(angle_rad: f64) -> Array2<f64> {
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    array![[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]]
}

/// Combines three rotation matrices (Rx, Ry, Rz) by multiplication in the order Rz * Ry * Rx.
pub fn combine_rotations(rx: &Array2<f64>, ry: &Array2<f64>, rz: &Array2<f64>) -> Array2<f64> {
    rz.dot(ry).dot(rx)
}

/// Transforms a 3D point (x, y, z) by a 4x4 matrix.
/// Assumes the input point has a w-component of 1.0.
/// Returns the transformed point [x', y', z'].
pub fn transform_point_3d(point: [f64; 3], matrix: &Array2<f64>) -> [f64; 3] {
    let point_homogeneous = ndarray::arr1(&[point[0], point[1], point[2], 1.0]);
    let transformed_homogeneous = matrix.dot(&point_homogeneous);
    [
        transformed_homogeneous[0] / transformed_homogeneous[3],
        transformed_homogeneous[1] / transformed_homogeneous[3],
        transformed_homogeneous[2] / transformed_homogeneous[3],
    ]
}

/// Transforms a 3D vector (x, y, z) by a 4x4 matrix.
/// Assumes the input vector has a w-component of 0.0.
/// Returns the transformed vector [x', y', z'].
pub fn transform_vector_3d(vector: [f64; 3], matrix: &Array2<f64>) -> [f64; 3] {
    let vector_homogeneous = ndarray::arr1(&[vector[0], vector[1], vector[2], 0.0]);
    let transformed_homogeneous = matrix.dot(&vector_homogeneous);
    // No perspective division for vectors
    [
        transformed_homogeneous[0],
        transformed_homogeneous[1],
        transformed_homogeneous[2],
    ]
}

/// Subtracts vector `b` from `a` and returns the result as a new vector.
pub fn subtract_vec3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Creates a 4x4 view matrix from a 3x3 rotation matrix.
/// For orthographic projection, the camera is the projection plane itself.
/// This creates a view matrix that transforms from world space to view space,
/// where the view direction becomes the Z axis.
pub fn create_view_matrix(rotation_matrix: &Array2<f64>) -> Array2<f64> {
    // In orthographic projection, we don't position the camera at a distance
    // Instead, the projection plane itself acts as the camera
    // The rotation matrix should already be set up to transform the view direction to the Z axis
    let r = rotation_matrix;

    // Create the 4x4 view matrix without translation component
    // The rotation part ensures the view direction is transformed to the Z axis
    array![
        [r[[0, 0]], r[[0, 1]], r[[0, 2]], 0.0],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]], 0.0],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]], 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]
}
