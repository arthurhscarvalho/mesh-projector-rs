//! Core linear algebra and vector math utilities for 3D graphics.
//!
//! This module provides essential functions for manipulating 3D vectors and matrices,
//! including dot products, normalization, rotation matrix construction, and geometric
//! transformations.

use crate::types::coordinate::Coordinate;
use ndarray::{array, Array2};

/// Transforms a 3D point using a 4x4 matrix.
/// Returns the transformed point as a Coordinate.
pub fn transform_point_3d(point: Coordinate, matrix: &Array2<f64>) -> Coordinate {
    point.to_view_space_as_point(matrix)
}

/// Transforms a 3D vector using a 4x4 matrix.
/// Returns the transformed vector as a Coordinate.
pub fn transform_vector_3d(vector: Coordinate, matrix: &Array2<f64>) -> Coordinate {
    vector.to_view_space_as_vector(matrix)
}

/// Subtracts vector `b` from `a` and returns the result as a new vector.
pub fn subtract_vec3(a: Coordinate, b: Coordinate) -> Coordinate {
    a - b
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
