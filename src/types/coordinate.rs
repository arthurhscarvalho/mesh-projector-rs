//! This module defines the `Coordinate` struct for representing 3D coordinates and vectors,
//! and provides methods for transformations and vector operations.

use crate::types::color::Color;
use ndarray::Array2;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

/// Represents a 3D coordinate or vector.
///
/// This struct is used to represent both points and vectors in 3D space.
/// The interpretation (point or vector) depends on the context and the
/// transformation method used.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coordinate(pub [f64; 3]);

impl Coordinate {
    /// Creates a new `Coordinate` from three f64 components.
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Coordinate([x, y, z])
    }

    /// Gets the X component
    pub fn x(&self) -> f64 {
        self.0[0]
    }

    /// Gets the Y component
    pub fn y(&self) -> f64 {
        self.0[1]
    }

    /// Gets the Z component
    pub fn z(&self) -> f64 {
        self.0[2]
    }

    /// Sets the X component
    pub fn set_x(&mut self, x: f64) {
        self.0[0] = x;
    }

    /// Sets the Y component
    pub fn set_y(&mut self, y: f64) {
        self.0[1] = y;
    }

    /// Sets the Z component
    pub fn set_z(&mut self, z: f64) {
        self.0[2] = z;
    }

    /// Transforms the coordinate as a point (w=1) using a 4x4 matrix.
    /// This is typically used for transforming positions.
    pub fn to_view_space_as_point(&self, matrix: &Array2<f64>) -> Coordinate {
        let point_homogeneous = ndarray::arr1(&[self.0[0], self.0[1], self.0[2], 1.0]);
        let transformed_homogeneous = matrix.dot(&point_homogeneous);
        Coordinate([
            transformed_homogeneous[0] / transformed_homogeneous[3],
            transformed_homogeneous[1] / transformed_homogeneous[3],
            transformed_homogeneous[2] / transformed_homogeneous[3],
        ])
    }

    /// Transforms the coordinate as a vector (w=0) using a 4x4 matrix.
    /// This is typically used for transforming directions or normals.
    pub fn to_view_space_as_vector(&self, matrix: &Array2<f64>) -> Coordinate {
        let vector_homogeneous = ndarray::arr1(&[self.0[0], self.0[1], self.0[2], 0.0]);
        let transformed_homogeneous = matrix.dot(&vector_homogeneous);
        Coordinate([
            transformed_homogeneous[0],
            transformed_homogeneous[1],
            transformed_homogeneous[2],
        ])
    }

    /// Normalizes the vector in-place.
    pub fn normalize(&mut self) {
        let length_squared = self.0[0] * self.0[0] + self.0[1] * self.0[1] + self.0[2] * self.0[2];
        if length_squared > 0.0 {
            let inv_length = 1.0 / length_squared.sqrt();
            self.0[0] *= inv_length;
            self.0[1] *= inv_length;
            self.0[2] *= inv_length;
        }
    }

    /// Returns a normalized version of the vector.
    pub fn normalized(&self) -> Coordinate {
        let mut new_coord = *self;
        new_coord.normalize();
        new_coord
    }

    /// Calculates the dot product with another `Coordinate`.
    pub fn dot(&self, other: &Coordinate) -> f64 {
        self.0.iter().zip(other.0.iter()).map(|(x, y)| x * y).sum()
    }

    /// Calculates the cross product with another `Coordinate`.
    pub fn cross(&self, other: &Coordinate) -> Coordinate {
        Coordinate([
            self.0[1] * other.0[2] - self.0[2] * other.0[1],
            self.0[2] * other.0[0] - self.0[0] * other.0[2],
            self.0[0] * other.0[1] - self.0[1] * other.0[0],
        ])
    }

    /// Returns the length (magnitude) of the vector.
    pub fn length(&self) -> f64 {
        (self.0[0] * self.0[0] + self.0[1] * self.0[1] + self.0[2] * self.0[2]).sqrt()
    }

    /// Returns the squared length of the vector (avoids sqrt computation).
    pub fn length_squared(&self) -> f64 {
        self.0[0] * self.0[0] + self.0[1] * self.0[1] + self.0[2] * self.0[2]
    }

    /// Maps each component of the coordinate using the provided function,
    /// returning a new `Coordinate` with the transformed values.
    pub fn map<F>(&self, f: F) -> Coordinate
    where
        F: Fn(f64) -> f64,
    {
        Coordinate([f(self.0[0]), f(self.0[1]), f(self.0[2])])
    }
}

impl From<[f64; 3]> for Coordinate {
    fn from(arr: [f64; 3]) -> Self {
        Coordinate(arr)
    }
}

impl From<Coordinate> for [f64; 3] {
    fn from(val: Coordinate) -> Self {
        val.0
    }
}

impl From<(f64, f64, f64)> for Coordinate {
    fn from(tuple: (f64, f64, f64)) -> Self {
        Coordinate([tuple.0, tuple.1, tuple.2])
    }
}

impl From<Coordinate> for (f64, f64, f64) {
    fn from(val: Coordinate) -> Self {
        (val.0[0], val.0[1], val.0[2])
    }
}

/// Converts a `Coordinate` to a `Color`.
/// This conversion is useful when mapping coordinates to images.
impl From<Coordinate> for Color {
    fn from(val: Coordinate) -> Self {
        Color([val.0[0], val.0[1], val.0[2]])
    }
}

impl Index<usize> for Coordinate {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Coordinate {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl Sub for Coordinate {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Coordinate([
            self.0[0] - other.0[0],
            self.0[1] - other.0[1],
            self.0[2] - other.0[2],
        ])
    }
}

impl Add for Coordinate {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Coordinate([
            self.0[0] + other.0[0],
            self.0[1] + other.0[1],
            self.0[2] + other.0[2],
        ])
    }
}

impl Neg for Coordinate {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Coordinate([-self.0[0], -self.0[1], -self.0[2]])
    }
}

impl Mul<f64> for Coordinate {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Coordinate([self.0[0] * rhs, self.0[1] * rhs, self.0[2] * rhs])
    }
}
