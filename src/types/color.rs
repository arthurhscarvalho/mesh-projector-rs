//! This module defines the `Color` struct and related functionality for color representation
//! and manipulation, including conversions between sRGB and linear color spaces.

use std::ops::{Add, Mul};

/// The gamma factor for the sRGB color space.
const GAMMA: f64 = 2.2;
/// The inverse gamma factor for the sRGB color space.
const INVERSE_GAMMA: f64 = 1.0 / GAMMA;

/// Represents a color with three components (e.g., R, G, B).
///
/// The color values are stored as `f64` and are typically expected to be in a normalized range
/// (e.g., [0.0, 1.0]), although this is not strictly enforced by the struct itself.
/// It's also expected that the color is in linear color space.
/// Use `Color::from_srgb` to create a color from sRGB components.
#[derive(Clone, Copy)]
pub struct Color(pub [f64; 3]);

impl Color {
    /// Creates a new `Color` from three f64 components, expected to be in linear color space.
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Color([r, g, b])
    }

    /// Creates a new `Color` from three f64 components, in sRGB color space.
    /// The saved color is in linear color space.
    pub fn from_srgb(r: f64, g: f64, b: f64) -> Self {
        Color([r.powf(GAMMA), g.powf(GAMMA), b.powf(GAMMA)])
    }

    /// Creates a black color (0, 0, 0).
    pub fn black() -> Self {
        Color([0.0, 0.0, 0.0])
    }

    /// Creates a white color (1, 1, 1).
    pub fn white() -> Self {
        Color([1.0, 1.0, 1.0])
    }

    /// Creates a red color (1, 0, 0).
    pub fn red() -> Self {
        Color([1.0, 0.0, 0.0])
    }

    /// Creates a green color (0, 1, 0).
    pub fn green() -> Self {
        Color([0.0, 1.0, 0.0])
    }

    /// Creates a blue color (0, 0, 1).
    pub fn blue() -> Self {
        Color([0.0, 0.0, 1.0])
    }

    /// Gets the red component.
    pub fn r(&self) -> f64 {
        self.0[0]
    }

    /// Gets the green component.
    pub fn g(&self) -> f64 {
        self.0[1]
    }

    /// Gets the blue component.
    pub fn b(&self) -> f64 {
        self.0[2]
    }

    /// Sets the red component.
    pub fn set_r(&mut self, r: f64) {
        self.0[0] = r;
    }

    /// Sets the green component.
    pub fn set_g(&mut self, g: f64) {
        self.0[1] = g;
    }

    /// Sets the blue component.
    pub fn set_b(&mut self, b: f64) {
        self.0[2] = b;
    }

    /// Converts the color from sRGB to linear color space.
    pub fn to_linear(&self) -> Color {
        Color([
            self.r().powf(GAMMA),
            self.g().powf(GAMMA),
            self.b().powf(GAMMA),
        ])
    }

    /// Converts the color from linear to sRGB color space.
    pub fn to_srgb(&self) -> Color {
        Color([
            self.r().powf(INVERSE_GAMMA),
            self.g().powf(INVERSE_GAMMA),
            self.b().powf(INVERSE_GAMMA),
        ])
    }

    /// Converts the color from linear to sRGB color space and denormalizes it to the range [0, 255].
    pub fn to_srgb_denormalized(&self) -> Color {
        Color([
            self.r()
                .powf(INVERSE_GAMMA)
                .mul(255.0)
                .round()
                .clamp(0.0, 255.0),
            self.g()
                .powf(INVERSE_GAMMA)
                .mul(255.0)
                .round()
                .clamp(0.0, 255.0),
            self.b()
                .powf(INVERSE_GAMMA)
                .mul(255.0)
                .round()
                .clamp(0.0, 255.0),
        ])
    }

    /// Clamps the color components to the given range.
    pub fn clamp(&self, min: f64, max: f64) -> Color {
        Color([
            self.r().clamp(min, max),
            self.g().clamp(min, max),
            self.b().clamp(min, max),
        ])
    }
}

impl From<[f64; 3]> for Color {
    fn from(arr: [f64; 3]) -> Self {
        Color(arr)
    }
}

impl From<Color> for [f64; 3] {
    fn from(val: Color) -> Self {
        val.0
    }
}

impl From<(f64, f64, f64)> for Color {
    fn from(tuple: (f64, f64, f64)) -> Self {
        Color([tuple.0, tuple.1, tuple.2])
    }
}

impl From<Color> for (f64, f64, f64) {
    fn from(val: Color) -> Self {
        (val.0[0], val.0[1], val.0[2])
    }
}

impl Add for Color {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Color([
            self.0[0] + other.0[0],
            self.0[1] + other.0[1],
            self.0[2] + other.0[2],
        ])
    }
}

impl Mul<f64> for Color {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Color([self.0[0] * rhs, self.0[1] * rhs, self.0[2] * rhs])
    }
}

impl Mul<Color> for Color {
    type Output = Self;

    /// Performs component-wise multiplication of two colors.
    fn mul(self, rhs: Color) -> Self::Output {
        Color([
            self.0[0] * rhs.0[0],
            self.0[1] * rhs.0[1],
            self.0[2] * rhs.0[2],
        ])
    }
}
