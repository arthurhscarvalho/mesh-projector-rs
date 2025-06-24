//! This module defines the `Color` struct and related functionality for color representation
//! and manipulation, including conversions between sRGB and linear color spaces.

use once_cell::sync::Lazy;
use std::ops::{Add, Mul};

/// The gamma factor for the sRGB color space.
const GAMMA: f32 = 2.2;
/// The inverse gamma factor for the sRGB color space.
const INVERSE_GAMMA: f32 = 1.0 / GAMMA;

// Compute sRGB LUT at first access, safely and once
static SRGB_LUT: Lazy<[u8; 256]> = Lazy::new(|| {
    let mut lut = [0; 256];
    for i in 0..=255 {
        let x = i as f32 / 255.0;
        let srgb = x.powf(INVERSE_GAMMA) * 255.0;
        lut[i] = srgb.round().clamp(0.0, 255.0) as u8;
    }
    lut
});

static LINEAR_LUT: Lazy<[f32; 256]> = Lazy::new(|| {
    let mut lut = [0.0; 256];
    for i in 0..=255 {
        lut[i] = (i as f32 / 255.0).powf(GAMMA);
    }
    lut
});

/// Fast conversion from linear [0,1] to 8-bit sRGB using the LUT.
#[inline]
fn linear_to_srgb_byte(linear: f32) -> u8 {
    let idx = (linear * 255.0).round().clamp(0.0, 255.0) as usize;
    SRGB_LUT[idx]
}

/// Fast conversion from 8-bit sRGB to linear [0,1].
#[inline]
pub fn srgb_byte_to_linear(b: u8) -> f32 {
    LINEAR_LUT[b as usize]
}

/// Represents a color with three components (e.g., R, G, B).
///
/// The color values are stored as `f32` and are typically expected to be in a normalized range
/// (e.g., [0.0, 1.0]), although this is not strictly enforced by the struct itself.
/// It's also expected that the color is in linear color space.
/// Use `Color::from_srgb` to create a color from sRGB components.
#[derive(Clone, Copy)]
pub struct Color(pub [f32; 3]);

impl Color {
    /// Creates a new `Color` from three f32 components, expected to be in linear color space.
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Color([r, g, b])
    }

    /// Creates a new `Color` from three f32 components, in sRGB color space.
    /// The saved color is in linear color space.
    pub fn from_srgb(r: f32, g: f32, b: f32) -> Self {
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
    pub fn r(&self) -> f32 {
        self.0[0]
    }

    /// Gets the green component.
    pub fn g(&self) -> f32 {
        self.0[1]
    }

    /// Gets the blue component.
    pub fn b(&self) -> f32 {
        self.0[2]
    }

    /// Sets the red component.
    pub fn set_r(&mut self, r: f32) {
        self.0[0] = r;
    }

    /// Sets the green component.
    pub fn set_g(&mut self, g: f32) {
        self.0[1] = g;
    }

    /// Sets the blue component.
    pub fn set_b(&mut self, b: f32) {
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

    /// Converts the color from linear to 8-bit sRGB quickly using a pre-computed LUT.
    pub fn to_srgb8(&self) -> [u8; 3] {
        [
            linear_to_srgb_byte(self.r()),
            linear_to_srgb_byte(self.g()),
            linear_to_srgb_byte(self.b()),
        ]
    }

    /// Clamps the color components to the given range.
    pub fn clamp(&self, min: f32, max: f32) -> Color {
        Color([
            self.r().clamp(min, max),
            self.g().clamp(min, max),
            self.b().clamp(min, max),
        ])
    }

    /// Legacy helper: Converts the color from linear to sRGB space and returns a Color whose
    /// components are in the 0–255 *floating-point* range (to match previous API). Internally
    /// this uses the same fast LUT, so it is inexpensive.
    pub fn to_srgb_denormalized(&self) -> Color {
        let [r, g, b] = self.to_srgb8();
        Color([r as f32, g as f32, b as f32])
    }
}

impl From<[f32; 3]> for Color {
    fn from(arr: [f32; 3]) -> Self {
        Color(arr)
    }
}

impl From<Color> for [f32; 3] {
    fn from(val: Color) -> Self {
        val.0
    }
}

impl From<(f32, f32, f32)> for Color {
    fn from(tuple: (f32, f32, f32)) -> Self {
        Color([tuple.0, tuple.1, tuple.2])
    }
}

impl From<Color> for (f32, f32, f32) {
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

impl Mul<f32> for Color {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
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
