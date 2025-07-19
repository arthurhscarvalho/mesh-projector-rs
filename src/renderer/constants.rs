//! This module defines constants for image projection, rendering, and filtering operations.
//! These constants control color representation, numerical tolerances, lighting parameters,
//! and default values used throughout the projector pipeline.

use crate::types::color::Color;

/// Default background color/brightness for output images.
pub const BACKGROUND_BRIGHTNESS: u8 = 255;
/// Epsilon value for checking division by zero in barycentric coordinate calculation.
pub const BARYCENTRIC_EPSILON: f32 = 1e-8;
/// Epsilon tolerance for checking if calculated barycentric coordinates (u,v,w) are within the valid range [0, 1].
pub const BARYCENTRIC_COORD_EPSILON: f32 = -1e-8;
/// Threshold for considering a triangle as degenerate (zero or near-zero area).
pub const TRIANGLE_AREA_DEGENERACY_THRESHOLD: f32 = 1e-8;
/// Default color for missing textures (magenta for visibility)
pub const DEFAULT_MISSING_TEXTURE_COLOR: Color = Color([1.0, 0.0, 1.0]);
/// Factor applied to the maximum dimension (`max_bound`) to determine the depth threshold for outlier filtering.
pub const FILTERING_NEIGHBORHOOD_FACTOR: f32 = 0.1;
