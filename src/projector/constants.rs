// src/projector/constants.rs
use std::f64;

/// Number of channels in the output image (RGB).
pub const NUM_CHANNELS: usize = 3;
/// Default background color/brightness for output images.
pub const BACKGROUND_BRIGHTNESS: u8 = 255;
/// Epsilon value for checking division by zero in barycentric coordinate calculation.
pub const BARYCENTRIC_EPSILON: f64 = f64::EPSILON;
/// Epsilon tolerance for checking if calculated barycentric coordinates (u,v,w) are within the valid range [0, 1].
pub const BARYCENTRIC_COORD_EPSILON: f64 = -f64::EPSILON;
/// Factor applied to the maximum dimension (`max_bound`) to determine the depth threshold for outlier filtering.
pub const FILTERING_NEIGHBORHOOD_FACTOR: f64 = 0.1;
/// Global ambient lighting factor used in illumination model 2+
pub const GLOBAL_AMBIENT: [f64; 3] = [0.2, 0.2, 0.2];
/// Default color for missing textures (magenta for visibility)
pub const DEFAULT_MISSING_TEXTURE_COLOR: [f64; 3] = [1.0, 0.0, 1.0];
