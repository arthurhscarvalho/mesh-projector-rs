// src/projector/constants.rs
use std::f32;

// --- Constants ---
/// Number of projection images (3 axes, 2 directions each: +X, -X, +Y, -Y, +Z, -Z).
pub const NUM_IMAGES: usize = 6;
/// Default background color/brightness for output images.
pub const BACKGROUND_BRIGHTNESS: u8 = 255;
/// Epsilon value for checking division by zero in barycentric coordinate calculation.
pub const BARYCENTRIC_EPSILON: f32 = 1e-7;
/// Epsilon tolerance for checking if calculated barycentric coordinates (u,v,w) are within the valid range [0, 1].
pub const BARYCENTRIC_COORD_EPSILON: f32 = -1e-7;
/// Factor applied to the maximum dimension (`max_bound`) to determine the depth threshold for outlier filtering.
pub const FILTERING_NEIGHBORHOOD_FACTOR: f32 = 0.1;
