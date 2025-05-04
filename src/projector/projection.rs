// src/projector/projection.rs
//! Projection settings and related structures.

/// Contains all the settings for controlling the projection process.
#[derive(Debug)]
pub struct ProjectionSettings {
    /// The precision/resolution of the output images (log2 of dimensions).
    pub precision: u64,
    /// Radius for outlier filtering (0 = disabled).
    pub filtering_window_radius: u64,
}

impl ProjectionSettings {
    /// Creates a new projection settings object.
    pub fn new(precision: u64, filtering_window_radius: u64) -> Self {
        ProjectionSettings {
            precision,
            filtering_window_radius,
        }
    }
}
