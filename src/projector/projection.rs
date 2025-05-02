// src/projector/projection.rs
//! Projection settings and related structures.

use crate::ProjectionMode;

/// Contains all the settings for controlling the projection process.
#[derive(Debug, Clone)]
pub struct ProjectionSettings {
    /// The precision/resolution of the output images (log2 of dimensions).
    pub precision: u64,
    /// Radius for outlier filtering (0 = disabled).
    pub filtering_window_radius: u64,
    /// The rendering mode (ObjOnly, ObjTexture, Full).
    pub mode: ProjectionMode,
}

impl ProjectionSettings {
    /// Creates a new projection settings object.
    pub fn new(precision: u64, filtering_window_radius: u64, mode: ProjectionMode) -> Self {
        ProjectionSettings {
            precision,
            filtering_window_radius,
            mode,
        }
    }
}
