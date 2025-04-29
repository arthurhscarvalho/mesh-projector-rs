// src/projector/projection.rs
//! Projection settings and related structures.

use crate::ProjectionMode;

/// Contains all the settings for controlling the projection process.
#[derive(Debug, Clone)]
pub struct ProjectionSettings {
    /// The precision/resolution of the output images (log2 of dimensions).
    pub precision: u32,
    /// Radius for outlier filtering (0 = disabled).
    pub filtering_window_radius: u32,
    /// The rendering mode (ObjOnly, ObjTexture, Full).
    pub mode: ProjectionMode,
}

impl ProjectionSettings {
    /// Creates a new projection settings object.
    pub fn new(precision: u32, filtering_window_radius: u32, mode: ProjectionMode) -> Self {
        ProjectionSettings {
            precision,
            filtering_window_radius,
            mode,
        }
    }
}
