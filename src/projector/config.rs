//! Configuration types and settings for controlling the 3D mesh projection process.
//!
//! This module defines the structures and parameters used to specify projection resolution,
//! rotation strategies, outlier filtering, and other options that influence how projections
//! are generated from 3D meshes. These settings are consumed by the projector pipeline to
//! produce images, depth maps, and related outputs.

use crate::types::orientation::Orientation;

/// Contains all the settings for controlling the projection process.
pub struct ProjectionConfig {
    /// The precision/resolution of the output images (log2 of dimensions).
    pub precision: u64,
    /// Radius for outlier filtering (0 = disabled).
    pub filtering_window_radius: u64,
    /// Angles for rotation (pitch, yaw, roll in radians)
    pub view_angles: Vec<Orientation>,
}

/// Defines the parameters for the point cloud generation.
pub struct PointCloudConfig {
    /// The precision/resolution of the output point cloud.
    pub precision: u64,
    /// Factor to control the density of points in the discretized point cloud.
    pub point_cloud_density_factor: f32,
}
