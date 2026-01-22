//! Configuration types and settings for controlling the 3D mesh projection process.
//!
//! This module defines the structures and parameters used to specify projection resolution,
//! rotation strategies, outlier filtering, and other options that influence how projections
//! are generated from 3D meshes. These settings are consumed by the projector pipeline to
//! produce images, depth maps, and related outputs.

use crate::renderer::config::SsaoConfig;
use crate::types::color::Color;
use crate::types::orientation::Orientation;

/// Defines which projection model the camera should use.
pub enum ProjectionType {
    /// Orthographic camera (default).
    Orthographic,
    /// Perspective camera with configurable parameters.
    Perspective(PerspectiveCameraConfig),
}

/// Perspective camera parameters.
pub struct PerspectiveCameraConfig {
    /// Vertical field of view in degrees.
    pub vertical_fov_degrees: f32,
}

/// Camera configuration for projections.
pub struct CameraConfig {
    /// Projection model and its parameters.
    pub projection: ProjectionType,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            projection: ProjectionType::Orthographic,
        }
    }
}

/// Contains all the settings for controlling the projection process.
pub struct ProjectionConfig {
    /// The resolution of the output images.
    pub resolution: u16,
    /// Radius for outlier filtering (0 = disabled).
    pub filtering_window_radius: u64,
    /// Angles for rotation (pitch, yaw, roll in radians)
    pub view_angles: Vec<Orientation>,
    /// Camera configuration used for the projection.
    pub camera: CameraConfig,
    /// Multi-sample anti-aliasing configuration.
    pub msaa: MsaaConfig,
    /// Screen-space ambient occlusion configuration.
    pub ssao: SsaoConfig,
    /// Background color (linear RGB) and alpha for the output images.
    pub background: BackgroundConfig,
}

/// Defines the parameters for the point cloud generation.
pub struct PointCloudConfig {
    /// The resolution of the output point cloud.
    pub resolution: u16,
    /// Factor to control the density of points in the discretized point cloud.
    pub point_cloud_density_factor: f32,
}

/// Supported multi-sample anti-aliasing sample counts.
pub enum MsaaSampleCount {
    X1,
    X2,
    X4,
    X8,
}

impl MsaaSampleCount {
    /// Returns the numeric sample count represented by this enum.
    pub fn samples(&self) -> u8 {
        match self {
            MsaaSampleCount::X1 => 1,
            MsaaSampleCount::X2 => 2,
            MsaaSampleCount::X4 => 4,
            MsaaSampleCount::X8 => 8,
        }
    }
}

/// Multi-sample anti-aliasing configuration.
pub struct MsaaConfig {
    /// Desired sample count (1 = disabled).
    pub sample_count: MsaaSampleCount,
}

impl Default for MsaaConfig {
    fn default() -> Self {
        Self {
            sample_count: MsaaSampleCount::X1,
        }
    }
}

/// Background color configuration for output images.
/// Determines both the background color and the number of output channels.
pub enum BackgroundConfig {
    /// RGB output (3 channels, no alpha). Material transparency is premultiplied.
    Rgb(Color),
    /// RGBA output (4 channels with alpha).
    Rgba(Color, u8),
}

impl BackgroundConfig {
    /// Returns the number of image channels (3 for RGB, 4 for RGBA).
    pub fn channels(&self) -> usize {
        match self {
            BackgroundConfig::Rgb(_) => 3,
            BackgroundConfig::Rgba(_, _) => 4,
        }
    }

    /// Returns the background color as a byte array (either [R,G,B] or [R,G,B,A]).
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            BackgroundConfig::Rgb(color) => {
                let srgb = color.to_srgb8();
                vec![srgb[0], srgb[1], srgb[2]]
            }
            BackgroundConfig::Rgba(color, alpha) => {
                let srgb = color.to_srgb8();
                vec![srgb[0], srgb[1], srgb[2], *alpha]
            }
        }
    }

    /// Returns the background color encoded as sRGBA8 for buffer initialization.
    /// For RGB mode, alpha is set to 255 (opaque).
    pub fn to_srgba8(&self) -> [u8; 4] {
        match self {
            BackgroundConfig::Rgb(color) => {
                let srgb = color.to_srgb8();
                [srgb[0], srgb[1], srgb[2], 255]
            }
            BackgroundConfig::Rgba(color, alpha) => {
                let srgb = color.to_srgb8();
                [srgb[0], srgb[1], srgb[2], *alpha]
            }
        }
    }

    /// Returns the background color.
    pub fn color(&self) -> &Color {
        match self {
            BackgroundConfig::Rgb(color) => color,
            BackgroundConfig::Rgba(color, _) => color,
        }
    }
}

impl Default for BackgroundConfig {
    fn default() -> Self {
        BackgroundConfig::Rgb(Color::white())
    }
}
