//! Configuration types and settings for controlling the 3D mesh projection process.
//!
//! This module defines the structures and parameters used to specify projection resolution,
//! rotation strategies, outlier filtering, and other options that influence how projections
//! are generated from 3D meshes. These settings are consumed by the projector pipeline to
//! produce images, depth maps, and related outputs.

use crate::types::{coordinate::Coordinate, orientation::Orientation};
use ndarray::Array2;

/// Contains settings for controlling mesh rotation during projection.
pub struct RotationConfig {
    /// Angle deltas for rotation (pitch, yaw, roll in radians)
    pub angle_deltas: Orientation,
    /// Number of steps for each rotation axis
    pub num_steps: usize,
}

impl RotationConfig {
    /// Creates a new rotation configuration with specified angle deltas and number of steps.
    pub fn new(angle_deltas: Orientation, num_steps: usize) -> Self {
        RotationConfig {
            angle_deltas,
            num_steps,
        }
    }

    /// Gets the X-axis angle delta (pitch).
    pub fn angle_delta_x(&self) -> f64 {
        self.angle_deltas.pitch()
    }

    /// Gets the Y-axis angle delta (yaw).
    pub fn angle_delta_y(&self) -> f64 {
        self.angle_deltas.yaw()
    }

    /// Gets the Z-axis angle delta (roll).
    pub fn angle_delta_z(&self) -> f64 {
        self.angle_deltas.roll()
    }
}

/// Contains all the settings for controlling the projection process.
pub struct ProjectorConfig {
    /// The precision/resolution of the output images (log2 of dimensions).
    pub precision: u64,
    /// Radius for outlier filtering (0 = disabled).
    pub filtering_window_radius: u64,
    /// Optional rotation configuration. If None, uses default 6-view orthographic projection.
    pub rotation_config: RotationConfig,
    /// Factor to control the density of points in the discretized point cloud.
    pub point_cloud_density_factor: Option<f64>,
}

impl ProjectorConfig {
    /// Creates a new projection settings object with rotation-based projection.
    pub fn new(
        precision: u64,
        filtering_window_radius: u64,
        num_steps: u32,
        rotation_config: Orientation,
        point_cloud_density_factor: Option<f64>,
    ) -> Self {
        ProjectorConfig {
            precision,
            filtering_window_radius,
            rotation_config: RotationConfig::new(rotation_config, num_steps as usize),
            point_cloud_density_factor,
        }
    }
}

/// Encapsulates parameters for a single orthographic projection view (camera settings).
pub struct ViewConfig {
    /// Normalized view direction vector corresponding to this projection
    /// (this will be the new Z-axis after transformation).
    pub view_dir: Coordinate,
    /// Rotation matrix to orient the mesh for this view.
    pub rotation_matrix: Array2<f64>,
    /// View matrix that transforms from world space to view space.
    /// For orthographic projection, this transforms world coordinates so the
    /// view direction becomes the Z axis, with the projection plane at the origin.
    pub view_matrix: Array2<f64>,
    /// The maximum bound of the projection space, used for scaling depth values.
    pub max_bound_f64: f64,
    /// Lighting configuration
    pub light_source: LightSource,
}

/// Defines the source of the light.
pub enum LightSource {
    /// Light is fixed in world coordinates.
    #[allow(dead_code)] // Temporary while we don't parametrize the light source
    World(Coordinate),
    /// Light originates from the camera.
    Camera,
}
