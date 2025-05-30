// src/projector/config.rs
//! Projection settings and related structures.

use ndarray::Array2;

/// Contains settings for controlling mesh rotation during projection.
pub struct RotationConfig {
    /// Angle delta for rotation around X axis (in radians)
    pub angle_delta_x: f64,
    /// Angle delta for rotation around Y axis (in radians)
    pub angle_delta_y: f64,
    /// Angle delta for rotation around Z axis (in radians)
    pub angle_delta_z: f64,
    /// Number of steps for each rotation axis
    pub num_steps: usize,
}

impl RotationConfig {
    /// Creates a new rotation configuration with specified angle deltas and number of steps.
    pub fn new(
        angle_delta_x: f64,
        angle_delta_y: f64,
        angle_delta_z: f64,
        num_steps: usize,
    ) -> Self {
        RotationConfig {
            angle_delta_x,
            angle_delta_y,
            angle_delta_z,
            num_steps,
        }
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
    /// A value of 1.0 aims for a total point count of roughly 0.5 * (2^precision)^2.
    pub point_cloud_density_factor: f64,
}

impl ProjectorConfig {
    /// Creates a new projection settings object with rotation-based projection.
    pub fn new(
        precision: u64,
        filtering_window_radius: u64,
        num_steps: u32,
        angle_delta_x: f64,
        angle_delta_y: f64,
        angle_delta_z: f64,
        point_cloud_density_factor: f64,
    ) -> Self {
        ProjectorConfig {
            precision,
            filtering_window_radius,
            rotation_config: RotationConfig::new(
                angle_delta_x,
                angle_delta_y,
                angle_delta_z,
                num_steps as usize,
            ),
            point_cloud_density_factor,
        }
    }
}

/// Encapsulates parameters for a single orthographic projection view (camera settings).
pub struct ViewConfig {
    /// Normalized view direction vector corresponding to this projection
    /// (this will be the new Z-axis after transformation).
    pub view_dir: (f64, f64, f64),
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
    World([f64; 3]),
    /// Light originates from the camera.
    Camera,
}
