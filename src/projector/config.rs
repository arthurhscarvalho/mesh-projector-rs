// src/projector/config.rs
//! Projection settings and related structures.

use ndarray::Array2;

/// Contains settings for controlling mesh rotation during projection.
#[derive(Debug, Clone)]
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
#[derive(Debug)]
pub struct ProjectorConfig {
    /// The precision/resolution of the output images (log2 of dimensions).
    pub precision: u64,
    /// Radius for outlier filtering (0 = disabled).
    pub filtering_window_radius: u64,
    /// Optional rotation configuration. If None, uses default 6-view orthographic projection.
    pub rotation_config: Option<RotationConfig>,
}

impl ProjectorConfig {
    /// Creates a new projection settings object with default 6-view orthographic projection.
    pub fn new(precision: u64, filtering_window_radius: u64) -> Self {
        ProjectorConfig {
            precision,
            filtering_window_radius,
            rotation_config: None,
        }
    }

    /// Creates a new projection settings object with rotation-based projection.
    pub fn new_with_rotation(
        precision: u64,
        filtering_window_radius: u64,
        num_steps: u32,
        angle_delta_x: f64,
        angle_delta_y: f64,
        angle_delta_z: f64,
    ) -> Self {
        ProjectorConfig {
            precision,
            filtering_window_radius,
            rotation_config: Some(RotationConfig::new(
                angle_delta_x,
                angle_delta_y,
                angle_delta_z,
                num_steps as usize,
            )),
        }
    }
}

/// Encapsulates parameters for a single orthographic projection view (camera settings).
#[derive(Debug)]
pub struct ViewConfig {
    /// True if this is a front view (looking along the positive direction of the transformed Z-axis),
    /// false if it's a back view (looking along the negative direction).
    pub is_front_view: bool,
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
    pub lighting_config: LightingConfig,
}

/// Defines the source of the light.
#[derive(Debug, Clone, Copy)]
pub enum LightSource {
    /// Light is fixed in world coordinates.
    World([f64; 3]),
    /// Light originates from the camera.
    Camera,
}

/// Configuration for the lighting system.
#[derive(Debug, Clone, Copy)]
pub struct LightingConfig {
    /// The source of the light.
    pub light_source: LightSource,
    /// Ambient light color.
    pub ambient_color: [f64; 3],
    /// Diffuse light color.
    pub diffuse_color: [f64; 3],
    /// Specular light color.
    pub specular_color: [f64; 3],
    /// Shininess factor for specular highlights.
    pub shininess: f64,
}

impl Default for LightingConfig {
    fn default() -> Self {
        LightingConfig {
            light_source: LightSource::Camera, // Default to camera light
            ambient_color: [0.1, 0.1, 0.1],
            diffuse_color: [0.8, 0.8, 0.8],
            specular_color: [1.0, 1.0, 1.0],
            shininess: 32.0,
        }
    }
}
