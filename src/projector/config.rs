// src/projector/config.rs
//! Projection settings and related structures.

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
    /// The axis representing depth relative to the mesh (0=X, 1=Y, 2=Z).
    pub depth_axis: usize,
    /// True if this is a front view (looking along the positive direction of `depth_axis`),
    /// false if it's a back view (looking along the negative direction).
    pub is_front_view: bool,
    /// Normalized view direction vector corresponding to this projection.
    pub view_dir: (f64, f64, f64),
    /// The primary axis index for the 2D projection plane (e.g., Y if depth is X).
    /// This typically corresponds to the image's column index.
    pub p_axis: usize,
    /// The secondary axis index for the 2D projection plane (e.g., Z if depth is X).
    /// This typically corresponds to the image's row index.
    pub q_axis: usize,
    /// The maximum bound of the projection space, used for scaling depth values.
    pub max_bound_f64: f64,
}
