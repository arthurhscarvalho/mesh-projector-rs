// src/projector/config.rs
//! Projection settings and related structures.
use crate::projector::constants::NUM_IMAGES;

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

impl ViewConfig {
    /// Creates projection settings based on the image layer index (0 to NUM_IMAGES - 1).
    /// Layers 0, 2, 4 are front views (+X, +Y, +Z).
    /// Layers 1, 3, 5 are back views (-X, -Y, -Z).
    pub fn new(img_layer: usize, max_bound_f64: f64) -> Self {
        assert!(
            img_layer < NUM_IMAGES,
            "Invalid image layer index: {}",
            img_layer
        );
        let depth_axis = img_layer / 2; // 0 for X, 1 for Y, 2 for Z
        let view_type = img_layer % 2; // 0 for front, 1 for back
        let is_front_view = view_type == 0;
        // Determine view direction based on axis and front/back type
        let view_dir = match (depth_axis, is_front_view) {
            (0, true) => (1.0, 0.0, 0.0),   // +X
            (0, false) => (-1.0, 0.0, 0.0), // -X
            (1, true) => (0.0, 1.0, 0.0),   // +Y
            (1, false) => (0.0, -1.0, 0.0), // -Y
            (2, true) => (0.0, 0.0, 1.0),   // +Z
            (2, false) => (0.0, 0.0, -1.0), // -Z
            _ => unreachable!(),            // Should be covered by the assert above
        };
        // Define the axes used for the 2D projection plane (P, Q) based on the depth axis
        // If depth is X (0), plane is YZ (1, 2) -> p=Y, q=Z
        // If depth is Y (1), plane is XZ (0, 2) -> p=X, q=Z
        // If depth is Z (2), plane is XY (0, 1) -> p=X, q=Y
        // Note: The order (p, q) maps to (cols, rows) typically.
        const PROJECTION_PLANE_AXES: [(usize, usize); 3] = [(1, 2), (0, 2), (0, 1)];
        let (p_axis, q_axis) = PROJECTION_PLANE_AXES[depth_axis];
        ViewConfig {
            depth_axis,
            is_front_view,
            view_dir,
            p_axis,
            q_axis,
            max_bound_f64,
        }
    }
}
