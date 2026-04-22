use crate::types::color::Color;
use crate::types::coordinate::Coordinate;
use ndarray::Array2;

/// Supported projection models for the renderer.
pub enum ProjectionModel {
    /// Orthographic projection (default).
    Orthographic,
    /// Perspective projection with explicit camera parameters.
    Perspective(PerspectiveParameters),
}

/// Derived perspective parameters for a render pass.
pub struct PerspectiveParameters {
    /// Camera position in world space.
    pub camera_position_ws: Coordinate,
    /// Focal length in pixel units (derived from FOV and resolution).
    pub focal_length_px: f32,
    /// Near plane distance in camera space.
    pub z_near: f32,
    /// Far plane distance in camera space.
    pub z_far: f32,
}

/// Defines the source of the light.
pub enum LightSource {
    /// Ambient light, constant across the mesh.
    Ambient,
    /// Directional light fixed in view coordinates.
    Directional(Coordinate),
    /// Headlight attached to the camera.
    Headlight,
    /// Point lights (omnidirectional) fixed in world coordinates.
    /// Supports multiple point light positions.
    Point(Vec<Coordinate>),
}

pub struct LightComponents {
    pub global_ambient: Color,
    pub global_diffuse: Color,
    pub global_specular: Color,
}

pub struct LightingConfig {
    pub source: LightSource,
    pub components: LightComponents,
}

/// Screen-space ambient occlusion configuration.
#[derive(Clone, Debug)]
pub struct SsaoConfig {
    /// Whether SSAO is enabled for this render pass.
    pub enabled: bool,
    /// Radius of the sampling hemisphere in view-space units.
    pub sample_radius: f32,
    /// Number of samples to take around each pixel.
    pub num_samples: usize,
    /// Bias to prevent self-occlusion artifacts.
    pub bias: f32,
    /// Strength in `[0, 1]` applied when darkening the final color.
    pub intensity: f32,
    /// Radius of the optional blur applied after computing raw occlusion.
    pub blur_radius: usize,
    /// Maximum distance for occlusion contribution (in view-space units).
    pub max_distance: f32,
}

impl Default for SsaoConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sample_radius: 0.75,
            num_samples: 32,
            bias: 0.025,
            intensity: 0.5,
            blur_radius: 4,
            max_distance: 2.0,
        }
    }
}

impl SsaoConfig {
    pub fn disabled() -> Self {
        let mut defaults = Self::default();
        defaults.enabled = false;
        defaults
    }
}

pub struct ViewConfig {
    pub view_direction: Coordinate,
    pub transformation_matrix: Array2<f32>,
    pub max_bound_f32: f32,
    pub projection: ProjectionModel,
}

pub struct RenderPassConfig<'a> {
    pub view: ViewConfig,
    pub lighting: &'a LightingConfig,
    pub ssao: &'a SsaoConfig,
}
