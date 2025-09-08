use crate::types::color::Color;
use crate::types::coordinate::Coordinate;
use ndarray::Array2;

/// Defines the source of the light.
pub enum LightSource {
    /// Ambient light, constant across the mesh.
    Ambient,
    /// Directional light fixed in view coordinates.
    Directional(Coordinate),
    /// Headlight attached to the camera.
    Headlight,
    /// Point light (omnidirectional) fixed in world coordinates.
    Point(Coordinate),
}

pub struct LightComponents {
    pub ambient: Color,
    pub diffuse: Color,
    pub specular: Color,
}

pub struct LightingConfig {
    pub source: LightSource,
    pub components: LightComponents,
}

pub struct ViewConfig {
    pub view_direction: Coordinate,
    pub transformation_matrix: Array2<f32>,
    pub max_bound_f32: f32,
}

pub struct RenderPassConfig<'a> {
    pub view: ViewConfig,
    pub lighting: &'a LightingConfig,
}
