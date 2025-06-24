use crate::types::coordinate::Coordinate;
use nd::Array2;

#[allow(dead_code)] // Temporary while we don't parametrize the light source
/// Defines the source of the light.
pub enum LightSource {
    /// Directional light, the light is fixed in world coordinates.
    Directional(Coordinate),
    /// Ambient light, the mesh is always lit by ambient light.
    Ambient,
    /// Headlight camera, the light originates from the camera.
    Camera,
}

/// Encapsulates parameters for a single orthographic projection view (camera settings).
pub struct ViewConfig {
    /// Normalized view direction vector corresponding to this projection
    pub view_direction: Coordinate,
    /// View matrix that transforms from world space to view space.
    pub transformation_matrix: Array2<f32>,
    /// The maximum bound of the projection space, used for scaling depth values.
    pub max_bound_f32: f32,
    /// Lighting configuration
    pub light_source: LightSource,
}
