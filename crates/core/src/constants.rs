pub mod projector;
pub mod renderer;

pub use projector::*;
pub use renderer::*;

/// Minimum radius used for degenerate bounding spheres to avoid division by zero during scaling.
pub const MIN_BOUNDING_SPHERE_RADIUS: f32 = 1e-9;
/// Tolerance used when checking transformed vertices against normalized cube bounds.
pub const BOUNDS_EPSILON: f32 = 1e-4;
