// src/projector/texture.rs
use nd::ArrayView3;

/// Represents texture properties and provides easy access to dimensions.
#[derive(Debug, Clone)]
pub struct TextureData<'a> {
    /// Read-only view into the texture pixel data (H, W, C).
    pub texture: ArrayView3<'a, u8>,
    /// Number of color channels in the texture (e.g., 3 for RGB, 4 for RGBA).
    pub channels: usize,
    /// Height of the texture as a floating-point number.
    pub height_f32: f32,
    /// Width of the texture as a floating-point number.
    pub width_f32: f32,
}

impl<'a> TextureData<'a> {
    /// Creates a new `TextureData` instance from a texture view.
    /// Calculates and stores dimensions for quick access.
    pub fn new(texture: ArrayView3<'a, u8>) -> Self {
        let shape = texture.shape();
        let height = shape[0];
        let width = shape[1];
        TextureData {
            texture,
            channels: shape[2],
            height_f32: height as f32,
            width_f32: width as f32,
        }
    }
}
