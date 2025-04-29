// src/projector/material.rs
//! Material properties and related structures.

/// Material properties matching typical MTL definitions.
#[derive(Debug, Clone)]
pub struct MaterialProperties {
    pub ka: [f32; 3], // Ambient color
    pub kd: [f32; 3], // Diffuse color
    pub ks: [f32; 3], // Specular color
    pub ns: f32,      // Specular exponent
    pub opacity: f32, // Opacity (0.0=transparent, 1.0=opaque) - Currently unused in shader
    pub illum: i32,   // Illumination model
}
