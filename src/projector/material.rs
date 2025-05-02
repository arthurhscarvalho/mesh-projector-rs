// src/projector/material.rs
//! Material properties and related structures.

/// Material properties matching typical MTL definitions.
#[derive(Debug, Clone)]
pub struct MaterialProperties {
    pub ka: [f64; 3], // Ambient color
    pub kd: [f64; 3], // Diffuse color
    pub ks: [f64; 3], // Specular color
    pub ns: f64,      // Specular exponent
    pub opacity: f64, // Opacity (0.0=transparent, 1.0=opaque) - Currently unused in shader
    pub illum: i64,   // Illumination model
}
