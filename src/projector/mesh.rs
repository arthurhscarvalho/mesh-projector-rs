// src/projector/mesh.rs
//! Mesh data structures for representing 3D geometry.

use crate::projector::projection::ProjectionMode;
use ndarray::{Array2, Array3};

#[derive(Debug)]
pub struct MeshGeometry {
    pub vertices: Array2<f64>,
    pub triangles: Array2<usize>,
    pub vertex_normals: Option<Array2<f64>>,
}

#[derive(Debug)]
pub struct MeshTexture {
    pub texture: Option<Array3<u8>>,
    pub vertex_colors: Option<Array2<f64>>,
    pub triangle_uvs: Option<Array2<f64>>,
    /// Number of color channels in the texture (e.g., 3 for RGB, 4 for RGBA).
    pub channels: usize,
    /// Height of the texture as a floating-point number.
    pub height_f64: f64,
    /// Width of the texture as a floating-point number.
    pub width_f64: f64,
}

#[derive(Debug)]
pub struct MeshMaterial {
    pub ka: [f64; 3], // Ambient color
    pub kd: [f64; 3], // Diffuse color
    pub ks: [f64; 3], // Specular color
    pub ns: f64,      // Specular exponent
    pub opacity: f64, // Opacity (0.0=transparent, 1.0=opaque)
    pub illum: i64,   // Illumination model
}

#[derive(Debug)]
pub struct Mesh {
    // Geometry
    pub geometry: MeshGeometry,
    // Texture
    pub texture: MeshTexture,
    // Material
    pub material: MeshMaterial,
    // Rendering mode
    pub mode: ProjectionMode,
}
