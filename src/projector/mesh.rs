// src/projector/mesh.rs
//! Mesh data structures for representing 3D geometry.
use ndarray::{Array2, Array3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshMode {
    ObjOnly,
    ObjTexture,
    Full,
}

#[derive(Debug)]
pub struct MeshGeometry {
    /// Vertex positions (N, 3).
    pub vertices: Array2<f64>,
    /// Triangles faces based on vertex indices (M, 3).
    pub triangles: Array2<usize>,
    /// Vertex normals (optional).
    pub vertex_normals: Option<Array2<f64>>,
}

#[derive(Debug)]
pub struct MeshTexture {
    /// Texture image data (optional).
    pub texture: Option<Array3<u8>>,
    /// Vertex colors (optional).
    pub vertex_colors: Option<Array2<f64>>,
    /// Triangle UV coordinates (optional).
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
    pub mode: MeshMode,
}
