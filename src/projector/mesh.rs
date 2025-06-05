//! Core data structures for representing 3D mesh geometry, textures, and materials.
//!
//! This module defines the types used to describe the geometry (vertices, faces, normals, colors),
//! texture mapping (UVs, images), and material properties (PBR parameters) of 3D models.
//! These structures form the foundation for mesh loading, transformation, and rendering
//! throughout the projection and rasterization pipeline.

use ndarray::{Array2, Array3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshMode {
    /// Only the OBJ file is used.
    ObjOnly,
    /// The OBJ file and texture are used.
    ObjTexture,
    /// The OBJ file, texture, and material are used.
    Full,
}

#[derive(Debug, Clone)]
pub struct MeshGeometry {
    /// Vertex positions (N, 3).
    pub vertices: Array2<f64>,
    /// Triangles faces based on vertex indices (M, 3).
    pub triangles: Array2<usize>,
    /// Vertex normals (optional).
    pub vertex_normals: Option<Array2<f64>>,
    /// Precomputed triangle face normals (M, 3), in world space (derived from scaled vertices).
    pub face_normals: Array2<f64>,
    /// Vertex colors (N, 3) (optional, primarily for ObjOnly mode).
    pub vertex_colors: Option<Array2<f64>>,
}

#[derive(Debug, Clone)]
pub struct MeshTexture {
    /// Texture image data (optional).
    pub texture: Option<Array3<u8>>,
    /// Triangle UV coordinates (optional).
    pub triangle_uvs: Option<Array2<f64>>,
    /// Number of color channels in the texture (e.g., 3 for RGB, 4 for RGBA).
    pub channels: usize,
    /// Height of the texture as a floating-point number.
    pub height_f64: f64,
    /// Width of the texture as a floating-point number.
    pub width_f64: f64,
}

#[derive(Debug, Clone)]
pub struct MeshMaterial {
    pub ka: [f64; 3], // Ambient color
    pub kd: [f64; 3], // Diffuse color
    pub ks: [f64; 3], // Specular color
    pub ns: f64,      // Specular exponent
    pub opacity: f64, // Opacity (0.0=transparent, 1.0=opaque)
    pub illum: i64,   // Illumination model
}

#[derive(Debug, Clone)]
pub struct SubMesh {
    pub geometry: MeshGeometry,
    pub material: MeshMaterial,
}

#[derive(Debug, Clone)]
pub struct Mesh {
    // Submeshes: each submesh has its own geometry, texture, material, and mode
    pub submeshes: Vec<SubMesh>,
    // Texture: shared texture for all submeshes
    pub texture: MeshTexture,
    // Rendering mode
    pub mode: MeshMode,
}
