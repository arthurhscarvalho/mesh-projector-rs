//! Core data structures for representing 3D mesh geometry, textures, and materials.
//!
//! This module defines the types used to describe the geometry (vertices, faces, normals, colors),
//! texture mapping (UVs, images), and material properties (PBR parameters) of 3D models.
//! These structures form the foundation for mesh loading, transformation, and rendering
//! throughout the projection and rasterization pipeline.

use ndarray::{Array2, Array3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadingMode {
    /// Uses per-vertex colors for shading without additional lighting.
    VertexColor,
    /// Applies a texture directly to the mesh (unlit).
    TextureMapped,
    /// Lit shading using the Blinn-Phong model.
    BlinnPhong,
}

pub struct MeshGeometry {
    /// Vertex positions (N, 3).
    pub vertices: Array2<f32>,
    /// Triangles faces based on vertex indices (M, 3).
    pub triangles: Array2<usize>,
    /// Vertex normals (optional).
    pub vertex_normals: Option<Array2<f32>>,
    /// Precomputed triangle face normals (M, 3), in world space (derived from scaled vertices).
    pub face_normals: Array2<f32>,
    /// Vertex colors (N, 3) (optional, primarily for VertexColor mode).
    pub vertex_colors: Option<Array2<f32>>,
}

pub struct MeshTexture {
    /// Texture image data (optional).
    pub texture: Option<Array3<u8>>,
    /// Triangle UV coordinates (optional).
    pub triangle_uvs: Option<Array2<f32>>,
    /// Number of color channels in the texture (e.g., 3 for RGB, 4 for RGBA).
    pub channels: usize,
    /// Height of the texture as a floating-point number.
    pub height_f32: f32,
    /// Width of the texture as a floating-point number.
    pub width_f32: f32,
}

#[derive(Debug, Clone)]
pub struct MeshMaterial {
    pub ka: [f32; 3], // Ambient color
    pub kd: [f32; 3], // Diffuse color
    pub ks: [f32; 3], // Specular color
    pub ns: f32,      // Specular exponent
    pub opacity: f32, // Opacity (0.0=transparent, 1.0=opaque)
    pub illum: i64,   // Illumination model
}

pub struct Mesh<'a> {
    /// Geometry of the mesh.
    pub geometry: MeshGeometry,
    /// Material of the mesh.
    pub material: MeshMaterial,
    /// Texture of the mesh.
    pub texture: Option<&'a MeshTexture>,
}

pub struct Scene<'a> {
    // Meshes: each mesh has its own geometry, texture, and material
    pub meshes: Vec<Mesh<'a>>,
    // Textures: shared textures for all meshes
    pub textures: Option<Vec<&'a MeshTexture>>,
    // Rendering mode
    pub mode: ShadingMode,
}
