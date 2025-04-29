// src/projector/mesh.rs
//! Mesh data structures for representing 3D geometry.

use crate::projector::texture::TextureData;
use nd::{ArrayView2, ArrayView3};

/// Represents the core geometry of a 3D mesh.
#[derive(Debug, Clone)]
pub struct Mesh<'a> {
    /// Vertex positions as a (N, 3) array of float32 values.
    pub vertices: ArrayView2<'a, f32>,
    /// Triangle face indices as a (M, 3) array of usize values.
    pub triangles: ArrayView2<'a, usize>,
}

/// Optional attributes that may be present for a mesh.
#[derive(Debug, Clone)]
pub struct MeshAttributes<'a> {
    /// Optional vertex normal vectors as a (N, 3) array.
    pub vertex_normals: Option<ArrayView2<'a, f32>>,
    /// Optional vertex colors as a (N, 3) array.
    pub vertex_colors: Option<ArrayView2<'a, f32>>,
    /// Optional texture data for the mesh.
    pub texture: Option<TextureData<'a>>,
    /// Optional UV coordinates for each triangle vertex as a (M*3, 2) array.
    pub triangle_uvs: Option<ArrayView2<'a, f32>>,
}

impl<'a> Mesh<'a> {
    /// Creates a new mesh from vertex and triangle data.
    pub fn new(vertices: ArrayView2<'a, f32>, triangles: ArrayView2<'a, usize>) -> Self {
        Mesh {
            vertices,
            triangles,
        }
    }
}

impl<'a> MeshAttributes<'a> {
    /// Creates a new mesh attributes object with the provided data.
    pub fn new(
        vertex_normals: Option<ArrayView2<'a, f32>>,
        vertex_colors: Option<ArrayView2<'a, f32>>,
        texture: Option<ArrayView3<'a, u8>>,
        triangle_uvs: Option<ArrayView2<'a, f32>>,
    ) -> Self {
        MeshAttributes {
            vertex_normals,
            vertex_colors,
            texture: texture.map(TextureData::new),
            triangle_uvs,
        }
    }
}
