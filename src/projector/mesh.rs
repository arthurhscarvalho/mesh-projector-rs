// src/projector/mesh.rs
//! Mesh data structures for representing 3D geometry.

use ndarray::{Array2, Array3};
use numpy::Element;
use numpy::PyArrayMethods;
use numpy::{PyReadonlyArray2, PyReadonlyArray3};
use pyo3::FromPyObject;

#[derive(Debug, Clone)]
pub struct Mesh {
    // Geometry
    pub vertices: Array2<f64>,
    pub triangles: Array2<usize>,
    pub vertex_normals: Option<Array2<f64>>,
    pub triangle_normals: Option<Array2<f64>>, // Face normals
    // Texture
    pub vertex_colors: Option<Array2<f64>>,
    pub texture: Option<Array3<u8>>,
    pub triangle_uvs: Option<Array2<f64>>,
    // Material
    pub ka: [f64; 3], // Ambient color
    pub kd: [f64; 3], // Diffuse color
    pub ks: [f64; 3], // Specular color
    pub ns: f64,      // Specular exponent
    pub opacity: f64, // Opacity (0.0=transparent, 1.0=opaque)
    pub illum: i64,   // Illumination model
}

#[derive(FromPyObject)]
pub struct PyMesh<'py> {
    // Geometry
    pub vertices: PyReadonlyArray2<'py, f64>,
    pub triangles: PyReadonlyArray2<'py, usize>,
    pub vertex_normals: Option<PyReadonlyArray2<'py, f64>>,
    pub triangle_normals: Option<PyReadonlyArray2<'py, f64>>,
    // Texture
    pub vertex_colors: Option<PyReadonlyArray2<'py, f64>>,
    pub texture: Option<PyReadonlyArray3<'py, u8>>,
    pub triangle_uvs: Option<PyReadonlyArray2<'py, f64>>,
    // Material
    pub ka: [f64; 3],
    pub kd: [f64; 3],
    pub ks: [f64; 3],
    pub ns: f64,
    pub opacity: f64,
    pub illum: i64,
}

impl<'py> From<PyMesh<'py>> for Mesh {
    fn from(raw: PyMesh<'py>) -> Self {
        Mesh {
            vertices: raw.vertices.to_owned_array(),
            triangles: raw.triangles.to_owned_array(),
            vertex_normals: raw.vertex_normals.to_owned_array_opt(),
            triangle_normals: raw.triangle_normals.to_owned_array_opt(),
            vertex_colors: raw.vertex_colors.to_owned_array_opt(),
            texture: raw.texture.to_owned_array_opt(),
            triangle_uvs: raw.triangle_uvs.to_owned_array_opt(),
            ka: raw.ka,
            kd: raw.kd,
            ks: raw.ks,
            ns: raw.ns,
            opacity: raw.opacity,
            illum: raw.illum,
        }
    }
}

trait ToOwnedArrayOpt {
    type Output;
    fn to_owned_array_opt(self) -> Option<Self::Output>;
}

impl<'py, T: Clone + Element + 'py, D: ndarray::Dimension> ToOwnedArrayOpt
    for Option<numpy::PyReadonlyArray<'py, T, D>>
{
    type Output = ndarray::Array<T, D>;
    fn to_owned_array_opt(self) -> Option<Self::Output> {
        self.map(|a| a.to_owned_array())
    }
}
