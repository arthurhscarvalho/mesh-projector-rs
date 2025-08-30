// src/mesh_io/mod.rs
//! Module for loading and preprocessing 3D mesh data.
//!
//! This module provides a clean, modular interface for:
//! - Loading OBJ files with materials and textures
//! - Preprocessing mesh geometry
//! - Managing different projection modes
//! - Saving projection results

pub mod errors;
pub mod image;
pub mod loaders;
pub mod material;
pub mod preprocessing;
pub mod utils;

use crate::types::mesh::Scene;
pub use errors::MeshError;
pub use image::save_image_data;

/// Read a mesh from an OBJ file. Optionally provide a mesh mode; when None, the mode is inferred
/// from the OBJ/MTL contents (textures/materials discovered while reading the OBJ).
pub fn read_mesh(
    obj_path: &str,
    mesh_mode: Option<crate::types::mesh::MeshMode>,
    precision: u32,
    flip_v_coordinate: bool,
    ignore_opacity: bool,
) -> Result<Scene<'static>, MeshError> {
    loaders::read_mesh_internal(
        obj_path,
        mesh_mode,
        precision,
        flip_v_coordinate,
        ignore_opacity,
    )
}
