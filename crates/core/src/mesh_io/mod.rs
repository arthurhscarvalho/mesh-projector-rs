// src/mesh_io/mod.rs
//! Module for loading and preprocessing 3D mesh data.
//!
//! This module provides a clean, modular interface for:
//! - Loading OBJ files with materials and textures
//! - Managing different projection modes
//! - Saving projection results

pub mod errors;
pub mod image;
pub mod loaders;
pub mod material;
pub mod utils;

use crate::types::mesh::Scene;
pub use errors::MeshError;
pub use image::save_image_data;

/// Read a mesh from an OBJ file without applying preprocessing.
pub fn read_mesh(
    obj_path: &str,
    mesh_mode: Option<crate::types::mesh::ShadingMode>,
    resolution: u16,
    flip_v_coordinate: bool,
    ignore_opacity: bool,
) -> Result<Scene, MeshError> {
    loaders::read_mesh_internal(
        obj_path,
        mesh_mode,
        resolution,
        flip_v_coordinate,
        ignore_opacity,
    )
}
