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
pub mod mesh_mode;
pub mod preprocessing;
pub mod utils;

use crate::types::mesh::Mesh;
pub use errors::MeshError;
pub use image::save_image_data;

/// Main entry point for loading a mesh with automatic mode inference
pub fn load_mesh_auto(
    obj_path: &str,
    texture_path: Option<&str>,
    mtl_path: Option<&str>,
    precision: u32,
    flip_v_coordinate: bool,
    ignore_opacity: bool,
) -> Result<Mesh, MeshError> {
    let mode = mesh_mode::infer_mesh_mode(texture_path, mtl_path)?;
    loaders::load_mesh_with_mode(
        obj_path,
        texture_path,
        mtl_path,
        mode,
        precision,
        flip_v_coordinate,
        ignore_opacity,
    )
}
