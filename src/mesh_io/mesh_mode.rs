//! Mesh mode inference.

use crate::mesh_io::errors::{MeshError, MeshResult};
use crate::types::mesh::MeshMode;
use log::info;
use std::path::Path;

/// Infers the mesh mode based on the provided texture and material paths
pub fn infer_mesh_mode(texture_path: Option<&str>, mtl_path: Option<&str>) -> MeshResult<MeshMode> {
    match (texture_path, mtl_path) {
        (Some(tex_path), Some(mtl_path)) => infer_mesh_mode_with_both_paths(tex_path, mtl_path),
        (Some(tex_path), None) => infer_mesh_mode_with_texture_only(tex_path),
        (None, Some(mtl_path)) => infer_mesh_mode_with_material_only(mtl_path),
        (None, None) => {
            info!("Inferred OBJ_ONLY mode (No texture, no MTL path)");
            Ok(MeshMode::ObjOnly)
        }
    }
}

/// Infer mode when both texture and material paths are provided
fn infer_mesh_mode_with_both_paths(tex_path: &str, mtl_path: &str) -> MeshResult<MeshMode> {
    if Path::new(tex_path).exists() {
        if Path::new(mtl_path).exists() {
            info!(
                "Inferred FULL mode (OBJ+Texture+Explicit MTL at {})",
                mtl_path
            );
            Ok(MeshMode::Full)
        } else {
            info!(
                "Inferred FULL mode (OBJ+Texture, MTL expected via OBJ or default). Explicit path {} not found or not primary.",
                mtl_path
            );
            Ok(MeshMode::Full)
        }
    } else {
        return Err(MeshError::UnsupportedMode(format!(
            "Cannot use Full or ObjTexture mode: texture file '{}' not found or not accessible",
            tex_path
        )));
    }
}

/// Infer mode when only texture path is provided
fn infer_mesh_mode_with_texture_only(tex_path: &str) -> MeshResult<MeshMode> {
    if Path::new(tex_path).exists() {
        info!("Inferred OBJ_TEXTURE mode (OBJ+Texture, no specific MTL path provided)");
        Ok(MeshMode::ObjTexture)
    } else {
        return Err(MeshError::UnsupportedMode(format!(
            "Cannot use ObjTexture mode: texture file '{}' not found or not accessible",
            tex_path
        )));
    }
}

/// Infer mode when only material path is provided
fn infer_mesh_mode_with_material_only(mtl_path: &str) -> MeshResult<MeshMode> {
    info!(
        "Inferred OBJ_ONLY mode (No texture, MTL path {} provided but unused without texture for Full/ObjTexture mode logic)",
        mtl_path
    );
    Ok(MeshMode::ObjOnly)
}
