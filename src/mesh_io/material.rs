//! Material handling for mesh I/O operations.

use crate::mesh_io::errors::{MeshError, MeshResult};
use crate::types::mesh::{MeshMaterial, ShadingMode};
use log::{info, warn};
use tobj;

/// Default material properties when no MTL file is provided or parsing fails
pub const DEFAULT_MATERIAL: MeshMaterial = MeshMaterial {
    ka: [0.0, 0.0, 0.0],
    kd: [1.0, 1.0, 1.0],
    ks: [0.0, 0.0, 0.0],
    ns: 10.0,
    opacity: 1.0,
    illum: 1,
};

/// Load and process materials from tobj result
pub fn process_materials_from_obj(
    obj_materials_result: Result<Vec<tobj::Material>, tobj::LoadError>,
    mode: ShadingMode,
    ignore_opacity: bool,
) -> MeshResult<Vec<MeshMaterial>> {
    match mode {
        ShadingMode::VertexColor => {
            // Always use hardcoded material for VertexColor mode
            info!("Using hardcoded material for VertexColor mode");
            Ok(vec![MeshMaterial {
                ka: [0.0, 0.0, 0.0], // No Ambient term from material
                kd: [1.0, 1.0, 1.0], // Base color modulation (vertex color is base)
                ks: [0.0, 0.0, 0.0], // No Specular
                ns: 10.0,
                opacity: 1.0,
                illum: 0, // Color = ShadingColor (no lighting calculation)
            }])
        }
        ShadingMode::TextureMapped => {
            // Always use hardcoded material for TextureMapped mode
            info!("Using hardcoded material for TextureMapped mode");
            Ok(vec![MeshMaterial {
                ka: [0.0, 0.0, 0.0], // No Ambient
                kd: [1.0, 1.0, 1.0], // Base color modulation (texture is base)
                ks: [0.0, 0.0, 0.0], // No Specular
                ns: 10.0,            // Default Shininess
                opacity: 1.0,        // Fully Opaque
                illum: 0,            // Color = ShadingColor (no lighting calc)
            }])
        }
        ShadingMode::BlinnPhong => {
            process_materials_for_full_mode(obj_materials_result, ignore_opacity)
        }
    }
}

/// Get appropriate material for a submesh based on material index
pub fn get_submesh_material(
    material_idx_opt: Option<usize>,
    materials: &[MeshMaterial],
    mode: ShadingMode,
    model_name: &str,
) -> MeshMaterial {
    match material_idx_opt {
        Some(idx) if idx < materials.len() => {
            log_material_assignment(&materials[idx], idx, model_name);
            materials[idx].clone()
        }
        Some(idx) => {
            warn!(
                "Material index {} out of bounds for model '{}' ({} materials loaded). Using first available material.",
                idx, model_name, materials.len()
            );
            if !materials.is_empty() {
                materials[0].clone()
            } else {
                DEFAULT_MATERIAL
            }
        }
        None => {
            // For modes with predefined materials, use the first (and likely only) material
            if !materials.is_empty() {
                info!(
                    "Model '{}' has no material assigned in OBJ. Using first available material for mode {:?}.",
                    model_name, mode
                );
                materials[0].clone()
            } else {
                warn!(
                    "Model '{}' has no material assigned and no materials loaded. Using default material.",
                    model_name
                );
                DEFAULT_MATERIAL
            }
        }
    }
}

/// Convert a tobj::Material to our MeshMaterial
fn convert_tobj_material(mtl: tobj::Material, ignore_opacity: bool) -> MeshMaterial {
    let mut material = MeshMaterial {
        ka: mtl.ambient.unwrap_or([0.1, 0.1, 0.1]).map(|v| v as f32),
        kd: mtl.diffuse.unwrap_or([0.8, 0.8, 0.8]).map(|v| v as f32),
        ks: mtl.specular.unwrap_or([0.1, 0.1, 0.1]).map(|v| v as f32),
        ns: mtl.shininess.unwrap_or(10.0) as f32,
        opacity: mtl.dissolve.unwrap_or(1.0) as f32,
        illum: mtl.illumination_model.unwrap_or(2) as i64,
    };
    if ignore_opacity {
        material.opacity = 1.0;
    }
    material.opacity = material.opacity.clamp(0.0, 1.0);
    material
}

/// Log material assignment information
fn log_material_assignment(material: &MeshMaterial, idx: usize, model_name: &str) {
    info!(
        "Assigning material '{}' ({}) to submesh from model '{}'",
        material
            .kd
            .iter()
            .map(|&x| format!("{:.2}", x))
            .collect::<Vec<_>>()
            .join(", "),
        idx,
        model_name
    );
}

/// Process materials for BlinnPhong mode: load from MTL file and map based on OBJ material references
fn process_materials_for_full_mode(
    obj_materials_result: Result<Vec<tobj::Material>, tobj::LoadError>,
    ignore_opacity: bool,
) -> MeshResult<Vec<MeshMaterial>> {
    match obj_materials_result {
        Ok(materials) if !materials.is_empty() => {
            info!(
                "Loaded {} materials from .mtl files linked in OBJ for BlinnPhong mode.",
                materials.len()
            );
            Ok(materials
                .into_iter()
                .map(|mtl| convert_tobj_material(mtl, ignore_opacity))
                .collect())
        }
        Err(e) => Err(MeshError::MaterialProcessingError(format!(
            "OBJ materials failed to load. Error: {}",
            e
        ))),
        Ok(_) => Ok(Vec::new()),
    }
}
