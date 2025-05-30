//! Material handling for mesh I/O operations.

use crate::mesh_io::errors::{MeshError, MeshResult};
use crate::projector::mesh::{MeshMaterial, MeshMode};
use log::{info, warn};
use std::path::Path;
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

/// Load materials from an MTL file path
pub fn load_material_from_mtl(mtl_path: &str, ignore_opacity: bool) -> MeshResult<MeshMaterial> {
    if !Path::new(mtl_path).exists() {
        return Err(MeshError::FileNotFound(format!(
            "MTL file not found: {}",
            mtl_path
        )));
    }

    let mut material = create_default_material();

    match tobj::load_mtl(mtl_path) {
        Ok((materials, _)) => {
            if materials.is_empty() {
                return Err(MeshError::MtlParseError(format!(
                    "MTL file {} exists but contains no materials",
                    mtl_path
                )));
            } else {
                apply_mtl_properties(&mut material, &materials[0], ignore_opacity);
                info!("Successfully parsed MTL '{}': {:?}", mtl_path, material);
            }
        }
        Err(e) => {
            return Err(MeshError::MtlParseError(format!(
                "Error parsing MTL file {}: {}",
                mtl_path, e
            )));
        }
    }

    Ok(material)
}

/// Load and process materials from tobj result
pub fn process_materials_from_obj(
    obj_materials_result: Result<Vec<tobj::Material>, tobj::LoadError>,
    mtl_path: Option<&str>,
    mode: MeshMode,
    ignore_opacity: bool,
) -> MeshResult<Vec<MeshMaterial>> {
    match mode {
        MeshMode::ObjOnly => {
            // Always use hardcoded material for ObjOnly mode
            info!("Using hardcoded material for ObjOnly mode");
            Ok(vec![MeshMaterial {
                ka: [0.0, 0.0, 0.0], // No Ambient term from material
                kd: [1.0, 1.0, 1.0], // Base color modulation (vertex color is base)
                ks: [0.0, 0.0, 0.0], // No Specular
                ns: 10.0,
                opacity: 1.0,
                illum: 0, // Color = ShadingColor (no lighting calculation)
            }])
        }
        MeshMode::ObjTexture => {
            // Always use hardcoded material for ObjTexture mode
            info!("Using hardcoded material for ObjTexture mode");
            Ok(vec![MeshMaterial {
                ka: [0.0, 0.0, 0.0], // No Ambient
                kd: [1.0, 1.0, 1.0], // Base color modulation (texture is base)
                ks: [0.0, 0.0, 0.0], // No Specular
                ns: 10.0,            // Default Shininess
                opacity: 1.0,        // Fully Opaque
                illum: 0,            // Color = ShadingColor (no lighting calc)
            }])
        }
        MeshMode::Full => {
            // Load materials from MTL file and map based on OBJ material references
            process_materials_for_full_mode(obj_materials_result, mtl_path, ignore_opacity)
        }
    }
}

/// Get appropriate material for a submesh based on material index
pub fn get_submesh_material(
    material_idx_opt: Option<usize>,
    materials: &[MeshMaterial],
    mode: MeshMode,
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

/// Create a basic default material structure
fn create_default_material() -> MeshMaterial {
    MeshMaterial {
        ka: [0.1, 0.1, 0.1], // Default Ambient
        kd: [0.8, 0.8, 0.8], // Default Diffuse
        ks: [0.1, 0.1, 0.1], // Default Specular
        ns: 10.0,            // Default Shininess
        opacity: 1.0,        // Default Opacity
        illum: 2,            // Default Illumination model (Blinn-Phong)
    }
}

/// Apply properties from a tobj Material to our MeshMaterial
fn apply_mtl_properties(material: &mut MeshMaterial, mtl: &tobj::Material, ignore_opacity: bool) {
    if let Some(ambient) = mtl.ambient {
        material.ka = [ambient[0] as f64, ambient[1] as f64, ambient[2] as f64];
    }
    if let Some(diffuse) = mtl.diffuse {
        material.kd = [diffuse[0] as f64, diffuse[1] as f64, diffuse[2] as f64];
    }
    if let Some(specular) = mtl.specular {
        material.ks = [specular[0] as f64, specular[1] as f64, specular[2] as f64];
    }
    if let Some(shininess) = mtl.shininess {
        material.ns = shininess as f64;
    }
    if let Some(dissolve) = mtl.dissolve {
        material.opacity = dissolve as f64;
    }
    if let Some(illum) = mtl.illumination_model {
        material.illum = illum as i64;
    }

    // Ensure opacity is in valid range
    material.opacity = material.opacity.clamp(0.0, 1.0);

    // Override opacity if requested
    if ignore_opacity {
        info!("Ignoring MTL opacity and setting it to 1.0.");
        material.opacity = 1.0;
    }
}

/// Convert a tobj::Material to our MeshMaterial
fn convert_tobj_material(mtl: tobj::Material, ignore_opacity: bool) -> MeshMaterial {
    let mut material = MeshMaterial {
        ka: mtl.ambient.unwrap_or([0.1, 0.1, 0.1]).map(|v| v as f64),
        kd: mtl.diffuse.unwrap_or([0.8, 0.8, 0.8]).map(|v| v as f64),
        ks: mtl.specular.unwrap_or([0.1, 0.1, 0.1]).map(|v| v as f64),
        ns: mtl.shininess.unwrap_or(10.0) as f64,
        opacity: mtl.dissolve.unwrap_or(1.0) as f64,
        illum: mtl.illumination_model.unwrap_or(2) as i64,
    };

    if ignore_opacity {
        material.opacity = 1.0;
    }
    material.opacity = material.opacity.clamp(0.0, 1.0);
    material
}

/// Handle fallback material loading when OBJ materials fail
fn handle_fallback_material_loading(
    mtl_path: Option<&str>,
    ignore_opacity: bool,
) -> MeshResult<Vec<MeshMaterial>> {
    if let Some(path) = mtl_path {
        if Path::new(path).exists() {
            info!(
                "OBJ-linked materials not found or empty. Attempting to load from explicitly provided MTL: {}",
                path
            );
            Ok(vec![load_material_from_mtl(path, ignore_opacity)?])
        } else {
            Err(MeshError::FileNotFound(format!(
                "Explicit MTL path provided ('{}') but file not found",
                path
            )))
        }
    } else {
        warn!(
            "No MTLs linked in OBJ and no explicit MTL path provided. Using default material(s)."
        );
        Ok(Vec::new())
    }
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

/// Process materials for Full mode: load from MTL file and map based on OBJ material references
fn process_materials_for_full_mode(
    obj_materials_result: Result<Vec<tobj::Material>, tobj::LoadError>,
    mtl_path: Option<&str>,
    ignore_opacity: bool,
) -> MeshResult<Vec<MeshMaterial>> {
    match obj_materials_result {
        Ok(materials) if !materials.is_empty() => {
            info!(
                "Loaded {} materials from .mtl files linked in OBJ for Full mode.",
                materials.len()
            );
            Ok(materials
                .into_iter()
                .map(|mtl| convert_tobj_material(mtl, ignore_opacity))
                .collect())
        }
        Err(e) => {
            // OBJ parsing failed, try fallback if available
            if let Some(path) = mtl_path {
                if Path::new(path).exists() {
                    info!(
                        "OBJ-linked materials failed to load ({}). Attempting fallback MTL: {}",
                        e, path
                    );
                    Ok(vec![load_material_from_mtl(path, ignore_opacity)?])
                } else {
                    Err(MeshError::MaterialProcessingError(format!(
                        "OBJ materials failed to load and fallback MTL path '{}' not found. Error: {}",
                        path, e
                    )))
                }
            } else {
                Err(MeshError::MaterialProcessingError(format!(
                    "OBJ materials failed to load and no fallback MTL path provided. Error: {}",
                    e
                )))
            }
        }
        Ok(_) => {
            // Empty materials, try fallback
            handle_fallback_material_loading(mtl_path, ignore_opacity)
        }
    }
}
