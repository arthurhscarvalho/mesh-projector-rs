// src/projector/io.rs
//! Module for loading and preprocessing 3D mesh data.
use crate::projector::geometry::precompute_triangle_normals;
use crate::projector::mesh::{Mesh, MeshGeometry, MeshMaterial, MeshMode, MeshTexture, SubMesh};
use image::{ImageBuffer, Rgb};
use log::{info, warn};
use ndarray::{s, Array, Array2, Array3, Array4, Axis};
use std::fs;
use std::path::Path;
use tobj;

/// Default material properties when no MTL file is provided or parsing fails
const DEFAULT_MATERIAL: MeshMaterial = MeshMaterial {
    ka: [0.0, 0.0, 0.0],
    kd: [1.0, 1.0, 1.0],
    ks: [0.0, 0.0, 0.0],
    ns: 10.0,
    opacity: 1.0,
    illum: 1,
};

/// Loads a 3D mesh from an OBJ file and optionally its texture and material
/// The projection mode is automatically inferred based on provided files.
///
/// # Arguments
/// * `obj_path` - Path to the OBJ file
/// * `texture_path` - Optional path to the texture image file
/// * `mtl_path` - Optional path to the MTL material file
/// * `precision` - Precision for vertex scaling (output resolution 2^precision)
/// * `flip_v_coordinate` - Whether to flip the V texture coordinate (1.0 - v)
/// * `ignore_opacity` - If true, forces opacity to 1.0 regardless of MTL
///
/// # Returns
/// A Mesh structure containing the loaded and preprocessed mesh data
pub fn load_mesh(
    obj_path: &str,
    texture_path: Option<&str>,
    mtl_path: Option<&str>,
    precision: u32,
    flip_v_coordinate: bool,
    ignore_opacity: bool,
) -> Result<Mesh, String> {
    // Infer mode based on provided paths
    let inferred_mode = infer_projection_mode(texture_path, mtl_path);
    // Load the mesh with the inferred mode
    load_mesh_with_mode(
        obj_path,
        texture_path,
        mtl_path,
        inferred_mode,
        precision,
        flip_v_coordinate,
        ignore_opacity,
    )
}

/// Loads a 3D mesh using a specific projection mode
///
/// # Arguments
/// * `obj_path` - Path to the OBJ file
/// * `texture_path` - Optional path to the texture image file
/// * `mtl_path` - Optional path to the MTL material file (can be inferred from OBJ)
/// * `mode` - Projection mode determining which components are needed
/// * `precision` - Precision for vertex scaling (output resolution 2^precision)
/// * `flip_v_coordinate` - Whether to flip the V texture coordinate (1.0 - v)
/// * `ignore_opacity` - If true, forces opacity to 1.0 regardless of MTL
///
/// # Returns
/// A Mesh structure containing the loaded and preprocessed mesh data
pub fn load_mesh_with_mode(
    obj_path: &str,
    texture_path: Option<&str>,
    mtl_path: Option<&str>, // This mtl_path can be an override or a hint
    mode: MeshMode,
    precision: u32,
    flip_v_coordinate: bool,
    ignore_opacity: bool,
) -> Result<Mesh, String> {
    info!("Loading mesh: {} in mode: {:?}", obj_path, mode);
    if !Path::new(obj_path).exists() {
        return Err(format!("OBJ file not found: {}", obj_path));
    }

    // Determine the MTL file to load.
    // tobj::load_obj attempts to load materials specified in the OBJ file itself.
    // If an mtl_path is provided externally, it might be an override or a specific choice.
    // For simplicity, we'll let tobj handle the initial material loading based on the OBJ.
    // The provided mtl_path could be used later if tobj fails or as a primary source if needed.
    let load_options = tobj::LoadOptions {
        triangulate: true,
        single_index: true, // Important for how we handle UVs and normals
        ..Default::default()
    };

    let obj_result = tobj::load_obj(obj_path, &load_options);

    let (obj_models, obj_materials_result) = match obj_result {
        Ok((models, materials_res)) => (models, materials_res),
        Err(e) => return Err(format!("Failed to load OBJ file: {}", e)),
    };

    if obj_models.is_empty() {
        return Err("No models (geometric groups) found in the OBJ file".to_string());
    }

    // Resolve materials:
    // 1. Try loading from the .mtl files specified in the OBJ.
    // 2. If mtl_path override is given and no materials were loaded by tobj, try that.
    // 3. Fallback to default materials if needed.
    let materials = match obj_materials_result {
        Ok(mats) if !mats.is_empty() => {
            info!(
                "Loaded {} materials from .mtl files linked in OBJ.",
                mats.len()
            );
            mats.into_iter()
                .map(|m| {
                    let mut mat = MeshMaterial {
                        // Convert from tobj::Material to our MeshMaterial
                        ka: m.ambient.unwrap_or([0.1, 0.1, 0.1]).map(|v| v as f64),
                        kd: m.diffuse.unwrap_or([0.8, 0.8, 0.8]).map(|v| v as f64),
                        ks: m.specular.unwrap_or([0.1, 0.1, 0.1]).map(|v| v as f64),
                        ns: m.shininess.unwrap_or(10.0) as f64,
                        opacity: m.dissolve.unwrap_or(1.0) as f64,
                        illum: m.illumination_model.unwrap_or(2) as i64,
                    };
                    if ignore_opacity {
                        mat.opacity = 1.0;
                    }
                    mat.opacity = mat.opacity.max(0.0).min(1.0);
                    mat
                })
                .collect::<Vec<_>>()
        }
        _ => {
            // tobj either failed to load/parse MTLs or OBJ didn't specify any
            if let Some(path) = mtl_path {
                if Path::new(path).exists() {
                    info!("OBJ-linked materials not found or empty. Attempting to load from explicitly provided MTL: {}", path);
                    // This part needs a function similar to load_material_from_mtl but returning Vec<MeshMaterial>
                    // For now, let's assume load_material_from_mtl can be adapted or we load the first one.
                    // This is a simplification. A robust solution would parse the entire MTL for multiple named materials.
                    vec![load_material_from_mtl(path, ignore_opacity)]
                } else {
                    warn!("Explicit MTL path provided ('{}') but file not found. Using default material(s).", path);
                    Vec::new() // Empty, will use default per submesh
                }
            } else {
                warn!("No MTLs linked in OBJ and no explicit MTL path provided. Using default material(s).");
                Vec::new() // Empty, will use default per submesh
            }
        }
    };

    let mut submeshes = Vec::new();
    let mut global_vertex_positions: Vec<f32> = Vec::new();
    let mut global_vertex_normals: Option<Vec<f32>> = Some(Vec::new());
    let mut global_vertex_texcoords: Option<Vec<f32>> = Some(Vec::new());
    let mut global_vertex_colors: Option<Vec<f32>> = Some(Vec::new()); // Not directly used by SubMesh yet, but good to parse

    let mut vertex_offset = 0;

    // First pass: aggregate all vertex data to create a global pool
    // This is because OBJ indices are global across the file if not split by 'o' objects
    // but tobj::Model already gives us per-model data with local indices.
    // We need to re-globalize them if we want to share vertex attributes like texture.
    // For simplicity here, we'll process each tobj::Model as a potential SubMesh,
    // but all its vertex data (positions, normals, UVs) will be relative to its own data block.
    // Then, we'll create a combined attribute set for the shared texture.

    let mut all_models_vertex_positions: Vec<f32> = Vec::new();
    let mut all_models_vertex_texcoords: Vec<f32> = Vec::new(); // For shared texture
    let mut all_models_vertex_count = 0;

    for model in &obj_models {
        all_models_vertex_positions.extend_from_slice(&model.mesh.positions);
        if !model.mesh.texcoords.is_empty() {
            all_models_vertex_texcoords.extend_from_slice(&model.mesh.texcoords);
        } else if mode == MeshMode::ObjTexture || mode == MeshMode::Full {
            // If any model is missing texcoords and they are needed, it's an issue for shared texture.
            warn!("Model '{}' is missing texture coordinates. This might affect shared texture mapping.", model.name);
            // Fill with zeros for this model's vertex count to maintain array correspondence
            all_models_vertex_texcoords.resize(
                all_models_vertex_texcoords.len() + model.mesh.positions.len() / 3 * 2,
                0.0,
            );
        }
        all_models_vertex_count += model.mesh.positions.len() / 3;
    }

    // Process texture and UVs once for the shared texture
    let (texture_data, global_triangle_uvs) = if mode == MeshMode::Full
        || mode == MeshMode::ObjTexture
    {
        let tex_path = texture_path.ok_or_else(|| {
            format!(
                "Texture file is required for mode {:?} but none was provided",
                mode
            )
        })?;
        if !all_models_vertex_texcoords.is_empty()
            && all_models_vertex_texcoords.len() == all_models_vertex_count * 2
        {
            info!("Using combined UV coordinates from all OBJ models for shared texture.");
            let loaded_texture = load_texture(tex_path)?;
            info!(
                "Shared texture load successful: {} ({}x{}x{})",
                tex_path,
                loaded_texture.shape()[0],
                loaded_texture.shape()[1],
                loaded_texture.shape()[2]
            );

            let mut uvs_arr = Array::zeros((all_models_vertex_count, 2));
            for i in 0..all_models_vertex_count {
                for j in 0..2 {
                    let mut value = all_models_vertex_texcoords[i * 2 + j] as f64;
                    value = value.max(0.0).min(1.0); // Clamp UVs
                    if j == 1 && flip_v_coordinate {
                        value = 1.0 - value;
                    }
                    uvs_arr[[i, j]] = value;
                }
            }
            // Note: global_triangle_uvs will be mapped per submesh later, using their local triangle indices
            // referring to this global uvs_arr. For now, we store the per-vertex UVs.
            // The MeshTexture struct expects triangle_uvs, so this needs careful handling.
            // Let's defer full triangle UV generation until we process submeshes,
            // or adapt MeshTexture to also store per-vertex UVs and map them during rendering.
            // For now, let's create a placeholder or an empty array for MeshTexture.triangle_uvs
            // and the actual mapping logic will need to be in how SubMesh geometry indices map to these global UVs.

            // This is tricky. If SubMeshes have their own vertex indexing,
            // they can't directly use a global UV array indexed 0..N-1 for all vertices in the file.
            // Each SubMesh's vertices are a subset.
            // A common approach:
            // 1. Load ALL unique vertices (pos, normal, uv) from the OBJ.
            // 2. Each SubMesh then has faces that index into this global pool.
            // 'tobj' with single_index gives us positions, normals, texcoords in separate arrays,
            // and face indices are for positions. Normals and texcoords are assumed to be 1:1 with positions.

            // Simpler for now: assume MeshTexture.triangle_uvs will be populated from SubMesh processing
            // if they all reference a common set of vertex attributes.
            // OR, the MeshTexture can hold the raw per-vertex UVs (`uvs_arr` here) and SubMeshes
            // construct their `triangle_uvs` from it using their specific `triangle` indices.
            // Let's assume the latter for now, meaning `MeshTexture` needs a field for per-vertex UVs,
            // or `triangle_uvs` is constructed carefully.
            // For now, let's pass `uvs_arr` and figure out `triangle_uvs` generation.
            // The current MeshTexture only has `triangle_uvs`. This is a structural change needed.
            // Let's assume for a moment `MeshTexture` will store this `uvs_arr` as `vertex_uvs`. (This is a placeholder for a needed struct change)

            // The current structure of MeshTexture expects triangle_uvs.
            // We need to build this for ALL triangles from ALL submeshes, using the global uvs_arr.
            let mut temp_all_triangles_for_uv_mapping: Vec<usize> = Vec::new();
            let mut current_vertex_idx_offset_for_uv = 0;
            for model in &obj_models {
                for index in &model.mesh.indices {
                    temp_all_triangles_for_uv_mapping
                        .push(*index as usize + current_vertex_idx_offset_for_uv);
                }
                current_vertex_idx_offset_for_uv += model.mesh.positions.len() / 3;
            }

            let num_total_corners = temp_all_triangles_for_uv_mapping.len();
            let mut all_tri_uvs_arr = Array::zeros((num_total_corners / 3, 3, 2)); // M, 3 corners, 2 UV components
            for i in 0..(num_total_corners / 3) {
                // For each triangle
                for k_vert_of_tri in 0..3 {
                    // For each vertex of the triangle
                    let global_vertex_idx =
                        temp_all_triangles_for_uv_mapping[i * 3 + k_vert_of_tri];
                    if global_vertex_idx < uvs_arr.shape()[0] {
                        all_tri_uvs_arr[[i, k_vert_of_tri, 0]] = uvs_arr[[global_vertex_idx, 0]];
                        all_tri_uvs_arr[[i, k_vert_of_tri, 1]] = uvs_arr[[global_vertex_idx, 1]];
                    } else {
                        warn!("Global vertex index {} out of bounds for UV mapping (UV array size: {}). Setting UV to (0,0).", global_vertex_idx, uvs_arr.shape()[0]);
                        all_tri_uvs_arr[[i, k_vert_of_tri, 0]] = 0.0;
                        all_tri_uvs_arr[[i, k_vert_of_tri, 1]] = 0.0;
                    }
                }
            }
            // Reshape M,3,2 to M*3, 2 for current MeshTexture.triangle_uvs
            let reshaped_tri_uvs = all_tri_uvs_arr.into_shape((num_total_corners, 2)).unwrap();

            (Some(loaded_texture), Some(reshaped_tri_uvs))
        } else {
            let reason = if all_models_vertex_texcoords.is_empty() {
                "entirely missing from all models"
            } else {
                "inconsistent or incomplete across models"
            };
            warn!(
                "Mesh has no usable UV coordinates for shared texture ({}). Texture mapping will not be possible for mode {:?}.",
                reason, mode
            );
            return Err(format!(
                "Mesh has no usable UV coordinates for shared texture ({}). Required for mode {:?}",
                reason, mode
            ));
        }
    } else {
        (None, None)
    };

    let mut current_global_vertex_offset = 0;
    for (model_idx, model) in obj_models.iter().enumerate() {
        let mesh = &model.mesh;
        if mesh.positions.is_empty() {
            warn!(
                "Skipping model '{}' as it has no vertex positions.",
                model.name
            );
            continue;
        }

        let num_model_vertices = mesh.positions.len() / 3;

        // --- Geometry Processing for SubMesh ---
        let mut submesh_vertices = Array::zeros((num_model_vertices, 3));
        for i in 0..num_model_vertices {
            for j in 0..3 {
                submesh_vertices[[i, j]] = mesh.positions[i * 3 + j] as f64;
            }
        }

        let mut submesh_triangles = Array::zeros((mesh.indices.len() / 3, 3));
        for i in 0..submesh_triangles.shape()[0] {
            for j in 0..3 {
                // Indices are local to this model's vertices
                submesh_triangles[[i, j]] = mesh.indices[i * 3 + j] as usize;
            }
        }

        let submesh_vertex_normals = if !mesh.normals.is_empty()
            && mesh.normals.len() == num_model_vertices * 3
        {
            info!(
                "Using vertex normals from OBJ data for model '{}'.",
                model.name
            );
            let mut normals_arr = Array::zeros((num_model_vertices, 3));
            for i in 0..num_model_vertices {
                for j in 0..3 {
                    normals_arr[[i, j]] = mesh.normals[i * 3 + j] as f64;
                }
            }
            normalize_normals(&mut normals_arr);
            Some(normals_arr)
        } else {
            if !mesh.normals.is_empty() {
                warn!("Vertex normal data for model '{}' was inconsistent or incomplete. Recomputing normals for this submesh.", model.name);
            } else {
                warn!(
                    "Model '{}' does not have vertex normals. Computing them for this submesh.",
                    model.name
                );
            }
            compute_vertex_normals(&submesh_vertices, &submesh_triangles)
        };

        // Vertex colors for this submesh
        let sub_vertex_colors = if !mesh.vertex_color.is_empty()
            && mesh.vertex_color.len() == num_model_vertices * 3
        {
            info!(
                "Using vertex colors from OBJ data for model '{}'.",
                model.name
            );
            let mut colors_arr = Array::zeros((num_model_vertices, 3));
            for i in 0..num_model_vertices {
                for j in 0..3 {
                    // Assuming 3 components (RGB) per color. tobj usually gives 3.
                    // Values are typically 0.0-1.0 range from tobj if present.
                    colors_arr[[i, j]] = mesh.vertex_color[i * 3 + j] as f64;
                }
            }
            Some(colors_arr)
        } else {
            if !mesh.vertex_color.is_empty() {
                warn!("Vertex color data for model '{}' was present but inconsistent (expected {}*3, got {}). Not using vertex colors for this submesh.", 
                       model.name, num_model_vertices, mesh.vertex_color.len());
            }
            // If no vertex colors, or inconsistent, this submesh won't have them.
            // ObjOnly mode in rasterizer will then default to white.
            None
        };

        let processed_submesh_vertices = preprocess_vertices(submesh_vertices.clone(), precision);

        let submesh_face_normals = precompute_triangle_normals(
            processed_submesh_vertices.view(),
            submesh_triangles.view(),
        );

        let geometry = MeshGeometry {
            vertices: processed_submesh_vertices,
            triangles: submesh_triangles,
            vertex_normals: submesh_vertex_normals,
            face_normals: submesh_face_normals,
            vertex_colors: sub_vertex_colors, // Assign parsed vertex colors here
        };

        // --- Material for SubMesh ---
        // Determine material for this submesh
        let material_idx_opt = mesh.material_id;
        let submesh_material = match material_idx_opt {
            Some(idx) if idx < materials.len() => {
                info!(
                    "Assigning material '{}' ({}) to submesh from model '{}'",
                    materials[idx]
                        .kd
                        .iter()
                        .map(|&x| format!("{:.2}", x))
                        .collect::<Vec<_>>()
                        .join(", "),
                    idx,
                    model.name
                );
                materials[idx].clone()
            }
            Some(idx) => {
                warn!("Material index {} out of bounds for model '{}' ({} materials loaded). Using default material.", idx, model.name, materials.len());
                DEFAULT_MATERIAL
            }
            None => {
                if mode == MeshMode::Full {
                    // Only warn if FULL mode expects materials
                    warn!(
                        "Model '{}' has no material assigned in OBJ. Using default material.",
                        model.name
                    );
                } else {
                    info!(
                        "Model '{}' has no material assigned in OBJ. Using mode-specific default.",
                        model.name
                    );
                }
                // Use mode-specific defaults if no explicit material
                match mode {
                    MeshMode::Full => DEFAULT_MATERIAL, // Should ideally have a material
                    MeshMode::ObjTexture => MeshMaterial {
                        // Default for texturing
                        ka: [0.0, 0.0, 0.0],
                        kd: [1.0, 1.0, 1.0],
                        ks: [0.0, 0.0, 0.0],
                        ns: 10.0,
                        opacity: 1.0,
                        illum: 0, // illum 0: Color = ShadingColor
                    },
                    MeshMode::ObjOnly => MeshMaterial {
                        // Default for vertex colors / simple shading
                        ka: [0.0, 0.0, 0.0],
                        kd: [1.0, 1.0, 1.0],
                        ks: [0.0, 0.0, 0.0],
                        ns: 10.0,
                        opacity: 1.0,
                        illum: 0,
                    },
                }
            }
        };

        submeshes.push(SubMesh {
            geometry,
            material: submesh_material,
        });
        current_global_vertex_offset += num_model_vertices;
    }

    if submeshes.is_empty() {
        return Err("No processable submeshes found in the OBJ file.".to_string());
    }

    // Create the shared MeshTexture
    // The `triangle_uvs` in `MeshTexture` should be the global one we prepared.
    // Vertex colors are not part of the shared texture in the new design, but part of individual geometry if needed.
    // For now, `vertex_colors` in `MeshTexture` can be None.
    let (texture_channels, texture_height, texture_width) = if let Some(ref tex) = texture_data {
        (tex.shape()[2], tex.shape()[0] as f64, tex.shape()[1] as f64)
    } else {
        (0, 0.0, 0.0) // Default to 0 channels if no texture
    };

    let shared_texture = MeshTexture {
        texture: texture_data,
        vertex_colors: None, // Ensure this is None, as vertex colors are now per-submesh
        triangle_uvs: global_triangle_uvs, // This was prepared earlier from all_models_vertex_texcoords
        channels: texture_channels,
        height_f64: texture_height,
        width_f64: texture_width,
    };

    let mesh_result = Mesh {
        submeshes,
        texture: shared_texture,
        mode,
    };

    Ok(mesh_result)
}

/// Loads material properties from an MTL file
// This function needs to be adapted if we expect it to parse all named materials from an MTL file.
// For now, it loads the *first* material or uses defaults.
// If an MTL file has multiple named materials, tobj::load_mtl itself returns a Vec<tobj::Material>.
// The main `load_mesh_with_mode` function already uses `tobj::load_mtl` (via `tobj::load_obj` which calls it).
// So this standalone `load_material_from_mtl` might be for an override scenario where OBJ doesn't link an MTL.
fn load_material_from_mtl(mtl_path: &str, ignore_opacity: bool) -> MeshMaterial {
    if !Path::new(mtl_path).exists() {
        warn!(
            "MTL file not found: {}. Using default material values.",
            mtl_path
        );
        return DEFAULT_MATERIAL;
    }
    let mut material = MeshMaterial {
        ka: [0.1, 0.1, 0.1], // Default Ambient
        kd: [0.8, 0.8, 0.8], // Default Diffuse
        ks: [0.1, 0.1, 0.1], // Default Specular
        ns: 10.0,            // Default Shininess
        opacity: 1.0,        // Default Opacity
        illum: 2,            // Default Illumination model (e.g., Blinn-Phong)
    };
    let parsed_successfully = match tobj::load_mtl(mtl_path) {
        Ok((materials, _)) => {
            if materials.is_empty() {
                warn!(
                    "MTL file {} exists but contains no materials. Using defaults.",
                    mtl_path
                );
                false
            } else {
                // Use the first material
                let mtl = &materials[0];
                // Apply material properties
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
                material.opacity = material.opacity.max(0.0).min(1.0);
                // Override opacity if requested
                if ignore_opacity {
                    info!("Ignoring MTL opacity and setting it to 1.0.");
                    material.opacity = 1.0;
                }
                true
            }
        }
        Err(e) => {
            warn!(
                "Error parsing MTL file {}: {}. Using default values.",
                mtl_path, e
            );
            false
        }
    };
    if parsed_successfully {
        info!("Successfully parsed MTL '{}': {:?}", mtl_path, material);
    } else {
        warn!(
            "Failed to parse MTL '{}', using defaults: {:?}",
            mtl_path, material
        );
    }
    material
}

/// Load a texture image and convert it to an RGB ndarray
fn load_texture(texture_path: &str) -> Result<Array3<u8>, String> {
    if !Path::new(texture_path).exists() {
        return Err(format!("Texture file not found: {}", texture_path));
    }
    match image::open(texture_path) {
        Ok(img) => {
            let rgb_img = img.to_rgb8();
            let (width, height) = rgb_img.dimensions();

            let mut texture_array = Array3::zeros((height as usize, width as usize, 3));

            for y in 0..height {
                for x in 0..width {
                    let pixel = rgb_img.get_pixel(x, y);
                    texture_array[[y as usize, x as usize, 0]] = pixel[0];
                    texture_array[[y as usize, x as usize, 1]] = pixel[1];
                    texture_array[[y as usize, x as usize, 2]] = pixel[2];
                }
            }

            Ok(texture_array)
        }
        Err(e) => Err(format!("Failed to load texture '{}': {}", texture_path, e)),
    }
}

/// Compute vertex normals for a mesh by averaging face normals
fn compute_vertex_normals(
    vertices: &Array2<f64>,
    triangles: &Array2<usize>,
) -> Option<Array2<f64>> {
    let vertices_count = vertices.shape()[0];
    if vertices_count == 0 || triangles.shape()[0] == 0 {
        return None;
    }
    // Initialize normals to zero
    let mut normals = Array2::zeros((vertices_count, 3));
    // For each triangle, compute its normal and add to the normals of its vertices
    for i in 0..triangles.shape()[0] {
        let idx1 = triangles[[i, 0]];
        let idx2 = triangles[[i, 1]];
        let idx3 = triangles[[i, 2]];
        // Get triangle vertices
        let v1 = [
            vertices[[idx1, 0]],
            vertices[[idx1, 1]],
            vertices[[idx1, 2]],
        ];
        let v2 = [
            vertices[[idx2, 0]],
            vertices[[idx2, 1]],
            vertices[[idx2, 2]],
        ];
        let v3 = [
            vertices[[idx3, 0]],
            vertices[[idx3, 1]],
            vertices[[idx3, 2]],
        ];
        // Compute edges
        let edge1 = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]];
        let edge2 = [v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]];
        // Compute cross product (normal)
        let normal = [
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0],
        ];
        // Normalize the normal if it's not too small
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if length > 1e-9 {
            let normalized = [normal[0] / length, normal[1] / length, normal[2] / length];
            // Add to vertex normals
            for j in 0..3 {
                normals[[idx1, j]] += normalized[j];
                normals[[idx2, j]] += normalized[j];
                normals[[idx3, j]] += normalized[j];
            }
        }
    }
    // Normalize all vertex normals
    normalize_normals(&mut normals);
    Some(normals)
}

/// Normalize an array of normal vectors
fn normalize_normals(normals: &mut Array2<f64>) {
    for i in 0..normals.shape()[0] {
        let mut norm = 0.0;
        for j in 0..3 {
            norm += normals[[i, j]] * normals[[i, j]];
        }
        norm = norm.sqrt();

        if norm > 1e-9 {
            for j in 0..3 {
                normals[[i, j]] /= norm;
            }
        } else {
            // For zero normals, set them to a default direction
            normals[[i, 0]] = 0.0;
            normals[[i, 1]] = 0.0;
            normals[[i, 2]] = 1.0;
        }
    }
}

/// Preprocess vertices by centering the mesh and scaling it to fit within the target precision
pub fn preprocess_vertices(mut vertices: Array2<f64>, precision: u32) -> Array2<f64> {
    // Get bounding box
    let mut min_bound = [f64::MAX, f64::MAX, f64::MAX];
    let mut max_bound = [f64::MIN, f64::MIN, f64::MIN];
    for i in 0..vertices.shape()[0] {
        for j in 0..3 {
            min_bound[j] = min_bound[j].min(vertices[[i, j]]);
            max_bound[j] = max_bound[j].max(vertices[[i, j]]);
        }
    }
    info!("Original BBox Min: {:?}, Max: {:?}", min_bound, max_bound);

    // Calculate center of mass
    let mut center = [0.0, 0.0, 0.0];
    for j in 0..3 {
        center[j] = (min_bound[j] + max_bound[j]) / 2.0;
    }
    info!("Center of mass: {:?}", center);

    // Center the mesh by translating vertices
    for i in 0..vertices.shape()[0] {
        for j in 0..3 {
            vertices[[i, j]] -= center[j];
        }
    }

    // Update bounds after centering
    min_bound = [f64::MAX, f64::MAX, f64::MAX];
    max_bound = [f64::MIN, f64::MIN, f64::MIN];
    for i in 0..vertices.shape()[0] {
        for j in 0..3 {
            min_bound[j] = min_bound[j].min(vertices[[i, j]]);
            max_bound[j] = max_bound[j].max(vertices[[i, j]]);
        }
    }
    info!("Centered BBox Min: {:?}, Max: {:?}", min_bound, max_bound);

    // Calculate scale factor based on max dimension
    let mut bbox_size = [
        max_bound[0] - min_bound[0],
        max_bound[1] - min_bound[1],
        max_bound[2] - min_bound[2],
    ];
    // Avoid division by zero
    for j in 0..3 {
        if bbox_size[j] == 0.0 {
            bbox_size[j] = 1e-9;
        }
    }
    let max_dim = bbox_size[0].max(bbox_size[1]).max(bbox_size[2]);
    let target_max_val = (1 << precision) as f64 - 1.0;

    // Add a small safety margin to ensure we have room for centering
    let safety_margin = 0.95; // Use 95% of the available space
    let scale_factor = (target_max_val * safety_margin) / max_dim;
    info!(
        "Scaling vertices. Max dimension: {}, Scale factor: {} (with {}% safety margin)",
        max_dim,
        scale_factor,
        safety_margin * 100.0
    );

    // Scale vertices
    for i in 0..vertices.shape()[0] {
        for j in 0..3 {
            vertices[[i, j]] *= scale_factor;
        }
    }

    // Calculate bounds after scaling
    min_bound = [f64::MAX, f64::MAX, f64::MAX];
    max_bound = [f64::MIN, f64::MIN, f64::MIN];
    for i in 0..vertices.shape()[0] {
        for j in 0..3 {
            min_bound[j] = min_bound[j].min(vertices[[i, j]]);
            max_bound[j] = max_bound[j].max(vertices[[i, j]]);
        }
    }

    // Calculate the center of the bounding cube
    let cube_center = target_max_val / 2.0;

    // Calculate the current center of the scaled mesh
    let mut scaled_center = [0.0, 0.0, 0.0];
    for j in 0..3 {
        scaled_center[j] = (min_bound[j] + max_bound[j]) / 2.0;
    }

    // Calculate the shift needed to center the mesh in the cube
    let shift = [
        cube_center - scaled_center[0],
        cube_center - scaled_center[1],
        cube_center - scaled_center[2],
    ];
    info!("Shifting vertices to center in cube by: {:?}", shift);

    // Apply the shift to center the mesh in the cube
    for i in 0..vertices.shape()[0] {
        for j in 0..3 {
            vertices[[i, j]] += shift[j];
        }
    }

    // Verify bounds after centering
    min_bound = [f64::MAX, f64::MAX, f64::MAX];
    max_bound = [f64::MIN, f64::MIN, f64::MIN];
    for i in 0..vertices.shape()[0] {
        for j in 0..3 {
            min_bound[j] = min_bound[j].min(vertices[[i, j]]);
            max_bound[j] = max_bound[j].max(vertices[[i, j]]);
        }
    }

    // Check if any vertices are out of bounds and clamp if necessary
    let mut needs_clamping = false;
    for j in 0..3 {
        if min_bound[j] < 0.0 || max_bound[j] > target_max_val {
            needs_clamping = true;
            break;
        }
    }

    if needs_clamping {
        warn!("Mesh vertices out of bounds after centering. Clamping to valid range.");
        for i in 0..vertices.shape()[0] {
            for j in 0..3 {
                vertices[[i, j]] = vertices[[i, j]].max(0.0).min(target_max_val);
            }
        }
        // Recalculate final bounds after clamping
        min_bound = [f64::MAX, f64::MAX, f64::MAX];
        max_bound = [f64::MIN, f64::MIN, f64::MIN];
        for i in 0..vertices.shape()[0] {
            for j in 0..3 {
                min_bound[j] = min_bound[j].min(vertices[[i, j]]);
                max_bound[j] = max_bound[j].max(vertices[[i, j]]);
            }
        }
    }

    info!("Final BBox Min: {:?}, Max: {:?}", min_bound, max_bound);
    if needs_clamping {
        info!("Note: Mesh was clamped to stay within bounds");
    }

    vertices
}

/// Infers the projection mode based on the provided texture and material paths
///
/// # Arguments
/// * `texture_path` - Optional path to the texture image file
/// * `mtl_path` - Optional path to the MTL material file (or indication if OBJ links one)
///
/// # Returns
/// The inferred MeshMode
// This function's logic might need adjustment if `mtl_path` behavior changes
// (e.g. if it's just a hint and `tobj` handles the primary MTL loading)
fn infer_projection_mode(texture_path: Option<&str>, mtl_path: Option<&str>) -> MeshMode {
    match (texture_path, mtl_path) {
        (Some(tex_path), Some(mtl_p)) => {
            // mtl_p could be a specific file or just an indicator that an mtl *should* exist
            if Path::new(tex_path).exists() {
                // If an mtl path is hinted, assume Full if texture exists.
                // Actual mtl loading success is handled in load_mesh_with_mode
                if Path::new(mtl_p).exists() {
                    // If explicit path is valid
                    info!("Inferred FULL mode (OBJ+Texture+Explicit MTL at {})", mtl_p);
                    MeshMode::Full
                } else {
                    // If explicit path is NOT valid, or it was just a general hint from OBJ
                    info!("Inferred FULL mode (OBJ+Texture, MTL expected via OBJ or default). Explicit path {} not found or not primary.", mtl_p);
                    MeshMode::Full // Still assume Full, loader will try to find/default MTL
                }
            } else {
                info!("Inferred OBJ_ONLY mode (Texture at {} not found)", tex_path);
                MeshMode::ObjOnly
            }
        }
        (Some(tex_path), None) => {
            if Path::new(tex_path).exists() {
                info!("Inferred OBJ_TEXTURE mode (OBJ+Texture, no specific MTL path provided)");
                MeshMode::ObjTexture
            } else {
                info!("Inferred OBJ_ONLY mode (Texture at {} not found)", tex_path);
                MeshMode::ObjOnly
            }
        }
        (None, Some(mtl_p)) => {
            // No texture, but MTL path provided. This is likely OBJ_ONLY, but with specific material properties.
            // However, our modes are primarily texture-driven.
            info!("Inferred OBJ_ONLY mode (No texture, MTL path {} provided but unused without texture for Full/ObjTexture mode logic here)", mtl_p);
            MeshMode::ObjOnly
        }
        (None, None) => {
            info!("Inferred OBJ_ONLY mode (No texture, no MTL path)");
            MeshMode::ObjOnly
        }
    }
}

/// Saves projection data for a single view to specified files.
/// Assumes the input arrays are 4D with shape (1, rows, cols, channels)
/// or 3D with shape (1, rows, cols) for occupancy.
pub fn save_projection_data(
    image: &Array4<u8>,
    output_dir: &Path,
    view_index: usize,
    flip_vertical: bool,
) -> Result<(), String> {
    if image.shape()[0] != 1 {
        return Err("Input array for saving must have a batch size of 1.".to_string());
    }
    let rows = image.shape()[1];
    let cols = image.shape()[2];

    // Ensure output directory exists
    if let Err(e) = fs::create_dir_all(output_dir) {
        return Err(format!(
            "Failed to create output directory {:?}: {}",
            output_dir, e
        ));
    }

    // Save Image (RGB)
    if image.shape()[3] == 3 || image.shape()[3] == 4 {
        let image_slice = image.slice(s![0, .., .., ..]);

        let final_image_slice = if flip_vertical {
            // Manually create a vertically flipped buffer
            let mut flipped = Array3::<u8>::zeros(image_slice.raw_dim());
            for (i, row) in image_slice.outer_iter().rev().enumerate() {
                flipped.index_axis_mut(Axis(0), i).assign(&row);
            }
            flipped
        } else {
            image_slice.into_owned()
        };

        let buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(
            cols as u32,
            rows as u32,
            final_image_slice.into_raw_vec_and_offset().0,
        )
        .ok_or("Failed to create image buffer for image.")?;
        let image_path = output_dir.join(format!("image_{:04}.png", view_index));
        if let Err(e) = buffer.save(&image_path) {
            warn!("Failed to save image {:?}: {}", image_path, e);
        }
    } else {
        warn!(
            "Image array has unsupported number of channels: {}. Skipping save.",
            image.shape()[3]
        );
    }

    Ok(())
}
