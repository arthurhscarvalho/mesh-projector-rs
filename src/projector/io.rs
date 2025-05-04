// src/projector/io.rs
//! Module for loading and preprocessing 3D mesh data.
use crate::projector::mesh::{Mesh, MeshGeometry, MeshMaterial, MeshMode, MeshTexture};
use log::{info, warn};
use ndarray::{Array, Array2, Array3};
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
/// * `mtl_path` - Optional path to the MTL material file
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
    mtl_path: Option<&str>,
    mode: MeshMode,
    precision: u32,
    flip_v_coordinate: bool,
    ignore_opacity: bool,
) -> Result<Mesh, String> {
    info!("Loading mesh: {} in mode: {:?}", obj_path, mode);
    // Check if the OBJ file exists
    if !Path::new(obj_path).exists() {
        return Err(format!("OBJ file not found: {}", obj_path));
    }
    // Load OBJ file
    let obj_result = tobj::load_obj(
        obj_path,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    );
    let models = match obj_result {
        Ok((models, _materials)) => models,
        Err(e) => return Err(format!("Failed to load OBJ file: {}", e)),
    };
    if models.is_empty() {
        return Err("No models found in the OBJ file".to_string());
    }
    // We'll work with the first model for simplicity
    if models.len() > 1 {
        warn!("Multiple models found in the OBJ file. Using the first one.");
    }
    let model = &models[0];
    let mesh = &model.mesh;
    // Check if the model has vertices
    if mesh.positions.is_empty() {
        return Err("Mesh has no vertices".to_string());
    }
    // Convert to ndarray format
    let mut vertices = Array::zeros((mesh.positions.len() / 3, 3));
    for i in 0..vertices.shape()[0] {
        for j in 0..3 {
            vertices[[i, j]] = mesh.positions[i * 3 + j] as f64;
        }
    }
    let mut triangles = Array::zeros((mesh.indices.len() / 3, 3));
    for i in 0..triangles.shape()[0] {
        for j in 0..3 {
            triangles[[i, j]] = mesh.indices[i * 3 + j] as usize;
        }
    }
    // Load normals if available
    let vertex_normals = if !mesh.normals.is_empty() {
        info!("Mesh has vertex normals");
        let mut normals = Array::zeros((mesh.normals.len() / 3, 3));
        for i in 0..normals.shape()[0] {
            for j in 0..3 {
                normals[[i, j]] = mesh.normals[i * 3 + j] as f64;
            }
        }
        // Normalize normals
        normalize_normals(&mut normals);
        Some(normals)
    } else {
        // If no normals in the OBJ, compute them
        warn!("Mesh does not have vertex normals. Computing them.");
        compute_vertex_normals(&vertices, &triangles)
    };
    // Extract vertex colors if available
    let vertex_colors = if !mesh.vertex_color.is_empty() {
        info!("Mesh has vertex colors");
        let mut colors = Array::zeros((mesh.vertex_color.len() / 3, 3));
        for i in 0..colors.shape()[0] {
            for j in 0..3 {
                colors[[i, j]] = mesh.vertex_color[i * 3 + j] as f64;
            }
        }
        Some(colors)
    } else if mode == MeshMode::ObjOnly {
        info!("No vertex colors found. Using defaults for OBJ_ONLY mode.");
        Some(Array::zeros((vertices.shape()[0], 3)))
    } else {
        None
    };
    // Load texture and UVs if required by the mode
    let (texture, triangle_uvs) = if mode == MeshMode::Full || mode == MeshMode::ObjTexture {
        // Texture is required for these modes
        if texture_path.is_none() {
            return Err(format!(
                "Texture file is required for mode {:?} but none was provided",
                mode
            ));
        }
        // UVs are required for these modes
        if mesh.texcoords.is_empty() {
            return Err(format!(
                "Mesh has no UV coordinates, required for mode {:?}",
                mode
            ));
        }
        // Load texture
        let texture_path = texture_path.unwrap();
        let texture_array = load_texture(texture_path)?;
        info!(
            "Texture load successful: {} ({}x{}x{})",
            texture_path,
            texture_array.shape()[0],
            texture_array.shape()[1],
            texture_array.shape()[2]
        );
        // Process UVs
        let mut uvs = Array::zeros((mesh.texcoords.len() / 2, 2));
        for i in 0..uvs.shape()[0] {
            for j in 0..2 {
                let mut value = mesh.texcoords[i * 2 + j] as f64;
                // Clamp UVs to [0, 1]
                value = value.max(0.0).min(1.0);
                // Flip V coordinate if requested
                if j == 1 && flip_v_coordinate {
                    value = 1.0 - value;
                }
                uvs[[i, j]] = value;
            }
        }
        // Map UVs to triangles if needed
        let triangle_uvs = if mesh.texcoord_indices.is_empty() {
            // If there are no explicit texcoord indices, assume they match vertex indices
            warn!("No explicit texcoord indices found, using vertex indices");
            let mut tri_uvs = Array::zeros((mesh.indices.len(), 2));
            for i in 0..mesh.indices.len() {
                let idx = mesh.indices[i] as usize;
                if idx < uvs.shape()[0] {
                    tri_uvs[[i, 0]] = uvs[[idx, 0]];
                    tri_uvs[[i, 1]] = uvs[[idx, 1]];
                }
            }
            tri_uvs
        } else {
            // Use explicit texcoord indices
            let mut tri_uvs = Array::zeros((mesh.texcoord_indices.len(), 2));
            for i in 0..mesh.texcoord_indices.len() {
                let idx = mesh.texcoord_indices[i] as usize;
                if idx < uvs.shape()[0] {
                    tri_uvs[[i, 0]] = uvs[[idx, 0]];
                    tri_uvs[[i, 1]] = uvs[[idx, 1]];
                }
            }
            tri_uvs
        };
        (Some(texture_array), Some(triangle_uvs))
    } else {
        // For OBJ_ONLY mode, we don't need texture or UVs
        (None, None)
    };
    // Load material properties based on mode
    let material = if mode == MeshMode::Full {
        if let Some(mtl_path) = mtl_path {
            // Load MTL separately from the specified path
            info!("Loading MTL from explicit path: {}", mtl_path);
            load_material_from_mtl(mtl_path, ignore_opacity)
        } else {
            warn!("No MTL file provided for FULL mode. Using default material.");
            DEFAULT_MATERIAL
        }
    } else if mode == MeshMode::ObjTexture {
        // Default material for texturing
        MeshMaterial {
            ka: [0.0, 0.0, 0.0], // No Ambient
            kd: [1.0, 1.0, 1.0], // Base color modulation (texture is base)
            ks: [0.0, 0.0, 0.0], // No Specular
            ns: 10.0,            // Default Shininess
            opacity: 1.0,        // Fully Opaque
            illum: 0,            // Color = ShadingColor (no lighting calc)
        }
    } else {
        // OBJ_ONLY mode
        MeshMaterial {
            ka: [0.0, 0.0, 0.0], // No Ambient term from material
            kd: [1.0, 1.0, 1.0], // Base color modulation (vertex color is base)
            ks: [0.0, 0.0, 0.0], // No Specular
            ns: 10.0,
            opacity: 1.0,
            illum: 0, // Color = ShadingColor (no lighting calculation)
        }
    };
    // Preprocess the mesh (scale vertices to fit in the target precision)
    let vertices = preprocess_vertices(vertices, precision);

    // Get texture dimensions if texture exists
    let (texture_channels, texture_height, texture_width) = if let Some(ref tex) = texture {
        (tex.shape()[2], tex.shape()[0] as f64, tex.shape()[1] as f64)
    } else {
        (3, 0.0, 0.0)
    };

    // Create the mesh structure with the new structure
    let mesh = Mesh {
        geometry: MeshGeometry {
            vertices,
            triangles,
            vertex_normals,
        },
        texture: MeshTexture {
            texture,
            vertex_colors,
            triangle_uvs,
            channels: texture_channels,
            height_f64: texture_height,
            width_f64: texture_width,
        },
        material: MeshMaterial {
            ka: material.ka,
            kd: material.kd,
            ks: material.ks,
            ns: material.ns,
            opacity: material.opacity,
            illum: material.illum,
        },
        mode, // Store the inferred/provided mode
    };
    Ok(mesh)
}

/// Loads material properties from an MTL file
fn load_material_from_mtl(mtl_path: &str, ignore_opacity: bool) -> MeshMaterial {
    if !Path::new(mtl_path).exists() {
        warn!(
            "MTL file not found: {}. Using default material values.",
            mtl_path
        );
        return DEFAULT_MATERIAL;
    }
    let mut material = MeshMaterial {
        ka: [0.1, 0.1, 0.1],
        kd: [0.8, 0.8, 0.8],
        ks: [0.1, 0.1, 0.1],
        ns: 10.0,
        opacity: 1.0,
        illum: 2,
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

/// Preprocess vertices by scaling them to fit within the target precision
fn preprocess_vertices(mut vertices: Array2<f64>, precision: u32) -> Array2<f64> {
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
    // Shift vertices if any minimum coordinate is negative
    let shift = [
        min_bound[0].min(0.0).abs(),
        min_bound[1].min(0.0).abs(),
        min_bound[2].min(0.0).abs(),
    ];
    if shift[0] > 0.0 || shift[1] > 0.0 || shift[2] > 0.0 {
        info!("Shifting vertices by: {:?}", shift);
        for i in 0..vertices.shape()[0] {
            for j in 0..3 {
                vertices[[i, j]] += shift[j];
            }
        }
        // Update bounds
        for j in 0..3 {
            min_bound[j] += shift[j];
            max_bound[j] += shift[j];
        }
        info!("Shifted BBox Min: {:?}, Max: {:?}", min_bound, max_bound);
    }
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
    let scale_factor = target_max_val / max_dim;
    info!(
        "Scaling vertices. Max dimension: {}, Scale factor: {}",
        max_dim, scale_factor
    );
    // Scale vertices
    for i in 0..vertices.shape()[0] {
        for j in 0..3 {
            vertices[[i, j]] *= scale_factor;
        }
    }
    // Calculate new bounds for verification
    min_bound = [f64::MAX, f64::MAX, f64::MAX];
    max_bound = [f64::MIN, f64::MIN, f64::MIN];
    for i in 0..vertices.shape()[0] {
        for j in 0..3 {
            min_bound[j] = min_bound[j].min(vertices[[i, j]]);
            max_bound[j] = max_bound[j].max(vertices[[i, j]]);
        }
    }
    info!("Scaled BBox Min: {:?}, Max: {:?}", min_bound, max_bound);
    vertices
}

/// Infers the projection mode based on the provided texture and material paths
///
/// # Arguments
/// * `texture_path` - Optional path to the texture image file
/// * `mtl_path` - Optional path to the MTL material file
///
/// # Returns
/// The inferred MeshMode
fn infer_projection_mode(texture_path: Option<&str>, mtl_path: Option<&str>) -> MeshMode {
    match (texture_path, mtl_path) {
        (Some(tex_path), Some(mtl_path)) => {
            // Both texture and MTL provided
            if Path::new(tex_path).exists() && Path::new(mtl_path).exists() {
                info!("Inferred FULL mode (OBJ+Texture+MTL)");
                MeshMode::Full
            } else if Path::new(tex_path).exists() {
                // Texture exists but MTL doesn't
                info!("Inferred OBJ_TEXTURE mode (OBJ+Texture, MTL not found)");
                MeshMode::ObjTexture
            } else {
                // Texture doesn't exist
                info!("Inferred OBJ_ONLY mode (Texture not found)");
                MeshMode::ObjOnly
            }
        }
        (Some(tex_path), None) => {
            // Texture provided but no MTL
            if Path::new(tex_path).exists() {
                info!("Inferred OBJ_TEXTURE mode (OBJ+Texture)");
                MeshMode::ObjTexture
            } else {
                info!("Inferred OBJ_ONLY mode (Texture not found)");
                MeshMode::ObjOnly
            }
        }
        (None, _) => {
            // No texture, so OBJ only
            info!("Inferred OBJ_ONLY mode (No texture provided)");
            MeshMode::ObjOnly
        }
    }
}
