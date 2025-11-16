//! Main mesh loading functionality orchestrating all specialized modules.

use crate::mesh_io::{
    errors::{MeshError, MeshResult},
    material::{get_submesh_material, process_materials_from_obj},
    preprocessing::{
        apply_transform_to_vertices, build_transform_from_bounds, compute_vertex_normals,
        normalize_normals, preprocess_vertices, PreprocessTransform,
    },
    utils::{
        convert_triangle_indices, convert_vertex_colors, convert_vertex_normals,
        convert_vertex_positions, create_global_triangle_indices, create_triangle_uvs,
        create_triangle_uvs_from_indices, load_texture, process_uv_coordinates,
        validate_file_exists,
    },
};
use crate::renderer::geometry::precompute_triangle_normals;
use crate::types::mesh::{Mesh, MeshGeometry, MeshMaterial, MeshTexture, Scene, ShadingMode};
use log::{info, warn};
use ndarray::Array3;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tobj;

/// Read a mesh from an OBJ, inferring materials/textures from the OBJ/MTL unless a mode is supplied
pub fn read_mesh_internal(
    obj_path: &str,
    mesh_mode: Option<ShadingMode>,
    resolution: u16,
    flip_v_coordinate: bool,
    ignore_opacity: bool,
) -> MeshResult<Scene> {
    // Validate input files
    validate_file_exists(obj_path)?;
    // Load OBJ file
    let (models, materials_result) = load_obj_file(obj_path)?;
    if models.is_empty() {
        return Err(MeshError::NoModelsFound);
    }
    // Decide mode if not provided
    let mut inferred_mode = ShadingMode::VertexColor;
    if let Some(mode) = mesh_mode {
        inferred_mode = mode;
    } else if let Ok(materials) = &materials_result {
        if !materials.is_empty() {
            inferred_mode = ShadingMode::BlinnPhong;
        }
    }
    info!("Reading mesh: {} in mode: {:?}", obj_path, inferred_mode);
    // Clone raw tobj materials (if available) for texture filename lookup without borrowing conflict
    let materials_raw_vec_opt: Option<Vec<tobj::Material>> =
        materials_result.as_ref().ok().cloned();
    // Process materials (for BlinnPhong/VertexColor/TextureMapped unified path)
    let materials = process_materials_from_obj(materials_result, inferred_mode, ignore_opacity)?;
    // Texture cache for deduplication and storage in Scene
    let mut texture_cache: HashMap<String, Arc<Array3<u8>>> = HashMap::new();
    let mut texture_images: Vec<Arc<Array3<u8>>> = Vec::new();
    // Load textures per-submesh when needed and build meshes
    let meshes = process_submeshes_with_textures(
        &models,
        &materials,
        inferred_mode,
        resolution,
        flip_v_coordinate,
        obj_path,
        materials_raw_vec_opt.as_deref(),
        &mut texture_cache,
        &mut texture_images,
    )?;
    if meshes.is_empty() {
        return Err(MeshError::NoProcessableSubmeshes);
    }
    let textures_vec = if texture_images.is_empty() {
        None
    } else {
        Some(texture_images)
    };
    Ok(Scene::new(meshes, inferred_mode, textures_vec))
}

/// Load OBJ file using tobj
fn load_obj_file(
    obj_path: &str,
) -> MeshResult<(
    Vec<tobj::Model>,
    Result<Vec<tobj::Material>, tobj::LoadError>,
)> {
    let load_options = tobj::GPU_LOAD_OPTIONS;
    tobj::load_obj(obj_path, &load_options).map_err(MeshError::from)
}

fn bounding_box(models: &[tobj::Model]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for model in models {
        for vertex in model.mesh.positions.chunks_exact(3) {
            for i in 0..3 {
                min[i] = min[i].min(vertex[i]);
                max[i] = max[i].max(vertex[i]);
            }
        }
    }
    (min, max)
}

/// Process all meshes from the loaded models
fn process_submeshes_with_textures(
    models: &[tobj::Model],
    materials: &[MeshMaterial],
    mode: ShadingMode,
    resolution: u16,
    flip_v_coordinate: bool,
    obj_path: &str,
    materials_raw_opt: Option<&[tobj::Material]>,
    texture_cache: &mut HashMap<String, Arc<Array3<u8>>>,
    texture_images: &mut Vec<Arc<Array3<u8>>>,
) -> MeshResult<Vec<Mesh>> {
    let mut meshes: Vec<Mesh> = Vec::new();
    let obj_dir = Path::new(obj_path).parent().map(Path::to_path_buf);
    // First pass: compute global bounds across all models
    let (global_min, global_max) = bounding_box(&models);
    // Build a single transform to be applied to every submesh
    let global_transform: PreprocessTransform =
        build_transform_from_bounds(global_min, global_max, resolution);
    // Second pass: process each model using the global transform
    for model in models {
        if let Some(submesh) = process_single_submesh_with_texture(
            model,
            materials,
            mode,
            resolution,
            flip_v_coordinate,
            obj_dir.as_ref(),
            materials_raw_opt,
            Some(&global_transform),
            texture_cache,
            texture_images,
        )? {
            meshes.push(submesh);
        }
    }
    Ok(meshes)
}

/// Process a single model into a submesh
fn process_single_submesh_with_texture(
    model: &tobj::Model,
    materials: &[MeshMaterial],
    mode: ShadingMode,
    resolution: u16,
    flip_v_coordinate: bool,
    obj_dir: Option<&PathBuf>,
    materials_raw_opt: Option<&[tobj::Material]>,
    global_transform_opt: Option<&PreprocessTransform>,
    texture_cache: &mut HashMap<String, Arc<Array3<u8>>>,
    texture_images: &mut Vec<Arc<Array3<u8>>>,
) -> MeshResult<Option<Mesh>> {
    let mesh = &model.mesh;
    if mesh.positions.is_empty() {
        warn!(
            "Skipping model '{}' as it has no vertex positions.",
            model.name
        );
        return Ok(None);
    }
    let num_vertices = mesh.positions.len() / 3;
    let vertices = convert_vertex_positions(&mesh.positions)?;
    let triangles = convert_triangle_indices(&mesh.indices)?;
    let vertex_normals = process_vertex_normals(mesh, num_vertices, &vertices, &triangles)?;
    let vertex_colors = process_vertex_colors(mesh, num_vertices, &model.name);
    let processed_vertices = if let Some(t) = global_transform_opt {
        let mut v = vertices;
        apply_transform_to_vertices(&mut v, t);
        v
    } else {
        preprocess_vertices(vertices, resolution)
    };
    let face_normals = precompute_triangle_normals(processed_vertices.view(), triangles.view());
    let geometry = MeshGeometry {
        vertices: processed_vertices,
        triangles,
        vertex_normals,
        face_normals,
        vertex_colors,
    };
    let material = get_submesh_material(mesh.material_id, materials, mode, &model.name);
    let mut mesh_texture_opt: Option<MeshTexture> = None;
    if matches!(mode, ShadingMode::BlinnPhong | ShadingMode::TextureMapped) {
        if let Some(tex) = load_texture_for_model(
            model,
            flip_v_coordinate,
            obj_dir,
            materials_raw_opt,
            texture_cache,
            texture_images,
        )? {
            mesh_texture_opt = Some(tex);
        }
    }
    let mesh = Mesh {
        geometry,
        material,
        texture: mesh_texture_opt,
    };
    Ok(Some(mesh))
}

/// Process vertex normals for a model
fn process_vertex_normals(
    mesh: &tobj::Mesh,
    num_vertices: usize,
    vertices: &ndarray::Array2<f32>,
    triangles: &ndarray::Array2<usize>,
) -> MeshResult<Option<ndarray::Array2<f32>>> {
    if !mesh.normals.is_empty() && mesh.normals.len() == num_vertices * 3 {
        info!("Using vertex normals from OBJ data.");
        match convert_vertex_normals(&mesh.normals, num_vertices) {
            Ok(mut normals_arr) => {
                normalize_normals(&mut normals_arr);
                Ok(Some(normals_arr))
            }
            Err(e) => {
                warn!(
                    "Failed to convert vertex normals: {}. Computing them instead.",
                    e
                );
                match compute_vertex_normals(vertices, triangles) {
                    Some(normals) => Ok(Some(normals)),
                    None => Err(MeshError::GeometryProcessingError(
                        "Failed to compute vertex normals from geometry".to_string(),
                    )),
                }
            }
        }
    } else {
        if !mesh.normals.is_empty() {
            warn!("Vertex normal data was inconsistent or incomplete. Recomputing normals.");
        } else {
            info!("Model does not have vertex normals. Computing them.");
        }
        match compute_vertex_normals(vertices, triangles) {
            Some(normals) => Ok(Some(normals)),
            None => Err(MeshError::GeometryProcessingError(
                "Failed to compute vertex normals from geometry".to_string(),
            )),
        }
    }
}

/// Process vertex colors for a model
fn process_vertex_colors(
    mesh: &tobj::Mesh,
    num_vertices: usize,
    model_name: &str,
) -> Option<ndarray::Array2<f32>> {
    if !mesh.vertex_color.is_empty() && mesh.vertex_color.len() == num_vertices * 3 {
        info!(
            "Using vertex colors from OBJ data for model '{}'.",
            model_name
        );
        match convert_vertex_colors(&mesh.vertex_color, num_vertices) {
            Ok(colors_arr) => Some(colors_arr),
            Err(e) => {
                warn!(
                    "Failed to convert vertex colors for model '{}': {}. Skipping vertex colors.",
                    model_name, e
                );
                None
            }
        }
    } else {
        if !mesh.vertex_color.is_empty() {
            warn!(
                "Vertex color data for model '{}' was present but inconsistent (expected {}*3, got {}). Not using vertex colors.",
                model_name, num_vertices, mesh.vertex_color.len()
            );
        }
        None
    }
}

/// Attempt to load a texture for a model by inspecting its material for a diffuse/albedo texture.
/// Returns a MeshTexture with per-triangle UVs for this model if successful, otherwise None.
fn load_texture_for_model(
    model: &tobj::Model,
    flip_v_coordinate: bool,
    obj_dir: Option<&PathBuf>,
    materials_raw_opt: Option<&[tobj::Material]>,
    texture_cache: &mut HashMap<String, Arc<Array3<u8>>>,
    texture_images: &mut Vec<Arc<Array3<u8>>>,
) -> MeshResult<Option<MeshTexture>> {
    // Fetch the texture path from the linked tobj material if available via material_id
    let tex_path_opt: Option<String> = match model.mesh.material_id {
        Some(idx) => materials_raw_opt.and_then(|materials_raw| {
            materials_raw.get(idx).and_then(|mtl| {
                mtl.diffuse_texture
                    .as_ref()
                    .cloned()
                    .or_else(|| mtl.ambient_texture.as_ref().cloned())
            })
        }),
        None => {
            // Fallback: Some exporters omit `usemtl`, leaving material_id unset even though an MTL exists.
            // In that case, try the first available material to discover a texture path.
            let fallback = materials_raw_opt.and_then(|materials_raw| {
                materials_raw.get(0).and_then(|mtl| {
                    mtl.diffuse_texture
                        .as_ref()
                        .cloned()
                        .or_else(|| mtl.ambient_texture.as_ref().cloned())
                })
            });
            if fallback.is_some() {
                warn!(
                    "Model '{}' has no material_id; using first material's texture as fallback.",
                    model.name
                );
            }
            fallback
        }
    };
    let tex_path_str = match tex_path_opt {
        Some(p) if !p.is_empty() => p,
        _ => return Ok(None),
    };
    // Resolve texture path relative to OBJ directory if needed
    let resolved_path = if Path::new(&tex_path_str).is_absolute() {
        PathBuf::from(&tex_path_str)
    } else if let Some(dir) = obj_dir {
        dir.join(&tex_path_str)
    } else {
        PathBuf::from(&tex_path_str)
    };
    // Load or fetch shared texture reference (per-scene cache)
    let (texture_ref, tex_height, tex_width, tex_channels) =
        match get_or_load_shared_texture_with_cache(&resolved_path, texture_cache, texture_images) {
            Ok((tex_ref, h, w, c)) => (tex_ref, h, w, c),
            Err(e) => {
                warn!(
                    "Failed to load texture for model '{}': {}. Proceeding without texture.",
                    model.name, e
                );
                return Ok(None);
            }
        };
    info!(
        "Loaded texture for model '{}': {} ({}x{}x{})",
        model.name,
        resolved_path.display(),
        tex_height,
        tex_width,
        tex_channels
    );
    // Compute triangle UVs for this model only
    let triangle_uvs = if !model.mesh.texcoord_indices.is_empty() {
        create_triangle_uvs_from_indices(std::slice::from_ref(model), flip_v_coordinate)?
    } else {
        // Legacy single-index path: create from vertex UVs and triangle indices
        let vertex_count = model.mesh.positions.len() / 3;
        if model.mesh.texcoords.len() < vertex_count * 2 {
            warn!(
                "Model '{}' has insufficient UV data; defaulting to zeros.",
                model.name
            );
            let num_corners = model.mesh.indices.len();
            let uv = ndarray::Array2::<f32>::zeros((num_corners, 2));
            uv
        } else {
            let vertex_uvs = process_uv_coordinates(
                &model.mesh.texcoords[..vertex_count * 2],
                vertex_count,
                flip_v_coordinate,
            )?;
            let triangle_indices = create_global_triangle_indices(std::slice::from_ref(model));
            create_triangle_uvs(&vertex_uvs, &triangle_indices)?
        }
    };
    Ok(Some(MeshTexture {
        texture: Some(texture_ref),
        triangle_uvs: Some(triangle_uvs),
        channels: tex_channels,
        height_f32: tex_height as f32,
        width_f32: tex_width as f32,
    }))
}

fn get_or_load_shared_texture_with_cache(
    path: &PathBuf,
    texture_cache: &mut HashMap<String, Arc<Array3<u8>>>,
    texture_images: &mut Vec<Arc<Array3<u8>>>,
) -> MeshResult<(Arc<Array3<u8>>, usize, usize, usize)> {
    let key = path.to_string_lossy().to_string();
    if let Some(tex_ref) = texture_cache.get(&key) {
        let (h, w, c) = (tex_ref.shape()[0], tex_ref.shape()[1], tex_ref.shape()[2]);
        return Ok((tex_ref.clone(), h, w, c));
    }
    let img = load_texture(&key)?;
    let (h, w, c) = (img.shape()[0], img.shape()[1], img.shape()[2]);
    let arc_img = Arc::new(img);
    texture_cache.insert(key, arc_img.clone());
    texture_images.push(arc_img.clone());
    Ok((arc_img, h, w, c))
}
