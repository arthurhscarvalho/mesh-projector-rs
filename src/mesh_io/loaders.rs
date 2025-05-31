//! Main mesh loading functionality orchestrating all specialized modules.

use crate::mesh_io::{
    errors::{MeshError, MeshResult},
    material::{get_submesh_material, process_materials_from_obj},
    preprocessing::{compute_vertex_normals, normalize_normals, preprocess_vertices},
    utils::{
        aggregate_model_data, convert_triangle_indices, convert_vertex_colors,
        convert_vertex_normals, convert_vertex_positions, create_global_triangle_indices,
        create_triangle_uvs, create_triangle_uvs_from_indices, has_separate_texcoord_indices,
        load_texture, process_uv_coordinates, validate_file_exists,
    },
};
use crate::projector::geometry::precompute_triangle_normals;
use crate::projector::mesh::{Mesh, MeshGeometry, MeshMode, MeshTexture, SubMesh};
use log::{info, warn};
use tobj;

/// Load a mesh using a specific projection mode
pub fn load_mesh_with_mode(
    obj_path: &str,
    texture_path: Option<&str>,
    mtl_path: Option<&str>,
    mode: MeshMode,
    precision: u32,
    flip_v_coordinate: bool,
    ignore_opacity: bool,
) -> MeshResult<Mesh> {
    info!("Loading mesh: {} in mode: {:?}", obj_path, mode);

    // Validate precision (This limit is set just to set consistent bounds)
    if precision < 8 || precision > 16 {
        return Err(MeshError::PrecisionError(format!(
            "Invalid precision value: {}. Must be between 8 and 16.",
            precision
        )));
    }

    // Validate input files
    validate_file_exists(obj_path)?;

    // Load OBJ file
    let (models, materials_result) = load_obj_file(obj_path)?;

    if models.is_empty() {
        return Err(MeshError::NoModelsFound);
    }

    // Process materials
    let materials = process_materials_from_obj(materials_result, mtl_path, mode, ignore_opacity)?;

    // Process texture if needed
    let shared_texture = process_texture_data(&models, texture_path, mode, flip_v_coordinate)?;

    // Process submeshes
    let submeshes = process_submeshes(&models, &materials, mode, precision)?;

    if submeshes.is_empty() {
        return Err(MeshError::NoProcessableSubmeshes);
    }

    Ok(Mesh {
        submeshes,
        texture: shared_texture,
        mode,
    })
}

/// Load OBJ file using tobj
fn load_obj_file(
    obj_path: &str,
) -> MeshResult<(
    Vec<tobj::Model>,
    Result<Vec<tobj::Material>, tobj::LoadError>,
)> {
    let load_options = tobj::LoadOptions {
        triangulate: true,
        single_index: false,
        ..Default::default()
    };

    tobj::load_obj(obj_path, &load_options).map_err(MeshError::from)
}

/// Process texture data for shared texture modes
fn process_texture_data(
    models: &[tobj::Model],
    texture_path: Option<&str>,
    mode: MeshMode,
    flip_v_coordinate: bool,
) -> MeshResult<MeshTexture> {
    match mode {
        MeshMode::Full | MeshMode::ObjTexture => {
            process_texture_for_mode(models, texture_path, flip_v_coordinate)
        }
        MeshMode::ObjOnly => {
            // No texture needed for ObjOnly mode
            Ok(create_empty_texture())
        }
    }
}

/// Process texture for modes that require it
fn process_texture_for_mode(
    models: &[tobj::Model],
    texture_path: Option<&str>,
    flip_v_coordinate: bool,
) -> MeshResult<MeshTexture> {
    let tex_path = texture_path.ok_or_else(|| {
        MeshError::MissingRequiredData("Texture file is required but none was provided".to_string())
    })?;

    let texture_data = load_texture(tex_path)?;
    info!(
        "Shared texture load successful: {} ({}x{}x{})",
        tex_path,
        texture_data.shape()[0],
        texture_data.shape()[1],
        texture_data.shape()[2]
    );

    // Check if we have separate texture coordinate indices (single_index: false)
    let triangle_uvs = if has_separate_texcoord_indices(models) {
        info!("Using separate texture coordinate indices from OBJ models for shared texture.");
        create_triangle_uvs_from_indices(models, flip_v_coordinate)?
    } else {
        info!("Using vertex-based UV coordinates from all OBJ models for shared texture.");
        let (_, all_texcoords, total_vertex_count) = aggregate_model_data(models);

        if all_texcoords.is_empty() || all_texcoords.len() != total_vertex_count * 2 {
            return Err(MeshError::InvalidUvData(
                "Mesh has no usable UV coordinates for shared texture".to_string(),
            ));
        }

        let vertex_uvs =
            process_uv_coordinates(&all_texcoords, total_vertex_count, flip_v_coordinate)?;
        let global_triangle_indices = create_global_triangle_indices(models);
        create_triangle_uvs(&vertex_uvs, &global_triangle_indices)?
    };

    Ok(MeshTexture {
        texture: Some(texture_data.clone()),
        triangle_uvs: Some(triangle_uvs),
        channels: texture_data.shape()[2],
        height_f64: texture_data.shape()[0] as f64,
        width_f64: texture_data.shape()[1] as f64,
    })
}

/// Create empty texture for modes that don't need it
fn create_empty_texture() -> MeshTexture {
    MeshTexture {
        texture: None,
        triangle_uvs: None,
        channels: 0,
        height_f64: 0.0,
        width_f64: 0.0,
    }
}

/// Process all submeshes from the loaded models
fn process_submeshes(
    models: &[tobj::Model],
    materials: &[crate::projector::mesh::MeshMaterial],
    mode: MeshMode,
    precision: u32,
) -> MeshResult<Vec<SubMesh>> {
    let mut submeshes = Vec::new();

    for model in models {
        if let Some(submesh) = process_single_submesh(model, materials, mode, precision)? {
            submeshes.push(submesh);
        }
    }

    Ok(submeshes)
}

/// Process a single model into a submesh
fn process_single_submesh(
    model: &tobj::Model,
    materials: &[crate::projector::mesh::MeshMaterial],
    mode: MeshMode,
    precision: u32,
) -> MeshResult<Option<SubMesh>> {
    let mesh = &model.mesh;

    if mesh.positions.is_empty() {
        warn!(
            "Skipping model '{}' as it has no vertex positions.",
            model.name
        );
        return Ok(None);
    }

    let num_vertices = mesh.positions.len() / 3;

    // Convert geometry data
    let vertices = convert_vertex_positions(&mesh.positions)?;
    let triangles = convert_triangle_indices(&mesh.indices)?;

    // Process normals
    let vertex_normals = process_vertex_normals(mesh, num_vertices, &vertices, &triangles)?;

    // Process vertex colors if available
    let vertex_colors = process_vertex_colors(mesh, num_vertices, &model.name);

    // Preprocess vertices
    let processed_vertices = preprocess_vertices(vertices, precision);

    // Compute face normals
    let face_normals = precompute_triangle_normals(processed_vertices.view(), triangles.view());

    // Create geometry
    let geometry = MeshGeometry {
        vertices: processed_vertices,
        triangles,
        vertex_normals,
        face_normals,
        vertex_colors,
    };

    // Get material for this submesh
    let material = get_submesh_material(mesh.material_id, materials, mode, &model.name);

    Ok(Some(SubMesh { geometry, material }))
}

/// Process vertex normals for a model
fn process_vertex_normals(
    mesh: &tobj::Mesh,
    num_vertices: usize,
    vertices: &ndarray::Array2<f64>,
    triangles: &ndarray::Array2<usize>,
) -> MeshResult<Option<ndarray::Array2<f64>>> {
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
) -> Option<ndarray::Array2<f64>> {
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
