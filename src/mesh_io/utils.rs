//! Utility functions for mesh I/O operations.

use crate::mesh_io::errors::{MeshError, MeshResult};
use ndarray::{Array2, Array3};
use std::path::Path;

/// Load a texture image and convert it to an RGB ndarray
pub fn load_texture(texture_path: &str) -> MeshResult<Array3<u8>> {
    validate_texture_path(texture_path)?;

    let img = image::open(texture_path).map_err(|e| {
        MeshError::TextureProcessingError(format!(
            "Failed to open texture file '{}': {}",
            texture_path, e
        ))
    })?;

    // Convert to RGB8 and obtain raw vec without an explicit per-pixel loop
    let rgb_img = img.into_rgb8();
    let (width, height) = rgb_img.dimensions();

    if width == 0 || height == 0 {
        return Err(MeshError::TextureProcessingError(format!(
            "Invalid texture dimensions: {}x{} for file '{}'",
            width, height, texture_path
        )));
    }

    let raw_pixels = rgb_img.into_raw(); // Vec<u8> in RGBRGB... layout

    // SAFETY: width * height * 3 is guaranteed by the image crate
    Array3::from_shape_vec((height as usize, width as usize, 3), raw_pixels).map_err(|e| {
        MeshError::TextureProcessingError(format!(
            "Failed to convert texture buffer into ndarray: {}",
            e
        ))
    })
}

/// Process UV coordinates from raw vertex texture coordinates
pub fn process_uv_coordinates(
    vertex_texcoords: &[f32],
    vertex_count: usize,
    flip_v_coordinate: bool,
) -> MeshResult<Array2<f64>> {
    if vertex_texcoords.len() != vertex_count * 2 {
        return Err(MeshError::InvalidUvData(format!(
            "UV coordinates length {} doesn't match vertex count {} * 2",
            vertex_texcoords.len(),
            vertex_count
        )));
    }

    // Allocate the full buffer up-front and fill it using a single pass iterator
    let mut uvs_vec: Vec<f64> = Vec::with_capacity(vertex_texcoords.len());

    for chunk in vertex_texcoords.chunks_exact(2) {
        // Safe because chunks_exact(2) guarantees size == 2
        let u = chunk[0].clamp(0.0, 1.0) as f64;
        let mut v = chunk[1].clamp(0.0, 1.0) as f64;

        if flip_v_coordinate {
            v = 1.0 - v;
        }

        uvs_vec.push(u);
        uvs_vec.push(v);
    }

    // SAFETY: we validated the length above and preallocated exactly len == vertex_count * 2
    Array2::from_shape_vec((vertex_count, 2), uvs_vec)
        .map_err(|e| MeshError::InvalidUvData(format!("Failed to convert UV coordinates: {}", e)))
}

/// Convert vertex positions from f32 to f64 array format
pub fn convert_vertex_positions(positions: &[f32]) -> MeshResult<Array2<f64>> {
    if positions.len() % 3 != 0 {
        return Err(MeshError::InvalidVertexData(
            "Position array length must be divisible by 3".to_string(),
        ));
    }

    let vertex_count = positions.len() / 3;

    // Convert and collect f32 -> f64 in a single pass
    let vertices_vec: Vec<f64> = positions.iter().map(|&v| v as f64).collect();

    Array2::from_shape_vec((vertex_count, 3), vertices_vec).map_err(|e| {
        MeshError::InvalidVertexData(format!("Failed to convert vertex positions: {}", e))
    })
}

/// Convert triangle indices from u32 to usize array format
pub fn convert_triangle_indices(indices: &[u32]) -> MeshResult<Array2<usize>> {
    if indices.len() % 3 != 0 {
        return Err(MeshError::InvalidTriangleData(
            "Index array length must be divisible by 3".to_string(),
        ));
    }

    let triangle_count = indices.len() / 3;

    let triangles_vec: Vec<usize> = indices.iter().map(|&v| v as usize).collect();

    Array2::from_shape_vec((triangle_count, 3), triangles_vec).map_err(|e| {
        MeshError::InvalidTriangleData(format!("Failed to convert triangle indices: {}", e))
    })
}

/// Convert vertex normals from f32 to f64 array format
pub fn convert_vertex_normals(normals: &[f32], vertex_count: usize) -> MeshResult<Array2<f64>> {
    if normals.len() != vertex_count * 3 {
        return Err(MeshError::InvalidVertexData(format!(
            "Normal array length {} doesn't match vertex count {} * 3",
            normals.len(),
            vertex_count
        )));
    }

    let normals_vec: Vec<f64> = normals.iter().map(|&v| v as f64).collect();

    Array2::from_shape_vec((vertex_count, 3), normals_vec).map_err(|e| {
        MeshError::InvalidVertexData(format!("Failed to convert vertex normals: {}", e))
    })
}

/// Convert vertex colors from f32 to f64 array format
pub fn convert_vertex_colors(colors: &[f32], vertex_count: usize) -> MeshResult<Array2<f64>> {
    if colors.len() != vertex_count * 3 {
        return Err(MeshError::InvalidVertexData(format!(
            "Color array length {} doesn't match vertex count {} * 3",
            colors.len(),
            vertex_count
        )));
    }

    let colors_vec: Vec<f64> = colors.iter().map(|&v| v as f64).collect();

    Array2::from_shape_vec((vertex_count, 3), colors_vec).map_err(|e| {
        MeshError::InvalidVertexData(format!("Failed to convert vertex colors: {}", e))
    })
}

/// Create triangle UV coordinates directly from texture coordinate indices
/// This is used when single_index is false and we have separate texcoord indices
pub fn create_triangle_uvs_from_indices(
    models: &[tobj::Model],
    flip_v_coordinate: bool,
) -> MeshResult<Array2<f64>> {
    // Pre-allocate the buffer to avoid frequent reallocations. Each triangle corner contributes
    // exactly 2 floating-point values (U and V).
    let estimated_corners: usize = models.iter().map(|m| m.mesh.indices.len()).sum();
    let mut all_triangle_uvs = Vec::with_capacity(estimated_corners * 2);

    for model in models {
        let mesh = &model.mesh;

        if mesh.texcoord_indices.is_empty() || mesh.texcoords.is_empty() {
            // Fallback for models without proper texture coordinate indices
            let triangle_count = mesh.indices.len() / 3;
            for _ in 0..triangle_count {
                for _ in 0..3 {
                    all_triangle_uvs.push(0.0); // U
                    all_triangle_uvs.push(0.0); // V
                }
            }
            continue;
        }

        if mesh.texcoord_indices.len() != mesh.indices.len() {
            return Err(MeshError::InvalidUvData(format!(
                "Texcoord indices length ({}) doesn't match position indices length ({}) for model '{}'",
                mesh.texcoord_indices.len(),
                mesh.indices.len(),
                model.name
            )));
        }

        // Process each triangle corner
        for &texcoord_idx in &mesh.texcoord_indices {
            let idx = texcoord_idx as usize;
            if idx * 2 + 1 < mesh.texcoords.len() {
                let mut u = mesh.texcoords[idx * 2] as f64;
                let mut v = mesh.texcoords[idx * 2 + 1] as f64;

                // Clamp UV coordinates to valid range
                u = u.clamp(0.0, 1.0);
                v = v.clamp(0.0, 1.0);

                if flip_v_coordinate {
                    v = 1.0 - v;
                }

                all_triangle_uvs.push(u);
                all_triangle_uvs.push(v);
            } else {
                // Use default UV coordinates if index is out of bounds
                all_triangle_uvs.push(0.0);
                all_triangle_uvs.push(0.0);
            }
        }
    }

    let num_corners = all_triangle_uvs.len() / 2;
    let mut triangle_uvs = Array2::zeros((num_corners, 2));

    for i in 0..num_corners {
        triangle_uvs[[i, 0]] = all_triangle_uvs[i * 2];
        triangle_uvs[[i, 1]] = all_triangle_uvs[i * 2 + 1];
    }

    Ok(triangle_uvs)
}

/// Create triangle UV coordinates from global vertex UVs and triangle indices
pub fn create_triangle_uvs(
    vertex_uvs: &Array2<f64>,
    all_triangle_indices: &[usize],
) -> MeshResult<Array2<f64>> {
    let num_corners = all_triangle_indices.len();
    if num_corners % 3 != 0 {
        return Err(MeshError::InvalidTriangleData(
            "Triangle indices must be divisible by 3".to_string(),
        ));
    }

    let num_triangles = num_corners / 3;
    let mut triangle_uvs = Array2::zeros((num_corners, 2));

    for i in 0..num_triangles {
        for k in 0..3 {
            let corner_idx = i * 3 + k;
            let vertex_idx = all_triangle_indices[corner_idx];

            if vertex_idx < vertex_uvs.shape()[0] {
                triangle_uvs[[corner_idx, 0]] = vertex_uvs[[vertex_idx, 0]];
                triangle_uvs[[corner_idx, 1]] = vertex_uvs[[vertex_idx, 1]];
            } else {
                return Err(MeshError::InvalidUvData(format!(
                    "Vertex index {} out of bounds for UV mapping (UV array size: {})",
                    vertex_idx,
                    vertex_uvs.shape()[0]
                )));
            }
        }
    }

    Ok(triangle_uvs)
}

/// Validate that a file path exists
pub fn validate_file_exists(path: &str) -> MeshResult<()> {
    if !Path::new(path).exists() {
        Err(MeshError::FileNotFound(path.to_string()))
    } else {
        Ok(())
    }
}

/// Validate texture file path
fn validate_texture_path(texture_path: &str) -> MeshResult<()> {
    validate_file_exists(texture_path).map_err(|_| {
        MeshError::TextureLoadError(format!("Texture file not found: {}", texture_path))
    })
}

/// Aggregate vertex data from multiple models for shared texture processing
pub fn aggregate_model_data(models: &[tobj::Model]) -> (Vec<f32>, Vec<f32>, usize) {
    // First pass: compute total lengths so we can reserve exact capacity.
    let total_positions_len: usize = models.iter().map(|m| m.mesh.positions.len()).sum();
    let total_vertex_count: usize = models.iter().map(|m| m.mesh.positions.len() / 3).sum();

    let mut all_positions = Vec::with_capacity(total_positions_len);
    // Each vertex can contribute up to 2 texcoords (u,v)
    let mut all_texcoords = Vec::with_capacity(total_vertex_count * 2);

    let mut processed_vertex_count = 0; // we will re-compute below for return value

    for (i, model) in models.iter().enumerate() {
        log::info!(
            "Model {}: '{}' - positions: {}, texcoords: {}, texcoord_indices: {}, indices: {}",
            i,
            model.name,
            model.mesh.positions.len(),
            model.mesh.texcoords.len(),
            model.mesh.texcoord_indices.len(),
            model.mesh.indices.len()
        );

        all_positions.extend_from_slice(&model.mesh.positions);

        // Handle texture coordinates based on whether we have separate indices or not
        if !model.mesh.texcoord_indices.is_empty() {
            // With separate texture coordinate indices, we need to create a per-vertex texture coordinate array
            // that matches the vertex order, using the texcoord_indices to map to the actual texcoords
            let vertex_count = model.mesh.positions.len() / 3;

            // Create a mapping from vertex indices to texture coordinates
            let mut vertex_texcoords = vec![0.0f32; vertex_count * 2];

            // Check if we have enough texcoord_indices (should match the number of triangle corners)
            if model.mesh.texcoord_indices.len() == model.mesh.indices.len() {
                // Map texture coordinates to vertices using both position indices and texcoord indices
                for (corner_idx, &pos_idx) in model.mesh.indices.iter().enumerate() {
                    let vertex_idx = pos_idx as usize;
                    let texcoord_idx = model.mesh.texcoord_indices[corner_idx] as usize;

                    if vertex_idx < vertex_count
                        && texcoord_idx * 2 + 1 < model.mesh.texcoords.len()
                    {
                        // Assign texture coordinates to this vertex (taking the first occurrence)
                        if vertex_texcoords[vertex_idx * 2] == 0.0
                            && vertex_texcoords[vertex_idx * 2 + 1] == 0.0
                        {
                            vertex_texcoords[vertex_idx * 2] =
                                model.mesh.texcoords[texcoord_idx * 2];
                            vertex_texcoords[vertex_idx * 2 + 1] =
                                model.mesh.texcoords[texcoord_idx * 2 + 1];
                        }
                    }
                }
                all_texcoords.extend_from_slice(&vertex_texcoords);
            } else {
                log::warn!(
                    "Texcoord indices count ({}) doesn't match triangle indices count ({}) for model '{}', using fallback",
                    model.mesh.texcoord_indices.len(),
                    model.mesh.indices.len(),
                    model.name
                );
                // Fallback: use original texcoords or create default ones
                if model.mesh.texcoords.len() >= vertex_count * 2 {
                    all_texcoords.extend_from_slice(&model.mesh.texcoords[..vertex_count * 2]);
                } else {
                    // Create default UV coordinates for all vertices
                    for _ in 0..vertex_count {
                        all_texcoords.push(0.0);
                        all_texcoords.push(0.0);
                    }
                }
            }
        } else {
            // No separate indices, use texture coordinates directly (legacy single-index behavior)
            let vertex_count = model.mesh.positions.len() / 3;
            if model.mesh.texcoords.len() >= vertex_count * 2 {
                all_texcoords.extend_from_slice(&model.mesh.texcoords[..vertex_count * 2]);
            } else {
                // Pad with default coordinates if not enough texture coordinates
                all_texcoords.extend_from_slice(&model.mesh.texcoords);
                let missing_coords = vertex_count * 2 - model.mesh.texcoords.len();
                for _ in 0..missing_coords {
                    all_texcoords.push(0.0);
                }
            }
        }

        processed_vertex_count += model.mesh.positions.len() / 3;
    }

    log::info!(
        "Aggregated data - total_vertex_count: {}, all_texcoords.len(): {}",
        processed_vertex_count,
        all_texcoords.len()
    );

    (all_positions, all_texcoords, processed_vertex_count)
}

/// Create global triangle indices for UV mapping
pub fn create_global_triangle_indices(models: &[tobj::Model]) -> Vec<usize> {
    let mut global_indices = Vec::new();
    let mut vertex_offset = 0;

    for model in models {
        for &index in &model.mesh.indices {
            global_indices.push(index as usize + vertex_offset);
        }
        vertex_offset += model.mesh.positions.len() / 3;
    }

    global_indices
}

/// Check if models have separate texture coordinate indices (single_index: false case)
pub fn has_separate_texcoord_indices(models: &[tobj::Model]) -> bool {
    models
        .iter()
        .any(|model| !model.mesh.texcoord_indices.is_empty())
}
