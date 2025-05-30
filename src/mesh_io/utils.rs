//! Utility functions for mesh I/O operations.

use crate::mesh_io::errors::{MeshError, MeshResult};
use ndarray::{Array, Array2, Array3};
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

    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();

    if width == 0 || height == 0 {
        return Err(MeshError::TextureProcessingError(format!(
            "Invalid texture dimensions: {}x{} for file '{}'",
            width, height, texture_path
        )));
    }

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

    let mut uvs_arr = Array::zeros((vertex_count, 2));

    for i in 0..vertex_count {
        for j in 0..2 {
            let mut value = vertex_texcoords[i * 2 + j] as f64;
            value = value.clamp(0.0, 1.0); // Clamp UVs to valid range

            if j == 1 && flip_v_coordinate {
                value = 1.0 - value; // Flip V coordinate
            }

            uvs_arr[[i, j]] = value;
        }
    }

    Ok(uvs_arr)
}

/// Convert vertex positions from f32 to f64 array format
pub fn convert_vertex_positions(positions: &[f32]) -> MeshResult<Array2<f64>> {
    if positions.len() % 3 != 0 {
        return Err(MeshError::InvalidVertexData(
            "Position array length must be divisible by 3".to_string(),
        ));
    }

    let vertex_count = positions.len() / 3;
    let mut vertices = Array2::zeros((vertex_count, 3));

    for i in 0..vertex_count {
        for j in 0..3 {
            vertices[[i, j]] = positions[i * 3 + j] as f64;
        }
    }

    Ok(vertices)
}

/// Convert triangle indices from u32 to usize array format
pub fn convert_triangle_indices(indices: &[u32]) -> MeshResult<Array2<usize>> {
    if indices.len() % 3 != 0 {
        return Err(MeshError::InvalidTriangleData(
            "Index array length must be divisible by 3".to_string(),
        ));
    }

    let triangle_count = indices.len() / 3;
    let mut triangles = Array2::zeros((triangle_count, 3));

    for i in 0..triangle_count {
        for j in 0..3 {
            triangles[[i, j]] = indices[i * 3 + j] as usize;
        }
    }

    Ok(triangles)
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

    let mut normals_arr = Array2::zeros((vertex_count, 3));

    for i in 0..vertex_count {
        for j in 0..3 {
            normals_arr[[i, j]] = normals[i * 3 + j] as f64;
        }
    }

    Ok(normals_arr)
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

    let mut colors_arr = Array2::zeros((vertex_count, 3));

    for i in 0..vertex_count {
        for j in 0..3 {
            // Values are typically 0.0-1.0 range from tobj
            colors_arr[[i, j]] = colors[i * 3 + j] as f64;
        }
    }

    Ok(colors_arr)
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
    let mut all_positions = Vec::new();
    let mut all_texcoords = Vec::new();
    let mut total_vertex_count = 0;

    for model in models {
        all_positions.extend_from_slice(&model.mesh.positions);
        all_texcoords.extend_from_slice(&model.mesh.texcoords);
        total_vertex_count += model.mesh.positions.len() / 3;
    }

    (all_positions, all_texcoords, total_vertex_count)
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
