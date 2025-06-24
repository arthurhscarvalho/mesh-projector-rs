//! Mesh preprocessing operations for vertex and normal data.

use log::{info, warn};
use ndarray::Array2;

/// Compute vertex normals for a mesh by averaging face normals
pub fn compute_vertex_normals(
    vertices: &Array2<f32>,
    triangles: &Array2<usize>,
) -> Option<Array2<f32>> {
    let vertex_count = vertices.shape()[0];
    let triangle_count = triangles.shape()[0];

    if vertex_count == 0 || triangle_count == 0 {
        return None;
    }

    let mut normals = Array2::zeros((vertex_count, 3));

    // Accumulate face normals for each vertex
    for triangle_idx in 0..triangle_count {
        if let Some(face_normal) = compute_triangle_normal(vertices, triangles, triangle_idx) {
            let indices = get_triangle_indices(triangles, triangle_idx);

            // Add face normal to each vertex of the triangle
            for &vertex_idx in &indices {
                if vertex_idx < vertex_count {
                    for component in 0..3 {
                        normals[[vertex_idx, component]] += face_normal[component];
                    }
                }
            }
        }
    }

    // Normalize all vertex normals
    normalize_normals(&mut normals);
    Some(normals)
}

/// Normalize an array of normal vectors
pub fn normalize_normals(normals: &mut Array2<f32>) {
    let vertex_count = normals.shape()[0];

    for vertex_idx in 0..vertex_count {
        let norm = calculate_vector_magnitude(normals, vertex_idx);

        if norm > 1e-9 {
            // Normalize the vector
            for component in 0..3 {
                normals[[vertex_idx, component]] /= norm;
            }
        } else {
            // Set default direction for zero normals
            set_default_normal(normals, vertex_idx);
        }
    }
}

/// Preprocess vertices by centering and scaling to fit within target precision
pub fn preprocess_vertices(mut vertices: Array2<f32>, precision: u32) -> Array2<f32> {
    let original_bounds = calculate_bounding_box(&vertices);
    info!(
        "Original BBox Min: {:?}, Max: {:?}",
        original_bounds.min, original_bounds.max
    );

    // Center the mesh
    center_vertices(&mut vertices, &original_bounds);

    let centered_bounds = calculate_bounding_box(&vertices);
    info!(
        "Centered BBox Min: {:?}, Max: {:?}",
        centered_bounds.min, centered_bounds.max
    );

    // Scale vertices to fit precision
    scale_vertices_to_precision(&mut vertices, precision);

    // Center in the precision cube
    center_in_precision_cube(&mut vertices, precision);

    let final_bounds = calculate_bounding_box(&vertices);
    info!(
        "Final BBox Min: {:?}, Max: {:?}",
        final_bounds.min, final_bounds.max
    );

    vertices
}

/// Bounding box structure for cleaner code
#[derive(Debug)]
struct BoundingBox {
    min: [f32; 3],
    max: [f32; 3],
}

impl BoundingBox {
    fn size(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) / 2.0,
            (self.min[1] + self.max[1]) / 2.0,
            (self.min[2] + self.max[2]) / 2.0,
        ]
    }
}

/// Calculate bounding box of vertices
fn calculate_bounding_box(vertices: &Array2<f32>) -> BoundingBox {
    let vertex_count = vertices.shape()[0];
    let mut min_bound = [f32::MAX; 3];
    let mut max_bound = [f32::MIN; 3];

    for vertex_idx in 0..vertex_count {
        for component in 0..3 {
            let value = vertices[[vertex_idx, component]];
            min_bound[component] = min_bound[component].min(value);
            max_bound[component] = max_bound[component].max(value);
        }
    }

    BoundingBox {
        min: min_bound,
        max: max_bound,
    }
}

/// Center vertices by translating to origin
fn center_vertices(vertices: &mut Array2<f32>, bounds: &BoundingBox) {
    let center = bounds.center();
    let vertex_count = vertices.shape()[0];

    info!("Center of mass: {:?}", center);

    for vertex_idx in 0..vertex_count {
        for component in 0..3 {
            vertices[[vertex_idx, component]] -= center[component];
        }
    }
}

/// Scale vertices to fit within precision bounds
fn scale_vertices_to_precision(vertices: &mut Array2<f32>, precision: u32) {
    let bounds = calculate_bounding_box(vertices);
    let mut bbox_size = bounds.size();

    // Avoid division by zero
    for component in 0..3 {
        if bbox_size[component] == 0.0 {
            bbox_size[component] = 1e-9;
        }
    }

    let max_dimension = bbox_size[0].max(bbox_size[1]).max(bbox_size[2]);
    let target_max_val = (1 << precision) as f32 - 1.0;
    // The safety margin is the amount of space to leave at the edges of the cube.
    let safety_margin = 1.0;
    let scale_factor = (target_max_val * safety_margin) / max_dimension;

    info!(
        "Scaling vertices. Max dimension: {}, Scale factor: {} (with {}% safety margin)",
        max_dimension,
        scale_factor,
        100.0 - safety_margin * 100.0
    );

    apply_scale_factor(vertices, scale_factor);
}

/// Center vertices in the precision cube
fn center_in_precision_cube(vertices: &mut Array2<f32>, precision: u32) {
    let target_max_val = (1 << precision) as f32 - 1.0;
    let cube_center = target_max_val / 2.0;

    let bounds = calculate_bounding_box(vertices);
    let current_center = bounds.center();

    let shift = [
        cube_center - current_center[0],
        cube_center - current_center[1],
        cube_center - current_center[2],
    ];

    info!("Shifting vertices to center in cube by: {:?}", shift);

    let vertex_count = vertices.shape()[0];
    for vertex_idx in 0..vertex_count {
        for component in 0..3 {
            vertices[[vertex_idx, component]] += shift[component];
        }
    }

    // Clamp to valid range if needed
    clamp_vertices_if_needed(vertices, target_max_val);
}

/// Apply a uniform scale factor to all vertices
fn apply_scale_factor(vertices: &mut Array2<f32>, scale_factor: f32) {
    let vertex_count = vertices.shape()[0];

    for vertex_idx in 0..vertex_count {
        for component in 0..3 {
            vertices[[vertex_idx, component]] *= scale_factor;
        }
    }
}

/// Clamp vertices to valid range if they exceed bounds
fn clamp_vertices_if_needed(vertices: &mut Array2<f32>, max_val: f32) {
    let bounds = calculate_bounding_box(vertices);
    let mut needs_clamping = false;

    for component in 0..3 {
        if bounds.min[component] < 0.0 || bounds.max[component] > max_val {
            needs_clamping = true;
            break;
        }
    }

    if needs_clamping {
        warn!("Mesh vertices out of bounds after centering. Clamping to valid range.");

        let vertex_count = vertices.shape()[0];
        for vertex_idx in 0..vertex_count {
            for component in 0..3 {
                vertices[[vertex_idx, component]] =
                    vertices[[vertex_idx, component]].clamp(0.0, max_val);
            }
        }

        info!("Note: Mesh was clamped to stay within bounds");
    }
}

/// Compute normal for a single triangle
fn compute_triangle_normal(
    vertices: &Array2<f32>,
    triangles: &Array2<usize>,
    triangle_idx: usize,
) -> Option<[f32; 3]> {
    let indices = get_triangle_indices(triangles, triangle_idx);
    let triangle_vertices = get_triangle_vertices(vertices, &indices)?;

    // Compute edges
    let edge1 = subtract_vertices(&triangle_vertices[1], &triangle_vertices[0]);
    let edge2 = subtract_vertices(&triangle_vertices[2], &triangle_vertices[0]);

    // Compute cross product (normal)
    let normal = cross_product(&edge1, &edge2);
    let length = vector_magnitude(&normal);

    if length > 1e-9 {
        Some([normal[0] / length, normal[1] / length, normal[2] / length])
    } else {
        None
    }
}

/// Get vertex indices for a triangle
fn get_triangle_indices(triangles: &Array2<usize>, triangle_idx: usize) -> [usize; 3] {
    [
        triangles[[triangle_idx, 0]],
        triangles[[triangle_idx, 1]],
        triangles[[triangle_idx, 2]],
    ]
}

/// Get vertex positions for a triangle
fn get_triangle_vertices(vertices: &Array2<f32>, indices: &[usize; 3]) -> Option<[[f32; 3]; 3]> {
    let vertex_count = vertices.shape()[0];

    for &idx in indices {
        if idx >= vertex_count {
            return None;
        }
    }

    Some([
        [
            vertices[[indices[0], 0]],
            vertices[[indices[0], 1]],
            vertices[[indices[0], 2]],
        ],
        [
            vertices[[indices[1], 0]],
            vertices[[indices[1], 1]],
            vertices[[indices[1], 2]],
        ],
        [
            vertices[[indices[2], 0]],
            vertices[[indices[2], 1]],
            vertices[[indices[2], 2]],
        ],
    ])
}

/// Vector subtraction
fn subtract_vertices(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Cross product of two 3D vectors
fn cross_product(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Calculate magnitude of a 3D vector
fn vector_magnitude(v: &[f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Calculate magnitude of a vector in the normals array
fn calculate_vector_magnitude(normals: &Array2<f32>, vertex_idx: usize) -> f32 {
    let mut norm = 0.0;
    for component in 0..3 {
        let value = normals[[vertex_idx, component]];
        norm += value * value;
    }
    norm.sqrt()
}

/// Set default normal direction for zero normals
fn set_default_normal(normals: &mut Array2<f32>, vertex_idx: usize) {
    normals[[vertex_idx, 0]] = 0.0;
    normals[[vertex_idx, 1]] = 0.0;
    normals[[vertex_idx, 2]] = 1.0;
}
