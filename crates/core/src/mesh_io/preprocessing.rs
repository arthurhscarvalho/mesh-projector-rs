//! Scene preprocessing operations for vertex and normal data.

use crate::constants::{BOUNDS_EPSILON, MIN_BOUNDING_SPHERE_RADIUS};
use log::{info, warn};
use ndarray::Array2;

/// Describes a deterministic preprocessing transform applied uniformly
/// across all meshes of a model: translate to global center, scale
/// to target resolution cube, then shift to the cube center.
#[derive(Clone, Copy, Debug)]
pub struct PreprocessTransform {
    pub center: [f32; 3],
    pub scale: f32,
    pub shift: [f32; 3],
    pub target_max_val: f32,
}

#[derive(Clone, Copy, Debug)]
struct BoundingSphere {
    center: [f32; 3],
    radius: f32,
}

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

/// Preprocess vertices by centering and scaling to fit within target resolution
pub fn preprocess_vertices(mut vertices: Array2<f32>, resolution: u16) -> Array2<f32> {
    let original_bounds = calculate_bounding_box(&vertices);
    info!(
        "Original BBox Min: {:?}, Max: {:?}",
        original_bounds.min, original_bounds.max
    );
    let points = collect_vertices(&vertices);
    let transform = build_transform_from_vertices(&points, resolution);
    apply_transform_to_vertices(&mut vertices, &transform);
    let final_bounds = calculate_bounding_box(&vertices);
    info!(
        "Final BBox Min: {:?}, Max: {:?}",
        final_bounds.min, final_bounds.max
    );
    vertices
}

/// Build a global preprocessing transform from vertex positions and resolution.
/// This lets us preprocess all meshes consistently using a single transform.
pub fn build_transform_from_vertices(
    vertices: &[[f32; 3]],
    resolution: u16,
) -> PreprocessTransform {
    let sphere = ritter_bounding_sphere(vertices);
    let target_max_val = resolution as f32 - 1.0;
    let diameter = 2.0 * sphere.radius.max(MIN_BOUNDING_SPHERE_RADIUS);
    let scale_factor = target_max_val / diameter;
    let cube_center = target_max_val / 2.0;
    let shift = [cube_center, cube_center, cube_center];
    info!(
        "Global sphere transform — center: {:?}, radius: {}, scale: {}, cube_center: {}",
        sphere.center, sphere.radius, scale_factor, cube_center
    );
    PreprocessTransform {
        center: sphere.center,
        scale: scale_factor,
        shift,
        target_max_val,
    }
}

/// Apply a previously built preprocessing transform to a vertex array.
/// Order: translate by -center, scale uniformly, then shift to cube center.
pub fn apply_transform_to_vertices(vertices: &mut Array2<f32>, t: &PreprocessTransform) {
    let vertex_count = vertices.shape()[0];
    // Translate by global center
    for vertex_idx in 0..vertex_count {
        for component in 0..3 {
            vertices[[vertex_idx, component]] -= t.center[component];
        }
    }
    // Uniform scale
    for vertex_idx in 0..vertex_count {
        for component in 0..3 {
            vertices[[vertex_idx, component]] *= t.scale;
        }
    }
    // Shift to cube center
    for vertex_idx in 0..vertex_count {
        for component in 0..3 {
            vertices[[vertex_idx, component]] += t.shift[component];
        }
    }
    // Clamp to valid range if needed
    clamp_vertices_if_needed(vertices, t.target_max_val);
}

fn collect_vertices(vertices: &Array2<f32>) -> Vec<[f32; 3]> {
    (0..vertices.shape()[0])
        .map(|vertex_idx| {
            [
                vertices[[vertex_idx, 0]],
                vertices[[vertex_idx, 1]],
                vertices[[vertex_idx, 2]],
            ]
        })
        .collect()
}

fn ritter_bounding_sphere(vertices: &[[f32; 3]]) -> BoundingSphere {
    if vertices.is_empty() {
        return BoundingSphere {
            center: [0.0, 0.0, 0.0],
            radius: MIN_BOUNDING_SPHERE_RADIUS,
        };
    }

    let x = farthest_point(vertices, vertices[0]);
    let y = farthest_point(vertices, x);
    let mut center = midpoint(x, y);
    let mut radius = distance(x, y) * 0.5;

    for &vertex in vertices {
        let offset = subtract_vertices(&vertex, &center);
        let distance_sq = dot(&offset, &offset);
        let radius_sq = radius * radius;
        if distance_sq > radius_sq {
            let distance = distance_sq.sqrt();
            if distance > MIN_BOUNDING_SPHERE_RADIUS {
                let new_radius = (radius + distance) * 0.5;
                let center_shift = (new_radius - radius) / distance;
                for component in 0..3 {
                    center[component] += offset[component] * center_shift;
                }
                radius = new_radius;
            }
        }
    }

    BoundingSphere {
        center,
        radius: radius.max(MIN_BOUNDING_SPHERE_RADIUS),
    }
}

fn farthest_point(vertices: &[[f32; 3]], from: [f32; 3]) -> [f32; 3] {
    let mut farthest = vertices[0];
    let mut farthest_distance_sq = distance_sq(farthest, from);
    for &vertex in &vertices[1..] {
        let candidate_distance_sq = distance_sq(vertex, from);
        if candidate_distance_sq > farthest_distance_sq {
            farthest = vertex;
            farthest_distance_sq = candidate_distance_sq;
        }
    }
    farthest
}

fn midpoint(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ]
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    distance_sq(a, b).sqrt()
}

fn distance_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
    let offset = subtract_vertices(&a, &b);
    dot(&offset, &offset)
}

fn dot(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Bounding box structure for cleaner code
#[derive(Debug)]
struct BoundingBox {
    min: [f32; 3],
    max: [f32; 3],
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

/// Clamp vertices to valid range if they exceed bounds
fn clamp_vertices_if_needed(vertices: &mut Array2<f32>, max_val: f32) {
    let bounds = calculate_bounding_box(vertices);
    let mut needs_clamping = false;
    for component in 0..3 {
        if bounds.min[component] < -BOUNDS_EPSILON
            || bounds.max[component] > max_val + BOUNDS_EPSILON
        {
            needs_clamping = true;
            break;
        }
    }
    if needs_clamping {
        warn!("Scene vertices out of bounds after centering. Clamping to valid range.");
        let vertex_count = vertices.shape()[0];
        for vertex_idx in 0..vertex_count {
            for component in 0..3 {
                vertices[[vertex_idx, component]] =
                    vertices[[vertex_idx, component]].clamp(0.0, max_val);
            }
        }
        info!("Note: Scene was clamped to stay within bounds");
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn assert_close(actual: f32, expected: f32, epsilon: f32) {
        assert!(
            (actual - expected).abs() <= epsilon,
            "expected {actual} to be within {epsilon} of {expected}"
        );
    }

    fn assert_contains_all(sphere: BoundingSphere, vertices: &[[f32; 3]]) {
        let radius_sq = sphere.radius * sphere.radius;
        for &vertex in vertices {
            assert!(
                distance_sq(vertex, sphere.center) <= radius_sq + BOUNDS_EPSILON,
                "sphere {:?} did not contain vertex {:?}",
                sphere,
                vertex
            );
        }
    }

    fn assert_vertices_in_resolution_cube(vertices: &Array2<f32>, max_val: f32) {
        for vertex_idx in 0..vertices.shape()[0] {
            for component in 0..3 {
                let value = vertices[[vertex_idx, component]];
                assert!(
                    value >= -BOUNDS_EPSILON && value <= max_val + BOUNDS_EPSILON,
                    "vertex {vertex_idx} component {component} out of bounds: {value}"
                );
            }
        }
    }

    #[test]
    fn ritter_sphere_for_two_points_uses_midpoint_and_half_distance() {
        let vertices = [[-2.0, 0.0, 0.0], [4.0, 0.0, 0.0]];

        let sphere = ritter_bounding_sphere(&vertices);

        assert_close(sphere.center[0], 1.0, 1e-6);
        assert_close(sphere.center[1], 0.0, 1e-6);
        assert_close(sphere.center[2], 0.0, 1e-6);
        assert_close(sphere.radius, 3.0, 1e-6);
        assert_contains_all(sphere, &vertices);
    }

    #[test]
    fn ritter_sphere_contains_cube_corners() {
        let vertices = [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ];

        let sphere = ritter_bounding_sphere(&vertices);

        assert_contains_all(sphere, &vertices);
        assert_close(sphere.radius, 3.0_f32.sqrt(), 1e-6);
    }

    #[test]
    fn ritter_sphere_expands_to_contain_asymmetric_point_cloud() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [2.0, 5.0, 0.5],
            [-1.0, 3.0, 2.0],
            [3.0, -4.0, -2.5],
            [2.0, 1.0, 6.0],
        ];

        let sphere = ritter_bounding_sphere(&vertices);

        assert_contains_all(sphere, &vertices);
        assert!(sphere.radius.is_finite());
        assert!(sphere.radius > MIN_BOUNDING_SPHERE_RADIUS);
    }

    #[test]
    fn degenerate_vertices_produce_finite_nonzero_transform() {
        let vertices = [[3.0, -2.0, 7.0]; 4];

        let transform = build_transform_from_vertices(&vertices, 64);

        assert_eq!(transform.center, [3.0, -2.0, 7.0]);
        assert!(transform.scale.is_finite());
        assert!(transform.scale > 0.0);
    }

    #[test]
    fn sphere_transform_places_vertices_inside_resolution_cube() {
        let points = [
            [-4.0, -1.0, 0.0],
            [7.0, 2.0, 1.0],
            [0.0, 5.0, -3.0],
            [2.0, -6.0, 4.0],
        ];
        let mut vertices = arr2(&[
            [-4.0, -1.0, 0.0],
            [7.0, 2.0, 1.0],
            [0.0, 5.0, -3.0],
            [2.0, -6.0, 4.0],
        ]);
        let resolution = 128;
        let max_val = resolution as f32 - 1.0;
        let transform = build_transform_from_vertices(&points, resolution);

        apply_transform_to_vertices(&mut vertices, &transform);

        assert_vertices_in_resolution_cube(&vertices, max_val);
    }

    #[test]
    fn transformed_vertices_remain_in_sphere_cube_after_non_axis_rotation() {
        let points = [
            [-4.0, -1.0, 0.0],
            [7.0, 2.0, 1.0],
            [0.0, 5.0, -3.0],
            [2.0, -6.0, 4.0],
            [-2.0, 3.0, 5.0],
        ];
        let mut vertices = arr2(&[
            [-4.0, -1.0, 0.0],
            [7.0, 2.0, 1.0],
            [0.0, 5.0, -3.0],
            [2.0, -6.0, 4.0],
            [-2.0, 3.0, 5.0],
        ]);
        let resolution = 256;
        let max_val = resolution as f32 - 1.0;
        let cube_center = max_val * 0.5;
        let transform = build_transform_from_vertices(&points, resolution);

        apply_transform_to_vertices(&mut vertices, &transform);
        rotate_vertices_around_center(&mut vertices, cube_center, 0.37, -0.71, 0.23);

        assert_vertices_in_resolution_cube(&vertices, max_val);
    }

    fn rotate_vertices_around_center(
        vertices: &mut Array2<f32>,
        center: f32,
        pitch: f32,
        yaw: f32,
        roll: f32,
    ) {
        let (sin_pitch, cos_pitch) = pitch.sin_cos();
        let (sin_yaw, cos_yaw) = yaw.sin_cos();
        let (sin_roll, cos_roll) = roll.sin_cos();

        for vertex_idx in 0..vertices.shape()[0] {
            let mut x = vertices[[vertex_idx, 0]] - center;
            let mut y = vertices[[vertex_idx, 1]] - center;
            let mut z = vertices[[vertex_idx, 2]] - center;

            let y_pitch = y * cos_pitch - z * sin_pitch;
            let z_pitch = y * sin_pitch + z * cos_pitch;
            y = y_pitch;
            z = z_pitch;

            let x_yaw = x * cos_yaw + z * sin_yaw;
            let z_yaw = -x * sin_yaw + z * cos_yaw;
            x = x_yaw;
            z = z_yaw;

            let x_roll = x * cos_roll - y * sin_roll;
            let y_roll = x * sin_roll + y * cos_roll;
            x = x_roll;
            y = y_roll;

            vertices[[vertex_idx, 0]] = x + center;
            vertices[[vertex_idx, 1]] = y + center;
            vertices[[vertex_idx, 2]] = z + center;
        }
    }
}
