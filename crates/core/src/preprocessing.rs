//! Scene preprocessing operations for vertex and normal data.

use crate::constants::preprocessing::{BOUNDS_EPSILON, MIN_BOUNDING_SPHERE_RADIUS};
use crate::geometry::precompute_triangle_normals;
use crate::types::mesh::{Mesh, Scene};
use log::{info, warn};
use ndarray::Array2;

/// Describes the complete geometry preprocessing recipe applied uniformly
/// across all meshes of a scene: translate to global center, scale
/// to the target resolution cube, optionally clamp to bounds, then shift to the cube center.
#[derive(Clone, Copy, Debug)]
pub struct MeshPreprocessing {
    pub center: [f32; 3],
    pub scale: f32,
    pub shift: [f32; 3],
    pub target_max_val: f32,
    pub clamp_to_target_bounds: bool,
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
    for triangle_idx in 0..triangle_count {
        if let Some(face_normal) = compute_triangle_normal(vertices, triangles, triangle_idx) {
            let indices = get_triangle_indices(triangles, triangle_idx);
            for &vertex_idx in &indices {
                if vertex_idx < vertex_count {
                    for component in 0..3 {
                        normals[[vertex_idx, component]] += face_normal[component];
                    }
                }
            }
        }
    }
    normalize_normals(&mut normals);
    Some(normals)
}

/// Normalize an array of normal vectors
pub fn normalize_normals(normals: &mut Array2<f32>) {
    let vertex_count = normals.shape()[0];
    for vertex_idx in 0..vertex_count {
        let norm = calculate_vector_magnitude(normals, vertex_idx);
        if norm > 1e-9 {
            for component in 0..3 {
                normals[[vertex_idx, component]] /= norm;
            }
        } else {
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
    let preprocessing = build_preprocessing_from_vertices(&points, resolution);
    apply_preprocessing_to_vertices(&mut vertices, &preprocessing);
    let final_bounds = calculate_bounding_box(&vertices);
    info!(
        "Final BBox Min: {:?}, Max: {:?}",
        final_bounds.min, final_bounds.max
    );
    vertices
}

/// Build a reusable preprocessing recipe from vertex positions and resolution.
pub fn build_preprocessing_from_vertices(
    vertices: &[[f32; 3]],
    resolution: u16,
) -> MeshPreprocessing {
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
    MeshPreprocessing {
        center: sphere.center,
        scale: scale_factor,
        shift,
        target_max_val,
        clamp_to_target_bounds: true,
    }
}

/// Build a reusable preprocessing recipe from all vertices in a scene.
pub fn build_preprocessing_from_scene(scene: &Scene, resolution: u16) -> MeshPreprocessing {
    let mut vertices = Vec::new();
    for mesh in &scene.meshes {
        vertices.extend(collect_vertices(&mesh.geometry.vertices));
    }
    build_preprocessing_from_vertices(&vertices, resolution)
}

/// Apply a previously built preprocessing recipe to a vertex array.
/// Order: translate by -center, scale uniformly, then shift to cube center.
pub fn apply_preprocessing_to_vertices(vertices: &mut Array2<f32>, p: &MeshPreprocessing) {
    let vertex_count = vertices.shape()[0];
    for vertex_idx in 0..vertex_count {
        for component in 0..3 {
            vertices[[vertex_idx, component]] -= p.center[component];
        }
    }
    for vertex_idx in 0..vertex_count {
        for component in 0..3 {
            vertices[[vertex_idx, component]] *= p.scale;
        }
    }
    for vertex_idx in 0..vertex_count {
        for component in 0..3 {
            vertices[[vertex_idx, component]] += p.shift[component];
        }
    }
    if p.clamp_to_target_bounds {
        clamp_vertices_if_needed(vertices, p.target_max_val);
    }
}

pub fn apply_preprocessing_to_mesh(mesh: &mut Mesh, preprocessing: &MeshPreprocessing) {
    apply_preprocessing_to_vertices(&mut mesh.geometry.vertices, preprocessing);
    mesh.geometry.face_normals = precompute_triangle_normals(
        mesh.geometry.vertices.view(),
        mesh.geometry.triangles.view(),
    );
    if let Some(normals) = mesh.geometry.vertex_normals.as_mut() {
        normalize_normals(normals);
    } else {
        mesh.geometry.vertex_normals =
            compute_vertex_normals(&mesh.geometry.vertices, &mesh.geometry.triangles);
    }
}

pub fn apply_preprocessing_to_scene(scene: &mut Scene, preprocessing: &MeshPreprocessing) {
    for mesh in &mut scene.meshes {
        apply_preprocessing_to_mesh(mesh, preprocessing);
    }
}

pub fn preprocessed_scene(scene: &Scene, preprocessing: &MeshPreprocessing) -> Scene {
    let mut scene = scene.clone();
    apply_preprocessing_to_scene(&mut scene, preprocessing);
    scene
}

pub fn preprocessed_mesh(mesh: &Mesh, preprocessing: &MeshPreprocessing) -> Mesh {
    let mut mesh = mesh.clone();
    apply_preprocessing_to_mesh(&mut mesh, preprocessing);
    mesh
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

#[derive(Debug)]
struct BoundingBox {
    min: [f32; 3],
    max: [f32; 3],
}

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

fn compute_triangle_normal(
    vertices: &Array2<f32>,
    triangles: &Array2<usize>,
    triangle_idx: usize,
) -> Option<[f32; 3]> {
    let indices = get_triangle_indices(triangles, triangle_idx);
    let triangle_vertices = get_triangle_vertices(vertices, &indices)?;
    let edge1 = subtract_vertices(&triangle_vertices[1], &triangle_vertices[0]);
    let edge2 = subtract_vertices(&triangle_vertices[2], &triangle_vertices[0]);
    let normal = cross_product(&edge1, &edge2);
    let length = vector_magnitude(&normal);
    if length > 1e-9 {
        Some([normal[0] / length, normal[1] / length, normal[2] / length])
    } else {
        None
    }
}

fn get_triangle_indices(triangles: &Array2<usize>, triangle_idx: usize) -> [usize; 3] {
    [
        triangles[[triangle_idx, 0]],
        triangles[[triangle_idx, 1]],
        triangles[[triangle_idx, 2]],
    ]
}

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

fn subtract_vertices(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross_product(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vector_magnitude(v: &[f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn calculate_vector_magnitude(normals: &Array2<f32>, vertex_idx: usize) -> f32 {
    let mut norm = 0.0;
    for component in 0..3 {
        let value = normals[[vertex_idx, component]];
        norm += value * value;
    }
    norm.sqrt()
}

fn set_default_normal(normals: &mut Array2<f32>, vertex_idx: usize) {
    normals[[vertex_idx, 0]] = 0.0;
    normals[[vertex_idx, 1]] = 0.0;
    normals[[vertex_idx, 2]] = 1.0;
}
