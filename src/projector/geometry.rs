//! This module provides core geometric algorithms and utilities.
//! It includes functions for computing barycentric coordinates, triangle normals, and other
//! essential operations required for rasterization, visibility testing, and surface analysis
//! during the projection pipeline.

use crate::projector::constants::{BARYCENTRIC_COORD_EPSILON, BARYCENTRIC_EPSILON};
use crate::types::coordinate::Coordinate;
use nd::{array, Array2, ArrayView2};
use std::f32;

/// Computes barycentric coordinates (u, v, w) for point P relative to triangle ABC in 2D.
///
/// Returns `Some((u, v, w))` such that P = uA + vB + wC, where u+v+w = 1.
/// Returns `None` if the triangle is degenerate (vertices are collinear, denominator is close to zero).
pub fn barycentric(
    a: (f32, f32), // Projected coordinates of vertex A
    b: (f32, f32), // Projected coordinates of vertex B
    c: (f32, f32), // Projected coordinates of vertex C
    p: (f32, f32), // Coordinates of the point P (e.g., pixel center)
) -> Option<(f32, f32, f32)> {
    // Vectors from A to B (v0) and A to C (v1)
    let v0x = b.0 - a.0;
    let v0y = b.1 - a.1;
    let v1x = c.0 - a.0;
    let v1y = c.1 - a.1;
    // Vector from A to P (v2)
    let v2x = p.0 - a.0;
    let v2y = p.1 - a.1;
    // Dot products needed for the calculation
    let d00 = v0x * v0x + v0y * v0y; // dot(v0, v0)
    let d01 = v0x * v1x + v0y * v1y; // dot(v0, v1)
    let d11 = v1x * v1x + v1y * v1y; // dot(v1, v1)
    let d20 = v2x * v0x + v2y * v0y; // dot(v2, v0)
    let d21 = v2x * v1x + v2y * v1y; // dot(v2, v1)
                                     // Denominator for Cramer's rule approach
    let denom = d00 * d11 - d01 * d01;
    // Check for degenerate triangle (collinear vertices)
    if denom.abs() < BARYCENTRIC_EPSILON {
        return None;
    }
    let inv_denom = 1.0 / denom;
    // Calculate v and w coordinates
    let v = (d11 * d20 - d01 * d21) * inv_denom;
    let w = (d00 * d21 - d01 * d20) * inv_denom;
    // Calculate u coordinate using the property u + v + w = 1
    let u = 1.0 - v - w;
    Some((u, v, w))
}

/// Precomputes the geometric (unnormalized) normal vector for each triangle face.
/// The normal is calculated using the cross product of two edge vectors (p1-p0) x (p2-p0).
pub fn precompute_triangle_normals(
    points: ArrayView2<f32>,      // Vertex positions (N, 3)
    triangles: ArrayView2<usize>, // Triangle vertex indices (M, 3)
) -> Array2<f32> {
    let num_faces = triangles.nrows();
    let mut triangle_normals = Array2::<f32>::zeros((num_faces, 3));
    // This loop computes the normal for each triangle face.
    // It could be parallelized if num_faces is extremely large, but for typical meshes,
    // sequential computation is often fast enough and simpler.
    for triangle_index in 0..num_faces {
        let tri_indices = triangles.row(triangle_index);
        let i0 = tri_indices[0];
        let i1 = tri_indices[1];
        let i2 = tri_indices[2];
        // Get the 3D coordinates of the triangle's vertices
        let p0 = points.row(i0);
        let p1 = points.row(i1);
        let p2 = points.row(i2);
        // Calculate the vectors representing two edges of the triangle starting from p0
        let v0x = p1[0] - p0[0]; // p1 - p0
        let v0y = p1[1] - p0[1];
        let v0z = p1[2] - p0[2];
        let v1x = p2[0] - p0[0]; // p2 - p0
        let v1y = p2[1] - p0[1];
        let v1z = p2[2] - p0[2];
        // Compute the cross product: (p1 - p0) x (p2 - p0)
        // This gives a vector perpendicular to the triangle's surface (the geometric normal).
        // The magnitude of this vector is twice the area of the triangle.
        let normal_x = v0y * v1z - v0z * v1y;
        let normal_y = v0z * v1x - v0x * v1z;
        let normal_z = v0x * v1y - v0y * v1x;
        // Store the computed normal components
        triangle_normals[[triangle_index, 0]] = normal_x;
        triangle_normals[[triangle_index, 1]] = normal_y;
        triangle_normals[[triangle_index, 2]] = normal_z;
    }
    triangle_normals
}

/// Checks if a triangle is facing towards the view direction using basic back-face culling.
///
/// Compares the triangle's geometric normal with the view direction. If the angle between them
/// is less than 90 degrees (dot product > 0), the triangle is facing the viewer and is culled.
/// We render triangles whose normal points *away* from the view direction (dot product < 0).
pub fn is_triangle_visible(
    normal: Coordinate,         // Triangle's normal vector
    view_direction: Coordinate, // Normalized view direction vector
) -> bool {
    // Calculate the dot product of the triangle normal and the view direction vector.
    let dot = normal.dot(&view_direction);
    // If the dot product is negative, the normal points generally away from the view direction
    // (angle > 90 degrees), meaning the front face of the triangle (using right-hand rule for normal)
    // is visible from this view direction.
    // If dot is positive, the normal points towards the viewer (angle < 90 degrees), culling it (back-face).
    // If dot is zero, the triangle is edge-on and might be culled or rendered based on convention; here it's culled.
    dot < 0.0
}

/// Calculates the 2D bounding box (in pixel coordinates) of a projected triangle.
/// The box is clamped to the screen dimensions.
///
/// Returns `Some((min_row, max_row, min_col, max_col))` if the box is valid and at least
/// partially on-screen, otherwise returns `None`.
pub fn calculate_bounding_box(
    p0_proj: (f32, f32), // Projected point 0 (p_axis_val, q_axis_val) -> (col, row)
    p1_proj: (f32, f32), // Projected point 1
    p2_proj: (f32, f32), // Projected point 2
    rows: usize,         // Number of rows in the target image buffer
    cols: usize,         // Number of columns in the target image buffer
) -> Option<(usize, usize, usize, usize)> {
    // (min_row, max_row, min_col, max_col)
    // Find the min/max coordinates of the projected vertices along the P (column) and Q (row) axes.
    let min_col_f = p0_proj.0.min(p1_proj.0.min(p2_proj.0));
    let max_col_f = p0_proj.0.max(p1_proj.0.max(p2_proj.0));
    let min_row_f = p0_proj.1.min(p1_proj.1.min(p2_proj.1));
    let max_row_f = p0_proj.1.max(p1_proj.1.max(p2_proj.1));
    // Convert floating-point bounds to integer pixel indices.
    // Use floor() for minimums and ceil() for maximums to ensure the bounding box
    // fully encloses the projected triangle area.
    // Clamp the results to the valid range of pixel indices [0, cols-1] and [0, rows-1].
    // Max(0) handles cases where coordinates are negative.
    let min_col = (min_col_f.floor() as i64).max(0) as usize;
    // Ensure max_col doesn't exceed the last valid index (cols - 1). Max(0) prevents underflow if result is negative.
    let max_col = (max_col_f.ceil() as i64).min(cols as i64 - 1).max(0) as usize;
    let min_row = (min_row_f.floor() as i64).max(0) as usize;
    let max_row = (max_row_f.ceil() as i64).min(rows as i64 - 1).max(0) as usize;
    // Check if the calculated bounding box is valid (min <= max) and
    // if it has any intersection with the screen area.
    // If min_col >= cols or min_row >= rows, the box starts outside the screen bounds.
    // If min_col > max_col or min_row > max_row, the box has zero or negative size,
    // which can happen if the triangle projects to a line or point outside the screen.
    if min_col >= cols || min_row >= rows || min_col > max_col || min_row > max_row {
        None // Bounding box is completely off-screen or invalid.
    } else {
        Some((min_row, max_row, min_col, max_col))
    }
}

/// Checks if a point defined by barycentric coordinates is inside the triangle.
/// Uses a small tolerance `BARYCENTRIC_COORD_EPSILON` to account for floating-point inaccuracies.
pub fn is_inside_triangle(u: f32, v: f32, w: f32) -> bool {
    u >= BARYCENTRIC_COORD_EPSILON
        && v >= BARYCENTRIC_COORD_EPSILON
        && w >= BARYCENTRIC_COORD_EPSILON
}

/// Calculates the area of a triangle given its vertices.
pub fn calculate_triangle_area(vertices: &[Coordinate; 3]) -> f32 {
    let ab = vertices[1] - vertices[0];
    let ac = vertices[2] - vertices[0];
    let cross = ab.cross(&ac);
    let triangle_area = 0.5 * cross.length();
    triangle_area
}

/// Creates a 4x4 view matrix from a 3x3 rotation matrix.
/// For orthographic projection, the camera is the projection plane itself.
/// This creates a view matrix that transforms from world space to view space,
/// where the view direction becomes the Z axis.
pub fn create_view_matrix(rotation_matrix: &Array2<f32>) -> Array2<f32> {
    // In orthographic projection, we don't position the camera at a distance
    // Instead, the projection plane itself acts as the camera
    // The rotation matrix should already be set up to transform the view direction to the Z axis
    let r = rotation_matrix;
    // Create the 4x4 view matrix without translation component
    // The rotation part ensures the view direction is transformed to the Z axis
    array![
        [r[[0, 0]], r[[0, 1]], r[[0, 2]], 0.0],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]], 0.0],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]], 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]
}
