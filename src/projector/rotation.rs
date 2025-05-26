use crate::projector::mesh::Mesh;
use ndarray::Array2;

// Function to rotate the mesh
pub fn rotate_mesh(mesh: &mut Mesh, angle_x: f64, angle_y: f64, angle_z: f64) {
    let rotation_matrix_x = rotation_matrix_x(angle_x);
    let rotation_matrix_y = rotation_matrix_y(angle_y);
    let rotation_matrix_z = rotation_matrix_z(angle_z);
    // Combine rotation matrices: R = Rz * Ry * Rx
    let rotation_matrix = rotation_matrix_z
        .dot(&rotation_matrix_y)
        .dot(&rotation_matrix_x);
    // Rotate vertices
    rotate_array(&mut mesh.geometry.vertices, &rotation_matrix);
    // Rotate vertex normals if they exist
    if let Some(ref mut normals) = mesh.geometry.vertex_normals {
        rotate_array(normals, &rotation_matrix);
    }
}

// Helper function to apply rotation to an Array2<f64>
fn rotate_array(array: &mut Array2<f64>, rotation_matrix: &Array2<f64>) {
    for mut row in array.rows_mut() {
        let point = row.to_owned().insert_axis(ndarray::Axis(1)); // Convert to column vector
        let rotated_point = rotation_matrix.dot(&point);
        for i in 0..3 {
            row[i] = rotated_point[[i, 0]];
        }
    }
}

// Functions to create rotation matrices
fn rotation_matrix_x(angle: f64) -> Array2<f64> {
    ndarray::arr2(&[
        [1.0, 0.0, 0.0],
        [0.0, angle.cos(), -angle.sin()],
        [0.0, angle.sin(), angle.cos()],
    ])
}

fn rotation_matrix_y(angle: f64) -> Array2<f64> {
    ndarray::arr2(&[
        [angle.cos(), 0.0, angle.sin()],
        [0.0, 1.0, 0.0],
        [-angle.sin(), 0.0, angle.cos()],
    ])
}

fn rotation_matrix_z(angle: f64) -> Array2<f64> {
    ndarray::arr2(&[
        [angle.cos(), -angle.sin(), 0.0],
        [angle.sin(), angle.cos(), 0.0],
        [0.0, 0.0, 1.0],
    ])
}
