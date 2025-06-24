extern crate colog;
extern crate log;
extern crate ndarray as nd;
extern crate num;
extern crate numpy;
extern crate once_cell;
extern crate pyo3;
extern crate rayon;

use crate::projector::config::{PointCloudConfig, ProjectionConfig, RotationConfig};
use crate::types::orientation::Orientation;
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArray4};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::path::Path;

mod logging;
mod mesh_io;
mod projector;
mod renderer;
mod types;

#[pyfunction]
fn generate_cube_projections<'py>(
    py: Python<'py>,
    obj_path: &str,
    texture_path: Option<&str>,
    mtl_path: Option<&str>,
    precision: u64,
    filtering_radius: u64,
    flip_uv: bool,
    ignore_opacity: bool,
    verbose: bool,
) -> PyResult<(
    Py<PyArray4<u8>>,
    Py<PyArray3<u8>>,
    Py<PyArray4<u8>>,
    Py<PyArray3<u8>>,
)> {
    // Initialize logger if requested
    if verbose {
        logging::init_logger().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to initialize logger: {}",
                e
            ))
        })?;
    }
    log::info!("Entering Rust generate_cube_projections backend");
    // Load and preprocess the mesh using io.rs (mode is automatically inferred)
    let mesh = match mesh_io::load_mesh_auto(
        obj_path,
        texture_path,
        mtl_path,
        precision as u32,
        flip_uv,
        ignore_opacity,
    ) {
        Ok(mesh) => mesh,
        Err(e) => {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load mesh: {}",
                e
            )));
        }
    };
    // Set up rotation config for cube projection (will be handled internally)
    let rotation_config = RotationConfig {
        angle_deltas: Orientation::from([0.0, 0.0, 0.0]),
        num_steps: 1,
    };
    // Set up projection settings for cube projection
    let settings = ProjectionConfig {
        precision,
        filtering_window_radius: filtering_radius,
        rotation_config,
    };
    // Call the backend to compute projections
    let image_batch = py.allow_threads(|| projector::compute_cube_projections(&mesh, &settings));
    let (projections, depth_maps, normal_maps, occlusion_maps) = (
        image_batch.images,
        image_batch.depth_maps,
        image_batch.normal_maps,
        image_batch.occupancy_maps,
    );
    log::info!("Cube projections generation complete!");
    // Convert the ndarray results directly to numpy arrays and return as a tuple
    Ok((
        projections.into_pyarray(py).into(),
        depth_maps.into_pyarray(py).into(),
        normal_maps.into_pyarray(py).into(),
        occlusion_maps.into_pyarray(py).into(),
    ))
}

#[pyfunction]
fn generate_projections<'py>(
    py: Python<'py>,
    obj_path: &str,
    texture_path: Option<&str>,
    mtl_path: Option<&str>,
    precision: u64,
    filtering_radius: u64,
    flip_uv: bool,
    ignore_opacity: bool,
    verbose: bool,
) -> PyResult<(
    Py<PyArray4<u8>>,
    Py<PyArray3<u8>>,
    Py<PyArray4<u8>>,
    Py<PyArray3<u8>>,
)> {
    // Initialize logger if requested
    if verbose {
        logging::init_logger().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to initialize logger: {}",
                e
            ))
        })?;
    }
    log::info!("Entering Rust generate_projections backend");
    // Set up projection settings for rotation-based projection
    let rotation_steps: u32 = 64;
    let angle_delta_x: f32 = 0.0;
    let angle_delta_y: f32 = std::f32::consts::PI / 32.0;
    let angle_delta_z: f32 = 0.0;
    // Load and preprocess the mesh using io.rs (mode is automatically inferred)
    let mesh = match mesh_io::load_mesh_auto(
        obj_path,
        texture_path,
        mtl_path,
        precision as u32,
        flip_uv,
        ignore_opacity,
    ) {
        Ok(mesh) => mesh,
        Err(e) => {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load mesh: {}",
                e
            )));
        }
    };
    // Set up projection settings with rotation config if provided
    log::info!(
        "Using rotation-based projection with {} steps",
        rotation_steps
    );
    log::info!(
        "Rotation deltas: X={}, Y={}, Z={} radians",
        angle_delta_x,
        angle_delta_y,
        angle_delta_z
    );
    // Create rotation config from the deltas
    let rotation_config = RotationConfig {
        angle_deltas: Orientation::new(angle_delta_x, angle_delta_y, angle_delta_z),
        num_steps: rotation_steps as usize,
    };
    // Create settings for the projection
    let settings = ProjectionConfig {
        precision,
        filtering_window_radius: filtering_radius,
        rotation_config,
    };
    // Call the backend to compute projections
    let image_batch = py.allow_threads(|| projector::compute_mesh_projections(&mesh, &settings));
    let (images, depth_maps, normal_maps, occlusion_maps) = (
        image_batch.images,
        image_batch.depth_maps,
        image_batch.normal_maps,
        image_batch.occupancy_maps,
    );
    log::info!("Back-end projections generation complete!");
    // Convert the ndarray results directly to numpy arrays and return as a tuple
    Ok((
        images.into_pyarray(py).into(),
        depth_maps.into_pyarray(py).into(),
        normal_maps.into_pyarray(py).into(),
        occlusion_maps.into_pyarray(py).into(),
    ))
}

#[pyfunction]
fn generate_projections_rotation<'py>(
    py: Python<'py>,
    obj_path: &str,
    texture_path: Option<&str>,
    mtl_path: Option<&str>,
    precision: u64,
    filtering_radius: u64,
    flip_uv: bool,
    ignore_opacity: bool,
    verbose: bool,
) -> PyResult<()> {
    // Initialize logger if requested
    if verbose {
        logging::init_logger().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to initialize logger: {}",
                e
            ))
        })?;
    }
    log::info!("Entering Rust generate_projections backend");
    // Set up projection settings for rotation-based projection
    let rotation_steps: u32 = 32;
    let angle_delta_x: f32 = 0.0;
    let angle_delta_y: f32 = std::f32::consts::PI / 4.0;
    let angle_delta_z: f32 = 0.0;
    // Load and preprocess the mesh using io.rs (mode is automatically inferred)
    let mesh = match mesh_io::load_mesh_auto(
        obj_path,
        texture_path,
        mtl_path,
        precision as u32,
        flip_uv,
        ignore_opacity,
    ) {
        Ok(mesh) => mesh,
        Err(e) => {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load mesh: {}",
                e
            )));
        }
    };
    // Set up projection settings for rotation-based projection
    log::info!(
        "Using rotation-based projection with {} steps",
        rotation_steps
    );
    log::info!(
        "Rotation deltas: X={}, Y={}, Z={} radians",
        angle_delta_x,
        angle_delta_y,
        angle_delta_z
    );
    // Create rotation config from the deltas
    let rotation_config = RotationConfig {
        angle_deltas: Orientation::new(angle_delta_x, angle_delta_y, angle_delta_z),
        num_steps: rotation_steps as usize,
    };
    let settings = ProjectionConfig {
        precision,
        filtering_window_radius: filtering_radius,
        rotation_config,
    };
    // Call the backend to compute projections
    let output_dir = Path::new("projections");
    py.allow_threads(|| {
        projector::compute_and_save_rotation_projections(&mesh, &settings, &output_dir)
    });
    log::info!("Back-end projections generation complete!");
    Ok(())
}

#[pyfunction]
fn generate_point_cloud<'py>(
    py: Python<'py>,
    obj_path: &str,
    texture_path: Option<&str>,
    mtl_path: Option<&str>,
    precision: u64,
    flip_uv: bool,
    point_cloud_density_factor: f32,
    ignore_opacity: bool,
    verbose: bool,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<u8>>)> {
    // Initialize logger if requested
    if verbose {
        logging::init_logger().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to initialize logger: {}",
                e
            ))
        })?;
    }
    log::info!("Entering Rust generate_point_cloud backend");
    // Load and preprocess the mesh using io.rs (mode is automatically inferred)
    let mesh = match mesh_io::load_mesh_auto(
        obj_path,
        texture_path,
        mtl_path,
        precision as u32,
        flip_uv,
        ignore_opacity,
    ) {
        Ok(mesh) => mesh,
        Err(e) => {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load mesh: {}",
                e
            )));
        }
    };
    // Set up point cloud settings
    let settings = PointCloudConfig {
        precision,
        point_cloud_density_factor,
    };
    // Call the backend to generate the point cloud
    let (points, colors) = py.allow_threads(|| projector::mesh_to_point_cloud(&mesh, &settings));
    log::info!(
        "Back-end point cloud generation complete! Generated {} points",
        points.shape()[0]
    );
    // Convert the ndarray results directly to numpy arrays and return as a tuple
    Ok((
        points.into_pyarray(py).into(),
        colors.into_pyarray(py).into(),
    ))
}

#[pymodule]
fn mesh_projector(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_cube_projections, m)?)?;
    m.add_function(wrap_pyfunction!(generate_projections, m)?)?;
    m.add_function(wrap_pyfunction!(generate_projections_rotation, m)?)?;
    m.add_function(wrap_pyfunction!(generate_point_cloud, m)?)?;
    Ok(())
}
