extern crate colog;
extern crate log;
extern crate ndarray as nd;
extern crate num;
extern crate numpy;
extern crate once_cell;
extern crate pyo3;
extern crate rayon;

use crate::projector::config::{PointCloudConfig, ProjectionConfig, RotationConfig};
use crate::types::mesh::ShadingMode;
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

#[pyclass]
pub struct MeshWrapper {
    pub mesh: crate::types::mesh::Scene<'static>,
    pub precision: u64,
}

#[pyfunction]
fn read_mesh<'py>(
    _py: Python<'py>,
    obj_path: &str,
    mesh_mode: Option<&str>,
    precision: u64,
    flip_uv: bool,
    ignore_opacity: bool,
    verbose: bool,
) -> PyResult<MeshWrapper> {
    // Initialize logger if requested
    logging::atomic_init_logger(verbose)?;
    log::info!("Entering Rust read_mesh backend");
    // Map optional string to ShadingMode if provided
    let mode_opt: Option<ShadingMode> = mesh_mode.and_then(|s| match s {
        "vertex_color" => Some(ShadingMode::VertexColor),
        "texture_mapped" => Some(ShadingMode::TextureMapped),
        "blinn_phong" => Some(ShadingMode::BlinnPhong),
        _ => None,
    });
    let mesh = match mesh_io::read_mesh(
        obj_path,
        mode_opt,
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
    log::info!("Scene loaded successfully");
    Ok(MeshWrapper { mesh, precision })
}

#[pyfunction]
fn project_to_image_on_bounding_cube<'py>(
    py: Python<'py>,
    mesh_wrapper: &MeshWrapper,
    filtering_radius: u64,
    verbose: bool,
) -> PyResult<(
    Py<PyArray4<u8>>,
    Py<PyArray3<u8>>,
    Py<PyArray4<u8>>,
    Py<PyArray3<u8>>,
)> {
    // Initialize logger if requested
    logging::atomic_init_logger(verbose)?;
    log::info!("Entering Rust project_to_image_on_bounding_cube backend");
    let mesh = &mesh_wrapper.mesh;
    let precision = mesh_wrapper.precision;
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
    let image_batch =
        py.allow_threads(|| projector::project_to_image_on_bounding_cube(&mesh, &settings));
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
fn project_to_image_by_rotation<'py>(
    py: Python<'py>,
    mesh_wrapper: &MeshWrapper,
    rotation_steps: u32,
    angle_delta_x: f32,
    angle_delta_y: f32,
    angle_delta_z: f32,
    filtering_radius: u64,
    verbose: bool,
) -> PyResult<(
    Py<PyArray4<u8>>,
    Py<PyArray3<u8>>,
    Py<PyArray4<u8>>,
    Py<PyArray3<u8>>,
)> {
    // Initialize logger if requested
    logging::atomic_init_logger(verbose)?;
    log::info!("Entering Rust project_to_image_by_rotation backend");
    let mesh = &mesh_wrapper.mesh;
    let precision = mesh_wrapper.precision;
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
    let image_batch =
        py.allow_threads(|| projector::project_to_image_by_rotation(&mesh, &settings));
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
fn project_to_image_by_rotation_to_file<'py>(
    py: Python<'py>,
    mesh_wrapper: &MeshWrapper,
    rotation_steps: u32,
    angle_delta_x: f32,
    angle_delta_y: f32,
    angle_delta_z: f32,
    filtering_radius: u64,
    verbose: bool,
) -> PyResult<()> {
    // Initialize logger if requested
    logging::atomic_init_logger(verbose)?;
    log::info!("Entering Rust project_to_image_by_rotation_to_file backend");
    let mesh = &mesh_wrapper.mesh;
    let precision = mesh_wrapper.precision;
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
    let output_dir = Path::new("rotation_projections");
    py.allow_threads(|| {
        projector::project_to_image_by_rotation_to_file(&mesh, &settings, &output_dir)
    });
    log::info!("Back-end projections generation complete!");
    Ok(())
}

#[pyfunction]
fn project_to_point_cloud_by_sampling<'py>(
    py: Python<'py>,
    mesh_wrapper: &MeshWrapper,
    point_cloud_density_factor: f32,
    verbose: bool,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<u8>>)> {
    // Initialize logger if requested
    logging::atomic_init_logger(verbose)?;
    log::info!("Entering Rust project_to_point_cloud_by_sampling backend");
    let mesh = &mesh_wrapper.mesh;
    let precision = mesh_wrapper.precision;
    // Set up point cloud settings
    let settings = PointCloudConfig {
        precision,
        point_cloud_density_factor,
    };
    // Call the backend to generate the point cloud
    let (points, colors) =
        py.allow_threads(|| projector::project_to_point_cloud_by_sampling(&mesh, &settings));
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
    m.add_class::<MeshWrapper>()?;
    m.add_function(wrap_pyfunction!(read_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(project_to_image_on_bounding_cube, m)?)?;
    m.add_function(wrap_pyfunction!(project_to_image_by_rotation, m)?)?;
    m.add_function(wrap_pyfunction!(project_to_image_by_rotation_to_file, m)?)?;
    m.add_function(wrap_pyfunction!(project_to_point_cloud_by_sampling, m)?)?;
    Ok(())
}
