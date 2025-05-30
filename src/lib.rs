extern crate colog;
extern crate log;
extern crate ndarray as nd;
extern crate num;
extern crate numpy;
extern crate pyo3;
extern crate rayon;

use crate::projector::config::ProjectorConfig;
use numpy::{IntoPyArray, PyArray3, PyArray4};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::path::Path;

mod logging;
mod mesh_io;
mod projector;

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
    Py<PyArray4<u8>>,
    Py<PyArray4<u8>>,
    Py<PyArray3<u8>>,
)> {
    if verbose {
        logging::init_logger().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to initialize logger: {}",
                e
            ))
        })?;
    }
    log::info!("Entering Rust generate_projections backend");

    let rotation_steps: u32 = 64;
    let angle_delta_x: f64 = 0.0;
    let angle_delta_y: f64 = std::f64::consts::PI / 32.0;
    let angle_delta_z: f64 = 0.0;

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
    let settings = ProjectorConfig::new(
        precision,
        filtering_radius,
        rotation_steps,
        angle_delta_x,
        angle_delta_y,
        angle_delta_z,
    );

    // Call the backend to compute projections
    let (images, depth_maps, normal_maps, occlusion_maps, outlier_counts) =
        py.allow_threads(|| projector::compute_mesh_projections(&mesh, &settings));
    log::info!("Back-end projections generation complete!");

    // Log outlier filtering results if requested
    if filtering_radius > 0 {
        let total_removed: u64 = outlier_counts.iter().copied().sum();
        if total_removed > 0 {
            log::info!(
                "Outlier Filtering Removed Points: Total = {} ({:?})",
                total_removed,
                outlier_counts
            );
        } else {
            log::info!("Outlier Filtering: No points removed.");
        }
    }

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
    if verbose {
        logging::init_logger().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to initialize logger: {}",
                e
            ))
        })?;
    }
    log::info!("Entering Rust generate_projections backend");

    let rotation_steps: u32 = 32;
    let angle_delta_x: f64 = 0.0;
    let angle_delta_y: f64 = std::f64::consts::PI / 4.0;
    let angle_delta_z: f64 = 0.0;

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
    let settings = ProjectorConfig::new(
        precision,
        filtering_radius,
        rotation_steps,
        angle_delta_x,
        angle_delta_y,
        angle_delta_z,
    );

    // Call the backend to compute projections
    let output_dir = Path::new("projections");
    py.allow_threads(|| {
        projector::compute_and_save_rotation_projections(&mesh, &settings, &output_dir)
    });
    log::info!("Back-end projections generation complete!");
    Ok(())
}

#[pymodule]
fn mesh_projector(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_projections, m)?)?;
    m.add_function(wrap_pyfunction!(generate_projections_rotation, m)?)?;
    Ok(())
}
