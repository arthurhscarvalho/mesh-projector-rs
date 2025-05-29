extern crate colog;
extern crate log;
extern crate ndarray as nd;
extern crate num;
extern crate numpy;
extern crate pyo3;
extern crate rayon;

use crate::projector::config::ProjectorConfig;
use crate::projector::io;
use numpy::{IntoPyArray, PyArray3, PyArray4};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::path::Path;

mod logging;
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

    let rotation_steps: Option<u32> = Some(64);
    let angle_delta_x: Option<f64> = Some(0.0);
    let angle_delta_y: Option<f64> = Some(std::f64::consts::PI / 32.0);
    let angle_delta_z: Option<f64> = Some(0.0);

    // Load and preprocess the mesh using io.rs (mode is automatically inferred)
    let mesh = match io::load_mesh(
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
    // Log material properties - REMOVED as material is now per-submesh
    // log::info!(
    //     "Material Properties: Ka={:?}, Kd={:?}, Ks={:?}, Ns={}, Opacity={}, Illum={}",
    //     mesh.material.ka,
    //     mesh.material.kd,
    //     mesh.material.ks,
    //     mesh.material.ns,
    //     mesh.material.opacity,
    //     mesh.material.illum
    // );

    // Set up projection settings with rotation config if provided
    let settings = if let (Some(steps), Some(dx), Some(dy), Some(dz)) =
        (rotation_steps, angle_delta_x, angle_delta_y, angle_delta_z)
    {
        log::info!("Using rotation-based projection with {} steps", steps);
        log::info!("Rotation deltas: X={}, Y={}, Z={} radians", dx, dy, dz);
        ProjectorConfig::new_with_rotation(precision, filtering_radius, steps, dx, dy, dz)
    } else {
        log::info!("Using standard 6-view orthographic projection");
        ProjectorConfig::new(precision, filtering_radius)
    };

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

    let rotation_steps: Option<u32> = Some(32);
    let angle_delta_x: Option<f64> = Some(std::f64::consts::PI / 4.0);
    let angle_delta_y: Option<f64> = Some(std::f64::consts::PI / 4.0);
    let angle_delta_z: Option<f64> = Some(0.0);

    // Load and preprocess the mesh using io.rs (mode is automatically inferred)
    let mesh = match io::load_mesh(
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
    let settings = if let (Some(steps), Some(dx), Some(dy), Some(dz)) =
        (rotation_steps, angle_delta_x, angle_delta_y, angle_delta_z)
    {
        log::info!("Using rotation-based projection with {} steps", steps);
        log::info!("Rotation deltas: X={}, Y={}, Z={} radians", dx, dy, dz);
        ProjectorConfig::new_with_rotation(precision, filtering_radius, steps, dx, dy, dz)
    } else {
        log::info!("Using standard 6-view orthographic projection");
        ProjectorConfig::new(precision, filtering_radius)
    };

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
