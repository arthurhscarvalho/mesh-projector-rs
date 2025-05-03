extern crate colog;
extern crate log;
extern crate ndarray as nd;
extern crate num;
extern crate numpy;
extern crate pyo3;
extern crate rayon;

use numpy::{IntoPyArray, PyArray4};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::projector::mesh::{Mesh, PyMesh};
use crate::projector::projection::{ProjectionMode, ProjectionSettings};

mod logging;
mod projector;
mod utils;

#[pyfunction]
fn generate_projections<'py>(
    py: Python<'py>,
    mesh: PyMesh<'py>,
    precision: u64,
    filtering_radius: u64,
    mode_str: &str,
    verbose: bool,
) -> PyResult<(Py<PyArray4<u8>>, Py<PyArray4<u8>>)> {
    if verbose {
        logging::init_logger().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to initialize logger: {}",
                e
            ))
        })?;
    }
    // Initial setup
    log::info!("Entering Rust generate_projections backend");
    let mesh: Mesh = mesh.into();
    let mode = ProjectionMode::from_str(mode_str)?;
    let settings = ProjectionSettings::new(precision, filtering_radius, mode);
    log::info!("Projection Mode: {:?}", mode);
    // Validate based on mode
    if mode == ProjectionMode::Full || mode == ProjectionMode::ObjTexture {
        if mesh.texture.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Texture data is required for mode {:?} but is None.",
                mode
            )));
        }
        if mesh.triangle_uvs.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Triangle UVs are required for mode {:?} but are None.",
                mode
            )));
        }
    }
    // Log material properties
    log::info!(
        "Material Properties: Ka={:?}, Kd={:?}, Ks={:?}, Ns={}, Opacity={}, Illum={}",
        mesh.ka,
        mesh.kd,
        mesh.ks,
        mesh.ns,
        mesh.opacity,
        mesh.illum
    );
    // Call the backend to compute projections
    let (projection_results, occlusion_maps, outlier_counts) =
        py.allow_threads(|| projector::compute_mesh_projections(&mesh, &settings));
    log::info!("Back-end projections generation complete!");
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
        projection_results.into_pyarray(py).into(),
        occlusion_maps.into_pyarray(py).into(),
    ))
}

#[pymodule]
fn mesh_projector(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_projections, m)?)?;
    Ok(())
}
