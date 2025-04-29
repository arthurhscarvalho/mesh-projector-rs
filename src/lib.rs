extern crate colog;
extern crate log;
extern crate ndarray as nd;
extern crate num;
extern crate numpy;
extern crate pyo3;
extern crate rayon;

use numpy::{IntoPyArray, PyArray4, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::wrap_pyfunction;

use crate::projector::material::MaterialProperties;
use crate::projector::mesh::{Mesh, MeshAttributes};
use crate::projector::projection::ProjectionSettings;

mod logging;
mod projector;
mod utils;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProjectionMode {
    ObjOnly,
    ObjTexture,
    Full,
}

impl ProjectionMode {
    fn from_str(s: &str) -> PyResult<Self> {
        match s {
            "obj_only" => Ok(ProjectionMode::ObjOnly),
            "obj_texture" => Ok(ProjectionMode::ObjTexture),
            "full" => Ok(ProjectionMode::Full),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid projection mode: {}",
                s
            ))),
        }
    }
}

#[pyfunction]
#[pyo3(signature = (
    vertices,
    triangle_indices,
    vertex_normals=None,
    vertex_colors=None,
    texture_data=None,
    triangle_uvs=None,
    ka_list=None,
    kd_list=None,
    ks_list=None,
    ns=0.0,
    opacity=1.0,
    illum=0,
    precision=6,
    filtering_radius=0,
    mode_str="obj_only",
))]
fn generate_projections<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<f32>,
    triangle_indices: PyReadonlyArray2<u32>,
    vertex_normals: Option<PyReadonlyArray2<f32>>,
    vertex_colors: Option<PyReadonlyArray2<f32>>,
    texture_data: Option<PyReadonlyArray3<u8>>,
    triangle_uvs: Option<PyReadonlyArray2<f32>>,
    ka_list: Option<&Bound<'py, PyList>>,
    kd_list: Option<&Bound<'py, PyList>>,
    ks_list: Option<&Bound<'py, PyList>>,
    ns: f32,
    opacity: f32,
    illum: i32,
    precision: u32,
    filtering_radius: u32,
    mode_str: &str,
) -> PyResult<Py<PyArray4<u8>>> {
    log::info!("Entering Rust generate_projections");

    let mode = ProjectionMode::from_str(mode_str)?;
    log::info!("Projection Mode: {:?}", mode);

    // Extract material colors
    let ka = extract_color_array(ka_list, &[0.2, 0.2, 0.2])?;
    let kd = extract_color_array(kd_list, &[0.8, 0.8, 0.8])?;
    let ks = extract_color_array(ks_list, &[1.0, 1.0, 1.0])?;

    log::info!(
        "Material Properties: Ka={:?}, Kd={:?}, Ks={:?}, Ns={}, Opacity={}, Illum={}",
        ka,
        kd,
        ks,
        ns,
        opacity,
        illum
    );

    // Create material properties
    let material = MaterialProperties {
        ka,
        kd,
        ks,
        ns,
        opacity,
        illum,
    };

    // Create projection settings
    let settings = ProjectionSettings::new(precision, filtering_radius, mode);

    // Get array views from numpy arrays
    let vertices_array = vertices.as_array();
    let triangles_array = triangle_indices.as_array().mapv(|x| x as usize);

    // Get views for optional arrays safely
    let vertex_normals_view = vertex_normals.as_ref().map(|arr| arr.as_array());
    let vertex_colors_view = vertex_colors.as_ref().map(|arr| arr.as_array());
    let texture_view = texture_data.as_ref().map(|arr| arr.as_array());
    let triangle_uvs_view = triangle_uvs.as_ref().map(|arr| arr.as_array());

    // Validate based on mode
    if (mode == ProjectionMode::Full || mode == ProjectionMode::ObjTexture)
        && texture_view.is_none()
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Texture data is required for mode {:?} but is None.",
            mode
        )));
    }

    if (mode == ProjectionMode::Full || mode == ProjectionMode::ObjTexture)
        && triangle_uvs_view.is_none()
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Triangle UVs are required for mode {:?} but are None.",
            mode
        )));
    }

    // Call the backend to compute projections
    let (projection_results, outlier_counts) = py.allow_threads(|| {
        // Create mesh and mesh attributes structures
        let mesh = Mesh::new(vertices_array, triangles_array.view());
        let attributes = MeshAttributes::new(
            vertex_normals_view,
            vertex_colors_view,
            texture_view,
            triangle_uvs_view,
        );

        projector::compute_mesh_projections(&mesh, &attributes, &material, &settings)
    });

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

    // Convert the ndarray result directly to a numpy array
    Ok(projection_results.into_pyarray(py).into())
}

/// Helper function to extract color arrays from Python lists or use defaults
fn extract_color_array(
    color_list: Option<&Bound<PyList>>,
    defaults: &[f32; 3],
) -> PyResult<[f32; 3]> {
    if let Some(list) = color_list {
        let vec: Vec<f32> = list.extract()?;
        if vec.len() != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Material colors must have 3 elements each",
            ));
        }
        Ok([vec[0], vec[1], vec[2]])
    } else {
        Ok(*defaults)
    }
}

#[pymodule]
fn mesh_projector(m: &Bound<'_, PyModule>) -> PyResult<()> {
    logging::init_logger().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to initialize logger: {}",
            e
        ))
    })?;
    log::info!("Initializing mesh_projector Rust extension module");

    m.add_function(wrap_pyfunction!(generate_projections, m)?)?;

    Ok(())
}
