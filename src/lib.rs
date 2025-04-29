extern crate colog;
extern crate log;
extern crate ndarray as nd;
extern crate num;
extern crate pyo3;
extern crate rayon;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use pyo3::wrap_pyfunction;

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
    vertex_normals,
    vertex_colors,
    texture_data,
    tex_height,
    tex_width,
    triangle_uvs,
    ka_list,
    kd_list,
    ks_list,
    ns,
    opacity,
    illum,
    precision,
    filtering_radius,
    mode_str,
))]
#[allow(clippy::too_many_arguments)]
fn generate_projections<'py>(
    py: Python<'py>,
    vertices: Vec<f32>,
    triangle_indices: Vec<u32>,
    vertex_normals: Option<Vec<f32>>,
    vertex_colors: Option<Vec<f32>>,
    texture_data: Option<Vec<u8>>,
    tex_height: usize,
    tex_width: usize,
    triangle_uvs: Option<Vec<f32>>,
    ka_list: &Bound<'py, PyList>,
    kd_list: &Bound<'py, PyList>,
    ks_list: &Bound<'py, PyList>,
    ns: f32,
    opacity: f32,
    illum: i32,
    precision: u32,
    filtering_radius: u32,
    mode_str: String,
) -> PyResult<Vec<PyObject>> {
    log::info!("Entering Rust generate_projections");

    let mode = ProjectionMode::from_str(&mode_str)?;
    log::info!("Projection Mode: {:?}", mode);

    let ka_vec: Vec<f32> = ka_list.extract()?;
    let kd_vec: Vec<f32> = kd_list.extract()?;
    let ks_vec: Vec<f32> = ks_list.extract()?;

    if ka_vec.len() != 3 || kd_vec.len() != 3 || ks_vec.len() != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Material colors (ka, kd, ks) must have 3 elements each",
        ));
    }

    let ka: [f32; 3] = [ka_vec[0], ka_vec[1], ka_vec[2]];
    let kd: [f32; 3] = [kd_vec[0], kd_vec[1], kd_vec[2]];
    let ks: [f32; 3] = [ks_vec[0], ks_vec[1], ks_vec[2]];
    log::info!(
        "Material Properties: Ka={:?}, Kd={:?}, Ks={:?}, Ns={}, Opacity={}, Illum={}",
        ka,
        kd,
        ks,
        ns,
        opacity,
        illum
    );

    let num_vertices = vertices.len() / 3;
    let vertices_view = nd::ArrayView::from_shape((num_vertices, 3), &vertices).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid vertices shape: {}", e))
    })?;

    let num_triangles = triangle_indices.len() / 3;
    let triangles_usize_array = nd::ArrayView::from_shape((num_triangles, 3), &triangle_indices)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid triangle_indices shape: {}",
                e
            ))
        })?
        .mapv(|x| x as usize);
    let triangles_view = triangles_usize_array.view();

    let vertex_normals_view = vertex_normals.as_ref().map(|data| {
        nd::ArrayView::from_shape((num_vertices, 3), data).expect("Invalid vertex_normals shape")
    });
    if vertex_normals_view.is_none() && mode != ProjectionMode::ObjOnly && illum > 0 {
        log::warn!("Vertex normals are None, but required for lighting in mode {:?} (illum={}). Shading may be incorrect.", mode, illum);
    }

    let vertex_colors_view = vertex_colors.as_ref().map(|data| {
        nd::ArrayView::from_shape((num_vertices, 3), data).expect("Invalid vertex_colors shape")
    });
    if mode == ProjectionMode::ObjOnly && vertex_colors_view.is_none() {
        log::warn!("Vertex colors are None for OBJ_ONLY mode. Default white will be used by the projector.");
    }

    let texture_view = texture_data.as_ref().map(|data| {
        nd::ArrayView::from_shape((tex_height, tex_width, 3), data)
            .expect("Invalid texture_data shape")
    });
    if (mode == ProjectionMode::Full || mode == ProjectionMode::ObjTexture)
        && texture_view.is_none()
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Texture data is required for mode {:?} but is None.",
            mode
        )));
    }

    let triangle_uvs_view = triangle_uvs.as_ref().map(|data| {
        nd::ArrayView::from_shape((num_triangles * 3, 2), data).expect("Invalid triangle_uvs shape")
    });
    if (mode == ProjectionMode::Full || mode == ProjectionMode::ObjTexture)
        && triangle_uvs_view.is_none()
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Triangle UVs are required for mode {:?} but are None.",
            mode
        )));
    }

    let (projection_results, outlier_counts) = py.allow_threads(|| {
        projector::compute_mesh_projections(
            vertices_view,
            triangles_view,
            vertex_normals_view,
            vertex_colors_view,
            texture_view,
            triangle_uvs_view,
            precision,
            ka,
            kd,
            ks,
            ns,
            opacity,
            illum,
            filtering_radius,
            mode,
        )
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

    let py_results: Vec<PyObject> = projection_results
        .axis_iter(nd::Axis(0))
        .map(|img_view| {
            let img_bytes = img_view.as_slice().expect("Image view is not contiguous");
            PyBytes::new(py, img_bytes).into()
        })
        .collect();

    Ok(py_results)
}

#[pymodule]
#[pyo3(name = "mesh_projector")]
fn mesh_projector_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_projections, m)?)?;
    Ok(())
}
