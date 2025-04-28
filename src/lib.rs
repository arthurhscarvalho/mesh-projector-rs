extern crate colog;
extern crate log;
extern crate ndarray as nd;
extern crate num;
extern crate numpy;
extern crate pyo3;
extern crate rayon;

use numpy::{PyArray3, PyArray4};
use numpy::{PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::wrap_pyfunction;

mod logging;
mod projector;
mod utils;

#[pyfunction]
#[pyo3(signature = (
    points,
    triangles,
    vertex_normals,
    texture,
    triangle_uvs,
    precision,
    ka,
    kd,
    ks,
    ns,
    opacity,
    illum,
    filtering,
    logging_enabled
))]
fn generate_projections<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
    triangles: PyReadonlyArray2<u64>,
    vertex_normals: PyReadonlyArray2<f32>,
    texture: PyReadonlyArray3<u8>,
    triangle_uvs: PyReadonlyArray2<f32>,
    precision: u32,
    ka: &Bound<'py, PyList>,
    kd: &Bound<'py, PyList>,
    ks: &Bound<'py, PyList>,
    ns: f32,
    opacity: f32,
    illum: u32,
    filtering: u32,
    logging_enabled: bool,
) -> PyResult<(Bound<'py, PyArray4<u8>>, Bound<'py, PyArray3<f32>>)> {
    conditional_log!(
        logging_enabled,
        info,
        "Back-end projector starting (with vertex normals)"
    );

    let ka_vec: Vec<f32> = ka.extract()?;
    let kd_vec: Vec<f32> = kd.extract()?;
    let ks_vec: Vec<f32> = ks.extract()?;

    if ka_vec.len() != 3 || kd_vec.len() != 3 || ks_vec.len() != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Material colors (ka, kd, ks) must have 3 elements each",
        ));
    }

    let ka_arr: [f32; 3] = [ka_vec[0], ka_vec[1], ka_vec[2]];
    let kd_arr: [f32; 3] = [kd_vec[0], kd_vec[1], kd_vec[2]];
    let ks_arr: [f32; 3] = [ks_vec[0], ks_vec[1], ks_vec[2]];

    let points = utils::to_ndarray2(&points).unwrap();
    let triangles_u64 = utils::to_ndarray2(&triangles).unwrap();
    let triangles_usize = triangles_u64.mapv(|x| x as usize);
    let vertex_normals = utils::to_ndarray2(&vertex_normals).unwrap();
    let texture = utils::to_ndarray3(&texture).unwrap().map(|&v| v as u8);
    let triangle_uvs = utils::to_ndarray2(&triangle_uvs).unwrap();

    let (images, ocp_maps, freqs) = py.allow_threads(|| {
        projector::compute_mesh_projections(
            points.view(),
            triangles_usize.view(),
            vertex_normals.view(),
            texture.view(),
            triangle_uvs.view(),
            precision,
            ka_arr,
            kd_arr,
            ks_arr,
            ns,
            opacity,
            illum,
            filtering,
            logging_enabled,
        )
    });

    conditional_log!(
        logging_enabled,
        info,
        "Back-end projections generation complete!"
    );
    let images = PyArray4::from_owned_array(py, images);
    let ocp_maps = PyArray3::from_owned_array(py, ocp_maps);
    log_removed_points!(logging_enabled && filtering > 0, freqs);
    Ok((images, ocp_maps))
}

#[pymodule]
fn mesh_projector(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_projections, m)?)?;
    Ok(())
}
