extern crate ndarray as nd;
extern crate num;
extern crate numpy;
extern crate pyo3;

use numpy::{PyArray3, PyArray4};
use numpy::{PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod projector;
mod utils;

#[pyfunction]
fn generate_projections<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f64>,
    triangles: PyReadonlyArray2<usize>,
    texture: PyReadonlyArray3<u8>,
    triangle_uvs: PyReadonlyArray2<f64>,
    precision: u32,
    filtering: u32,
    verbose: bool,
) -> (Bound<'py, PyArray4<u8>>, Bound<'py, PyArray3<f64>>) {
    if verbose {
        println!("Generating projections");
    }
    let points = utils::to_ndarray2(&points).unwrap();
    let triangles = utils::to_ndarray2(&triangles).unwrap().map(|&v| v as usize);
    let texture = utils::to_ndarray3(&texture).unwrap().map(|&v| v as u8);
    let triangle_uvs = utils::to_ndarray2(&triangle_uvs).unwrap();
    let (images, ocp_maps, freqs) = projector::compute_mesh_projections_textured(
        points,
        triangles,
        texture,
        triangle_uvs,
        precision,
        filtering,
    );
    let images = PyArray4::from_owned_array(py, images);
    let ocp_maps = PyArray3::from_owned_array(py, ocp_maps);
    if verbose {
        for i in 0..6 {
            println!("{} points removed from projection {}", &freqs[i], &i);
        }
    }
    (images, ocp_maps)
}

#[pymodule]
fn mesh_projector(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_projections, m)?)?;
    Ok(())
}
