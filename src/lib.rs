extern crate ndarray as nd;
extern crate num;
extern crate numpy;
extern crate pyo3;

use numpy::{PyArray3, PyArray4};
use numpy::{PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod projector;
mod utils;

#[pyfunction]
fn generate_projections<'py>(
    _py: Python<'py>,
    points: PyReadonlyArray2<f64>,
    triangles: PyReadonlyArray2<usize>,
    texture: PyReadonlyArray3<u8>,
    triangle_uvs: PyReadonlyArray2<f64>,
    precision: u32,
    filtering: u32,
    verbose: bool,
) -> (&'py PyArray4<u8>, &'py PyArray3<f64>) {
    if verbose {
        println!("Generating projections");
    }
    let points = utils::to_ndarray(&points);
    let triangles = utils::to_ndarray(&triangles).map(|&v| v as usize);
    let texture = utils::to_ndarray3(&texture).map(|&v| v as u8);
    let triangle_uvs = utils::to_ndarray(&triangle_uvs);
    let (images, ocp_maps, freqs) = projector::compute_mesh_projections_textured(
        points,
        triangles,
        texture,
        triangle_uvs,
        precision,
        filtering,
    );
    let (images, ocp_maps) = (
        images.to_owned().to_pyarray(_py),
        ocp_maps.to_owned().to_pyarray(_py),
    );
    if verbose {
        for i in 0..6 {
            println!("{} points removed from projection {}", &freqs[i], &i);
        }
    }
    (images, ocp_maps)
}

#[pymodule]
fn mesh_projector(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_projections, m)?)?;
    Ok(())
}
