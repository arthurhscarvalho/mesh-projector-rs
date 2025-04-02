use nd::{Array2, Array3};
use num::ToPrimitive;
use numpy::{PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::PyErr;

pub fn to_ndarray2<T>(pyarray: &PyReadonlyArray2<T>) -> Result<Array2<f32>, PyErr>
where
    T: numpy::Element + ToPrimitive,
{
    let shape = pyarray.shape();
    let v: Vec<f32> = pyarray
        .as_array()
        .iter()
        .filter_map(|x| x.to_f32())
        .collect();
    Array2::from_shape_vec((shape[0], shape[1]), v)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid shape"))
}

pub fn to_ndarray3<T>(pyarray: &PyReadonlyArray3<T>) -> Result<Array3<f32>, PyErr>
where
    T: numpy::Element + ToPrimitive,
{
    let shape = pyarray.shape();
    let v: Vec<f32> = pyarray
        .as_array()
        .iter()
        .filter_map(|x| x.to_f32())
        .collect();
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), v)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid shape"))
}
