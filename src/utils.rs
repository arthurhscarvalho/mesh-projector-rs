use nd::{Array2, Array3};
use num::ToPrimitive;
use numpy::{PyReadonlyArray2, PyReadonlyArray3};

pub fn to_ndarray2<'a, T>(pyarray: &'a PyReadonlyArray2<T>) -> Array2<f64>
where
    T: numpy::Element + ToPrimitive,
{
    let v: Vec<f64> = pyarray
        .as_array()
        .iter()
        .map(|x| x.to_f64().unwrap())
        .collect();
    let shape = (pyarray.shape()[0], pyarray.shape()[1]);
    Array2::from_shape_vec(shape, v).unwrap()
}

pub fn to_ndarray3<'a, T>(pyarray: &'a PyReadonlyArray3<T>) -> Array3<f64>
where
    T: numpy::Element + ToPrimitive,
{
    let v: Vec<f64> = pyarray
        .as_array()
        .iter()
        .map(|x| x.to_f64().unwrap())
        .collect();
    let shape = (pyarray.shape()[0], pyarray.shape()[1], pyarray.shape()[2]);
    Array3::from_shape_vec(shape, v).unwrap()
}
