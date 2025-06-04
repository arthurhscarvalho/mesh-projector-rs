extern crate colog;
extern crate log;
extern crate ndarray as nd;
extern crate num;
extern crate numpy;
extern crate pyo3;
extern crate rayon;

use crate::projector::config::ProjectorConfig;
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArray4};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
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

    let rotation_config = [angle_delta_x, angle_delta_y, angle_delta_z];
    let settings = ProjectorConfig::new(
        precision,
        filtering_radius,
        rotation_steps,
        rotation_config,
        None,
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
fn generate_cube_projections<'py>(
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
    log::info!("Entering Rust generate_cube_projections backend");

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

    // Define the 6 cube face camera positions and orientations
    // Each tuple is (camera_x, camera_y, camera_z, rotation_x, rotation_y, rotation_z, label)
    // Rotations are in radians. Y is pitch, X is roll, Z is yaw.
    let pi = std::f64::consts::PI;
    let cube_faces = vec![
        (0.0, 0.0, 0.0, "front"),      // Front: Cam at +Z, looking -Z. No rotation.
        (0.0, pi, 0.0, "back"),        // Back: Cam at -Z, looking +Z. Rotate 180 deg around Y.
        (0.0, pi / 2.0, 0.0, "right"), // Right: Cam at +X, looking -X. Rotate 90 deg around Y.
        (0.0, -pi / 2.0, 0.0, "left"), // Left: Cam at -X, looking +X. Rotate -90 deg around Y.
        (pi / 2.0, 0.0, 0.0, "top"),   // Top: Cam at +Y, looking -Y. Rotate 90 deg around X.
        (-pi / 2.0, 0.0, 0.0, "bottom"), // Bottom: Cam at -Y, looking +Y. Rotate -90 deg around X.
    ];

    // Use rayon to compute all 6 projections in parallel
    let results: Vec<_> = cube_faces
        .into_par_iter()
        .map(|(rot_x, rot_y, rot_z, label)| {
            log::info!("Computing {} face projection", label);

            // Set camera orientation and position for this face
            let rotation_config_for_face = [rot_x, rot_y, rot_z];
            let settings = ProjectorConfig::new(
                precision,
                filtering_radius,
                1, // Single step for each face
                rotation_config_for_face,
                None,
            );

            // Compute projection for this face
            let (images, depth_maps, normal_maps, occlusion_maps, _outlier_counts) =
                projector::compute_mesh_projections(&mesh, &settings);

            (images, depth_maps, normal_maps, occlusion_maps)
        })
        .collect();

    // Combine all results into single arrays with 6 views
    let max_bound: u64 = 1 << precision;
    let max_bound_u: usize = max_bound as usize;
    let rows = max_bound_u;
    let cols = max_bound_u;
    const NUM_CHANNELS: usize = 3;

    let mut combined_images = ndarray::Array4::<u8>::zeros((6, rows, cols, NUM_CHANNELS));
    let mut combined_depth_maps = ndarray::Array4::<u8>::zeros((6, rows, cols, NUM_CHANNELS));
    let mut combined_normal_maps = ndarray::Array4::<u8>::zeros((6, rows, cols, NUM_CHANNELS));
    let mut combined_occlusion_maps = ndarray::Array3::<u8>::zeros((6, rows, cols));

    // Copy results from each face into the combined arrays
    for (i, (images, depth_maps, normal_maps, occlusion_maps)) in results.into_iter().enumerate() {
        // Each result has shape (1, rows, cols, channels) since we used 1 step
        combined_images
            .slice_mut(ndarray::s![i, .., .., ..])
            .assign(&images.slice(ndarray::s![0, .., .., ..]));
        combined_depth_maps
            .slice_mut(ndarray::s![i, .., .., ..])
            .assign(&depth_maps.slice(ndarray::s![0, .., .., ..]));
        combined_normal_maps
            .slice_mut(ndarray::s![i, .., .., ..])
            .assign(&normal_maps.slice(ndarray::s![0, .., .., ..]));
        combined_occlusion_maps
            .slice_mut(ndarray::s![i, .., ..])
            .assign(&occlusion_maps.slice(ndarray::s![0, .., ..]));
    }

    log::info!("Cube projections generation complete!");

    // Convert the ndarray results directly to numpy arrays and return as a tuple
    Ok((
        combined_images.into_pyarray(py).into(),
        combined_depth_maps.into_pyarray(py).into(),
        combined_normal_maps.into_pyarray(py).into(),
        combined_occlusion_maps.into_pyarray(py).into(),
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
    let rotation_config = [angle_delta_x, angle_delta_y, angle_delta_z];
    let settings = ProjectorConfig::new(
        precision,
        filtering_radius,
        rotation_steps,
        rotation_config,
        None,
    );

    // Call the backend to compute projections
    let output_dir = Path::new("projections");
    py.allow_threads(|| {
        projector::compute_and_save_rotation_projections(&mesh, &settings, &output_dir)
    });
    log::info!("Back-end projections generation complete!");
    Ok(())
}

#[pyfunction]
fn generate_point_cloud<'py>(
    py: Python<'py>,
    obj_path: &str,
    texture_path: Option<&str>,
    mtl_path: Option<&str>,
    precision: u64,
    filtering_radius: u64,
    flip_uv: bool,
    point_cloud_density_factor: f64,
    ignore_opacity: bool,
    verbose: bool,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<u8>>)> {
    if verbose {
        logging::init_logger().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to initialize logger: {}",
                e
            ))
        })?;
    }
    log::info!("Entering Rust generate_point_cloud backend");

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

    // Set up projection settings
    // Note: rotation parameters aren't used for point cloud generation,
    // but we need to provide them for the ProjectorConfig
    let rotation_steps: u32 = 1; // Only need one step for point cloud
    let angle_delta_x: f64 = 0.0;
    let angle_delta_y: f64 = 0.0;
    let angle_delta_z: f64 = 0.0;
    let rotation_config = [angle_delta_x, angle_delta_y, angle_delta_z];
    let settings = ProjectorConfig::new(
        precision,
        filtering_radius,
        rotation_steps,
        rotation_config,
        Some(point_cloud_density_factor),
    );

    // Call the backend to generate the point cloud
    let (points, colors) =
        py.allow_threads(|| projector::discretize_mesh_to_point_cloud(&mesh, &settings));
    log::info!(
        "Back-end point cloud generation complete! Generated {} points",
        points.shape()[0]
    );

    // Convert the ndarray results directly to numpy arrays and return as a tuple
    Ok((
        points.into_pyarray(py).into(),
        colors.into_pyarray(py).into(),
    ))
}

#[pymodule]
fn mesh_projector(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_projections, m)?)?;
    m.add_function(wrap_pyfunction!(generate_cube_projections, m)?)?;
    m.add_function(wrap_pyfunction!(generate_projections_rotation, m)?)?;
    m.add_function(wrap_pyfunction!(generate_point_cloud, m)?)?;
    Ok(())
}
