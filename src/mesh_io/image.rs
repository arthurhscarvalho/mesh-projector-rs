//! Image saving operations.

use crate::mesh_io::errors::{MeshError, MeshResult};
use image::{ImageBuffer, Rgb};
use log::warn;
use ndarray::Array3;
use std::fs;
use std::path::Path;

/// Saves projection data for a single view to specified files
pub fn save_image_data(image: &Array3<u8>, output_dir: &Path, view_index: usize) -> MeshResult<()> {
    create_output_directory(output_dir)?;
    let (rows, cols) = (image.shape()[0], image.shape()[1]);
    let channels = image.shape()[2];
    if channels == 3 || channels == 4 {
        save_rgb_image(image, output_dir, view_index, rows, cols)?;
    } else {
        warn!(
            "Image array has unsupported number of channels: {}. Skipping save.",
            channels
        );
    }
    Ok(())
}

/// Create output directory if it doesn't exist
fn create_output_directory(output_dir: &Path) -> MeshResult<()> {
    fs::create_dir_all(output_dir).map_err(|e| {
        MeshError::DirectoryCreateError(format!(
            "Failed to create output directory {:?}: {}",
            output_dir, e
        ))
    })
}

/// Save RGB image data to file
fn save_rgb_image(
    image: &Array3<u8>,
    output_dir: &Path,
    view_index: usize,
    rows: usize,
    cols: usize,
) -> MeshResult<()> {
    let image_data = image.clone();
    let buffer = create_image_buffer(image_data, cols, rows)?;
    let image_path = output_dir.join(format!("image_{:04}.png", view_index));
    save_image_buffer(buffer, &image_path)
}

/// Create image buffer from array data
fn create_image_buffer(
    image_data: Array3<u8>,
    cols: usize,
    rows: usize,
) -> MeshResult<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    ImageBuffer::<Rgb<u8>, _>::from_raw(
        cols as u32,
        rows as u32,
        image_data.into_raw_vec_and_offset().0,
    )
    .ok_or_else(|| MeshError::ImageBufferError("Failed to create image buffer".to_string()))
}

/// Save image buffer to file
fn save_image_buffer(buffer: ImageBuffer<Rgb<u8>, Vec<u8>>, image_path: &Path) -> MeshResult<()> {
    buffer.save(image_path).map_err(|e| {
        MeshError::FileReadError(format!("Failed to save image {:?}: {}", image_path, e))
    })
}
