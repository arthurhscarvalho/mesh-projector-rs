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
    // SAFETY: ndarray stores values in a contiguous Vec<u8>, which we can reuse directly.
    // This avoids an intermediate deep copy of the entire image.
    let slice = image.as_slice().ok_or_else(|| {
        MeshError::ImageBufferError("Image array is not contiguous in memory".to_string())
    })?;

    // We still need to create an owned Vec for ImageBuffer, but we avoid an *additional* clone
    // that was previously performed on the Array3 itself.
    let buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(cols as u32, rows as u32, slice.to_vec())
        .ok_or_else(|| MeshError::ImageBufferError("Failed to create image buffer".to_string()))?;
    let image_path = output_dir.join(format!("image_{:04}.png", view_index));
    save_image_buffer(buffer, &image_path)
}

/// Save image buffer to file
fn save_image_buffer(buffer: ImageBuffer<Rgb<u8>, Vec<u8>>, image_path: &Path) -> MeshResult<()> {
    buffer.save(image_path).map_err(|e| {
        MeshError::FileReadError(format!("Failed to save image {:?}: {}", image_path, e))
    })
}
