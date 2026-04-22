//! Image saving operations.

use crate::mesh_io::errors::{MeshError, MeshResult};
use image::{ImageBuffer, Luma, Rgb, Rgba};
use log::warn;
use ndarray::ArrayViewD;
use std::fs;
use std::path::Path;

/// Saves image data (H, W) or (H, W, C) to a specific output path (creates parent directories if needed)
pub fn save_image_data(image: ArrayViewD<u8>, output_path: &Path) -> MeshResult<()> {
    // Ensure parent directory exists
    if let Some(parent_dir) = output_path.parent() {
        if !parent_dir.as_os_str().is_empty() {
            create_output_directory(parent_dir)?;
        }
    }
    match image.ndim() {
        2 => {
            let (rows, cols) = (image.shape()[0], image.shape()[1]);
            let pixels: Vec<u8> = image.iter().cloned().collect();
            let buffer = ImageBuffer::<Luma<u8>, _>::from_raw(cols as u32, rows as u32, pixels)
                .ok_or_else(|| {
                    MeshError::ImageBufferError(
                        "Failed to create grayscale image buffer".to_string(),
                    )
                })?;
            buffer.save(output_path).map_err(|e| {
                MeshError::FileReadError(format!("Failed to save image {:?}: {}", output_path, e))
            })
        }
        3 => {
            let (rows, cols) = (image.shape()[0], image.shape()[1]);
            let channels = image.shape()[2];
            match channels {
                3 => {
                    let rgb_pixels: Vec<u8> = image.iter().cloned().collect();
                    let buffer =
                        ImageBuffer::<Rgb<u8>, _>::from_raw(cols as u32, rows as u32, rgb_pixels)
                            .ok_or_else(|| {
                            MeshError::ImageBufferError("Failed to create image buffer".to_string())
                        })?;
                    buffer.save(output_path).map_err(|e| {
                        MeshError::FileReadError(format!(
                            "Failed to save image {:?}: {}",
                            output_path, e
                        ))
                    })
                }
                4 => {
                    let rgba_pixels: Vec<u8> = image.iter().cloned().collect();
                    let buffer =
                        ImageBuffer::<Rgba<u8>, _>::from_raw(cols as u32, rows as u32, rgba_pixels)
                            .ok_or_else(|| {
                                MeshError::ImageBufferError(
                                    "Failed to create RGBA image buffer".to_string(),
                                )
                            })?;
                    buffer.save(output_path).map_err(|e| {
                        MeshError::FileReadError(format!(
                            "Failed to save image {:?}: {}",
                            output_path, e
                        ))
                    })
                }
                other => {
                    warn!(
                        "Image array has unsupported number of channels: {}. Skipping save.",
                        other
                    );
                    return Ok(());
                }
            }
        }
        other => {
            warn!(
                "Image array has unsupported number of dimensions: {}. Skipping save.",
                other
            );
            Ok(())
        }
    }
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
