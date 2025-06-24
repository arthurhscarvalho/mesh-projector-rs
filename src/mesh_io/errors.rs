//! Error types for mesh I/O operations.

use std::fmt;

/// Error type for mesh loading and processing operations
#[derive(Debug, Clone)]
pub enum MeshError {
    /// File I/O related errors
    FileNotFound(String),
    FileReadError(String),
    DirectoryCreateError(String),

    /// Parsing and format errors
    ObjParseError(String),
    MtlParseError(String),
    TextureLoadError(String),
    ImageBufferError(String),

    /// Data validation errors
    NoModelsFound,
    NoProcessableSubmeshes,
    InvalidVertexData(String),
    InvalidTriangleData(String),
    InvalidUvData(String),
    MissingRequiredData(String),

    /// Configuration errors
    UnsupportedMode(String),
    PrecisionError(String),

    /// Internal processing errors
    GeometryProcessingError(String),
    MaterialProcessingError(String),
    TextureProcessingError(String),
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeshError::FileNotFound(path) => write!(f, "File not found: {}", path),
            MeshError::FileReadError(details) => write!(f, "Failed to read file: {}", details),
            MeshError::DirectoryCreateError(details) => {
                write!(f, "Failed to create directory: {}", details)
            }

            MeshError::ObjParseError(details) => write!(f, "Failed to parse OBJ file: {}", details),
            MeshError::MtlParseError(details) => write!(f, "Failed to parse MTL file: {}", details),
            MeshError::TextureLoadError(details) => {
                write!(f, "Failed to load texture: {}", details)
            }
            MeshError::ImageBufferError(details) => write!(f, "Image buffer error: {}", details),

            MeshError::NoModelsFound => write!(f, "No models found in the OBJ file"),
            MeshError::NoProcessableSubmeshes => write!(f, "No processable submeshes found"),
            MeshError::InvalidVertexData(details) => write!(f, "Invalid vertex data: {}", details),
            MeshError::InvalidTriangleData(details) => {
                write!(f, "Invalid triangle data: {}", details)
            }
            MeshError::InvalidUvData(details) => write!(f, "Invalid UV data: {}", details),
            MeshError::MissingRequiredData(details) => {
                write!(f, "Missing required data: {}", details)
            }

            MeshError::UnsupportedMode(details) => write!(f, "Unsupported mode: {}", details),
            MeshError::PrecisionError(details) => write!(f, "Precision error: {}", details),

            MeshError::GeometryProcessingError(details) => {
                write!(f, "Geometry processing error: {}", details)
            }
            MeshError::MaterialProcessingError(details) => {
                write!(f, "Material processing error: {}", details)
            }
            MeshError::TextureProcessingError(details) => {
                write!(f, "Texture processing error: {}", details)
            }
        }
    }
}

impl std::error::Error for MeshError {}

/// Convenience result type for mesh operations
pub type MeshResult<T> = Result<T, MeshError>;

/// Convert from tobj loading errors
impl From<tobj::LoadError> for MeshError {
    fn from(error: tobj::LoadError) -> Self {
        MeshError::ObjParseError(error.to_string())
    }
}

/// Convert from image loading errors
impl From<image::ImageError> for MeshError {
    fn from(error: image::ImageError) -> Self {
        MeshError::TextureLoadError(error.to_string())
    }
}

/// Convert from std::io errors
impl From<std::io::Error> for MeshError {
    fn from(error: std::io::Error) -> Self {
        MeshError::FileReadError(error.to_string())
    }
}
