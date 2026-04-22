use std::fmt;

#[derive(Debug)]
pub enum WgpuProjectorError {
    AdapterUnavailable,
    RequestAdapter(wgpu::RequestAdapterError),
    RequestDevice(wgpu::RequestDeviceError),
    NoOutputsSelected,
    UnsupportedOutlierFiltering,
    PreparedSceneProjectorMismatch,
    UnsupportedMsaaSampleCount { sample_count: u32 },
    TooManyPointLights { count: usize, max: usize },
    OutputTooLarge { points: usize },
    BufferMap(wgpu::BufferAsyncError),
    BufferMapChannelClosed,
    Shape(ndarray::ShapeError),
}

impl fmt::Display for WgpuProjectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AdapterUnavailable => write!(f, "no compatible wgpu adapter was found"),
            Self::RequestAdapter(err) => write!(f, "failed to request wgpu adapter: {err}"),
            Self::RequestDevice(err) => write!(f, "failed to request wgpu device: {err}"),
            Self::NoOutputsSelected => write!(f, "at least one wgpu image output must be selected"),
            Self::UnsupportedOutlierFiltering => write!(
                f,
                "wgpu image projection does not support outlier filtering; use mesh-projector-cpu explicitly for filtering"
            ),
            Self::PreparedSceneProjectorMismatch => {
                write!(
                    f,
                    "prepared scene was created by a different wgpu projector"
                )
            }
            Self::UnsupportedMsaaSampleCount { sample_count } => {
                write!(f, "unsupported wgpu MSAA sample count: {sample_count}")
            }
            Self::TooManyPointLights { count, max } => write!(
                f,
                "point light count {count} exceeds the wgpu backend limit of {max}"
            ),
            Self::OutputTooLarge { points } => write!(
                f,
                "point cloud has {points} points, exceeding the current u32-indexed GPU path"
            ),
            Self::BufferMap(err) => write!(f, "failed to map GPU readback buffer: {err}"),
            Self::BufferMapChannelClosed => write!(f, "GPU readback callback channel closed"),
            Self::Shape(err) => write!(f, "failed to build ndarray from GPU output: {err}"),
        }
    }
}

impl std::error::Error for WgpuProjectorError {}

impl From<wgpu::RequestAdapterError> for WgpuProjectorError {
    fn from(value: wgpu::RequestAdapterError) -> Self {
        Self::RequestAdapter(value)
    }
}

impl From<wgpu::RequestDeviceError> for WgpuProjectorError {
    fn from(value: wgpu::RequestDeviceError) -> Self {
        Self::RequestDevice(value)
    }
}

impl From<ndarray::ShapeError> for WgpuProjectorError {
    fn from(value: ndarray::ShapeError) -> Self {
        Self::Shape(value)
    }
}

pub type WgpuResult<T> = Result<T, WgpuProjectorError>;
