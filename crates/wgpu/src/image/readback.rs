//! Packed image output readback.

use crate::error::WgpuResult;
use crate::gpu::buffers;
use crate::image::output::PackedWgpuImageOutput;
use crate::image::resources::PackedOutputGpuBuffers;

pub(crate) fn read_packed_output_buffers(
    instance: &wgpu::Instance,
    gpu_buffers: &PackedOutputGpuBuffers,
    output: &mut PackedWgpuImageOutput,
) -> WgpuResult<()> {
    let pixels_per_view = output.resolution * output.resolution;
    let color_bytes_per_view = pixels_per_view * 4;

    for (view_index, view_buffers) in gpu_buffers.views.iter().enumerate() {
        if let (Some(readback), Some(color)) = (&view_buffers.color_readback, output.color.as_mut())
        {
            let bytes = buffers::read_buffer(instance, readback)?;
            let dst_start = view_index * color_bytes_per_view;
            let dst_end = dst_start + color_bytes_per_view;
            color[dst_start..dst_end].copy_from_slice(&bytes[..color_bytes_per_view]);
        }

        if let (Some(readback), Some(normal)) =
            (&view_buffers.normal_readback, output.normal.as_mut())
        {
            let bytes = buffers::read_buffer(instance, readback)?;
            let dst_start = view_index * color_bytes_per_view;
            let dst_end = dst_start + color_bytes_per_view;
            normal[dst_start..dst_end].copy_from_slice(&bytes[..color_bytes_per_view]);
        }

        if let (Some(readback), Some(depth)) = (&view_buffers.depth_readback, output.depth.as_mut())
        {
            let bytes = buffers::read_buffer(instance, readback)?;
            let dst_start = view_index * pixels_per_view;
            let dst_end = dst_start + pixels_per_view;
            depth[dst_start..dst_end].copy_from_slice(&bytes[..pixels_per_view]);
        }

        if let (Some(readback), Some(occupancy)) =
            (&view_buffers.occupancy_readback, output.occupancy.as_mut())
        {
            let bytes = buffers::read_buffer(instance, readback)?;
            let dst_start = view_index * pixels_per_view;
            let dst_end = dst_start + pixels_per_view;
            occupancy[dst_start..dst_end].copy_from_slice(&bytes[..pixels_per_view]);
        }
    }

    Ok(())
}

#[cfg(test)]
fn copy_packed_scalar_bytes(
    packed_words: &[u8],
    dst: &mut [u8],
    num_views: usize,
    resolution: usize,
) {
    let pixels_per_view = resolution * resolution;
    let scalar_words_per_view = pixels_per_view.div_ceil(4);
    let src_stride = scalar_words_per_view * 4;

    for view_index in 0..num_views {
        let src_start = view_index * src_stride;
        let dst_start = view_index * pixels_per_view;
        let src_end = src_start + pixels_per_view;
        let dst_end = dst_start + pixels_per_view;
        dst[dst_start..dst_end].copy_from_slice(&packed_words[src_start..src_end]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_scalar_bytes_skip_per_view_padding() {
        let mut output = vec![0; 18];
        let packed_words = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 0, 0,
        ];

        copy_packed_scalar_bytes(&packed_words, &mut output, 2, 3);

        assert_eq!(
            output,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        );
    }
}
