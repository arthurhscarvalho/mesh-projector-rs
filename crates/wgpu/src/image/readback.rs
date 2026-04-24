//! Packed image output readback.

use crate::error::{WgpuProjectorError, WgpuResult};
use crate::image::output::{
    OwnedPlaneView, PlaneView, WgpuMappedImageReadback, WgpuOutputSelection,
    WgpuOwnedMappedImageReadback,
};
use crate::image::resources::{PackedOutputGpuBuffers, WgpuReadbackLease, WgpuReadbackLeaseGuard};
use std::sync::mpsc;
use std::time::{Duration, Instant};

pub(crate) struct ScheduledReadbacks {
    receiver: mpsc::Receiver<(usize, Result<(), wgpu::BufferAsyncError>)>,
    count: usize,
}

#[derive(Clone, Copy)]
struct PendingPlane<'a> {
    buffer: &'a wgpu::Buffer,
}

#[derive(Default)]
pub(crate) struct ReadbackWaitTimings {
    pub(crate) map_wait: Duration,
}

pub(crate) fn schedule_readback_maps(
    encoder: &wgpu::CommandEncoder,
    readback: &WgpuReadbackLease,
    selection: WgpuOutputSelection,
) -> ScheduledReadbacks {
    let pending = collect_pending_planes(readback, selection);
    let (sender, receiver) = mpsc::channel();
    for (index, plane) in pending.iter().enumerate() {
        let sender = sender.clone();
        encoder.map_buffer_on_submit(plane.buffer, wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send((index, result));
        });
    }
    drop(sender);

    ScheduledReadbacks {
        receiver,
        count: pending.len(),
    }
}

pub(crate) fn with_mapped_readback<R>(
    instance: &wgpu::Instance,
    buffers: &PackedOutputGpuBuffers,
    readback: &WgpuReadbackLease,
    selection: WgpuOutputSelection,
    num_views: usize,
    resolution: usize,
    scheduled: ScheduledReadbacks,
    callback: impl FnOnce(WgpuMappedImageReadback<'_>) -> R,
) -> WgpuResult<(R, ReadbackWaitTimings)> {
    let map_started = Instant::now();
    instance.poll_all(true);
    wait_for_maps(scheduled)?;
    let map_wait = map_started.elapsed();

    let color = readback
        .color
        .as_ref()
        .map(|buffer| buffer.get_mapped_range(..));
    let depth = readback
        .depth
        .as_ref()
        .map(|buffer| buffer.get_mapped_range(..));
    let normal = readback
        .normal
        .as_ref()
        .map(|buffer| buffer.get_mapped_range(..));
    let occupancy = readback
        .occupancy
        .as_ref()
        .map(|buffer| buffer.get_mapped_range(..));

    let mapped = WgpuMappedImageReadback {
        selection,
        num_views,
        resolution,
        color: color.as_ref().map(|view| PlaneView {
            bytes: &view[..],
            bytes_per_view: buffers.color.bytes_per_view as usize,
            stride_bytes: buffers.color.stride_bytes as usize,
        }),
        depth: depth.as_ref().map(|view| PlaneView {
            bytes: &view[..],
            bytes_per_view: buffers.depth.bytes_per_view as usize,
            stride_bytes: buffers.depth.stride_bytes as usize,
        }),
        normal: normal.as_ref().map(|view| PlaneView {
            bytes: &view[..],
            bytes_per_view: buffers.normal.bytes_per_view as usize,
            stride_bytes: buffers.normal.stride_bytes as usize,
        }),
        occupancy: occupancy.as_ref().map(|view| PlaneView {
            bytes: &view[..],
            bytes_per_view: buffers.occupancy.bytes_per_view as usize,
            stride_bytes: buffers.occupancy.stride_bytes as usize,
        }),
    };

    let result = callback(mapped);

    drop(color);
    drop(depth);
    drop(normal);
    drop(occupancy);
    unmap_readbacks(readback);

    Ok((result, ReadbackWaitTimings { map_wait }))
}

pub(crate) fn into_owned_mapped_readback(
    instance: &wgpu::Instance,
    buffers: &PackedOutputGpuBuffers,
    mut readback: WgpuReadbackLeaseGuard,
    selection: WgpuOutputSelection,
    num_views: usize,
    resolution: usize,
    scheduled: ScheduledReadbacks,
) -> WgpuResult<(WgpuOwnedMappedImageReadback, ReadbackWaitTimings)> {
    let map_started = Instant::now();
    instance.poll_all(true);
    wait_for_maps(scheduled)?;
    let map_wait = map_started.elapsed();
    readback.mark_mapped();

    let color = readback.lease().color.as_ref().map(|buffer| {
        OwnedPlaneView::mapped(
            buffer.get_mapped_range(..),
            buffers.color.bytes_per_view as usize,
            buffers.color.stride_bytes as usize,
        )
    });
    let depth = readback.lease().depth.as_ref().map(|buffer| {
        OwnedPlaneView::mapped(
            buffer.get_mapped_range(..),
            buffers.depth.bytes_per_view as usize,
            buffers.depth.stride_bytes as usize,
        )
    });
    let normal = readback.lease().normal.as_ref().map(|buffer| {
        OwnedPlaneView::mapped(
            buffer.get_mapped_range(..),
            buffers.normal.bytes_per_view as usize,
            buffers.normal.stride_bytes as usize,
        )
    });
    let occupancy = readback.lease().occupancy.as_ref().map(|buffer| {
        OwnedPlaneView::mapped(
            buffer.get_mapped_range(..),
            buffers.occupancy.bytes_per_view as usize,
            buffers.occupancy.stride_bytes as usize,
        )
    });

    Ok((
        WgpuOwnedMappedImageReadback::new(
            selection, num_views, resolution, color, depth, normal, occupancy, readback,
        ),
        ReadbackWaitTimings { map_wait },
    ))
}

fn collect_pending_planes(
    readback: &WgpuReadbackLease,
    selection: WgpuOutputSelection,
) -> Vec<PendingPlane<'_>> {
    let mut pending = Vec::with_capacity(4);
    if selection.color {
        if let Some(buffer) = &readback.color {
            pending.push(PendingPlane { buffer });
        }
    }
    if selection.depth {
        if let Some(buffer) = &readback.depth {
            pending.push(PendingPlane { buffer });
        }
    }
    if selection.normal {
        if let Some(buffer) = &readback.normal {
            pending.push(PendingPlane { buffer });
        }
    }
    if selection.occupancy {
        if let Some(buffer) = &readback.occupancy {
            pending.push(PendingPlane { buffer });
        }
    }
    pending
}

fn wait_for_maps(scheduled: ScheduledReadbacks) -> WgpuResult<()> {
    let mut results = vec![None; scheduled.count];
    for _ in 0..scheduled.count {
        let (index, result) = scheduled
            .receiver
            .recv()
            .map_err(|_| WgpuProjectorError::BufferMapChannelClosed)?;
        results[index] = Some(result);
    }

    for result in results {
        result
            .expect("readback map result was not delivered")
            .map_err(WgpuProjectorError::BufferMap)?;
    }

    Ok(())
}

fn unmap_readbacks(readback: &WgpuReadbackLease) {
    if let Some(buffer) = &readback.color {
        buffer.unmap();
    }
    if let Some(buffer) = &readback.depth {
        buffer.unmap();
    }
    if let Some(buffer) = &readback.normal {
        buffer.unmap();
    }
    if let Some(buffer) = &readback.occupancy {
        buffer.unmap();
    }
}

#[cfg(test)]
fn copy_packed_scalar_bytes(
    packed_words: &[u8],
    dst: &mut [u8],
    num_views: usize,
    resolution: usize,
    stride_bytes: usize,
) {
    let pixels_per_view = resolution * resolution;

    for view_index in 0..num_views {
        let src_start = view_index * stride_bytes;
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

        copy_packed_scalar_bytes(&packed_words, &mut output, 2, 3, 12);

        assert_eq!(
            output,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        );
    }
}
