use crate::error::{WgpuProjectorError, WgpuResult};
use crate::image::output::OwnedPlaneView;
use crate::point_cloud::output::WgpuOwnedMappedPointCloudReadback;
use crate::point_cloud::resources::{
    PointCloudOutputBuffers, PointCloudReadbackLease, PointCloudReadbackLeaseGuard,
};
use std::sync::mpsc;
use std::time::{Duration, Instant};

pub(crate) struct ScheduledPointCloudReadbacks {
    receiver: mpsc::Receiver<(usize, Result<(), wgpu::BufferAsyncError>)>,
    count: usize,
}

#[derive(Default)]
pub(crate) struct PointCloudReadbackWaitTimings {
    pub(crate) map_wait: Duration,
}

pub(crate) fn schedule_point_cloud_readback_maps(
    encoder: &wgpu::CommandEncoder,
    readback: &PointCloudReadbackLease,
) -> ScheduledPointCloudReadbacks {
    let (sender, receiver) = mpsc::channel();
    for (index, buffer) in [&readback.positions, &readback.colors]
        .into_iter()
        .enumerate()
    {
        let sender = sender.clone();
        encoder.map_buffer_on_submit(buffer, wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send((index, result));
        });
    }
    drop(sender);

    ScheduledPointCloudReadbacks { receiver, count: 2 }
}

pub(crate) fn into_owned_mapped_point_cloud_readback(
    instance: &wgpu::Instance,
    buffers: &PointCloudOutputBuffers,
    mut readback: PointCloudReadbackLeaseGuard,
    num_points: usize,
    scheduled: ScheduledPointCloudReadbacks,
) -> WgpuResult<(
    WgpuOwnedMappedPointCloudReadback,
    PointCloudReadbackWaitTimings,
)> {
    let map_started = Instant::now();
    instance.poll_all(true);
    wait_for_maps(scheduled)?;
    let map_wait = map_started.elapsed();
    readback.mark_mapped();

    let positions = OwnedPlaneView::mapped(
        readback.lease().positions.get_mapped_range(..),
        buffers.positions_size as usize,
        std::mem::size_of::<[f32; 4]>(),
    );
    let colors = OwnedPlaneView::mapped(
        readback.lease().colors.get_mapped_range(..),
        buffers.colors_size as usize,
        std::mem::size_of::<u32>(),
    );

    Ok((
        WgpuOwnedMappedPointCloudReadback::new(num_points, positions, colors, readback),
        PointCloudReadbackWaitTimings { map_wait },
    ))
}

fn wait_for_maps(scheduled: ScheduledPointCloudReadbacks) -> WgpuResult<()> {
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
            .expect("point cloud readback map result was not delivered")
            .map_err(WgpuProjectorError::BufferMap)?;
    }

    Ok(())
}
