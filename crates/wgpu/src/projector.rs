//! WGPU-backed projection entry points.
//!
//! This module starts the GPU backend with a deterministic compute implementation for
//! point-cloud sampling and an offscreen render pipeline for the fast image projection
//! path.

use crate::error::{WgpuProjectorError, WgpuResult};
use crate::gpu::bindings::*;
use crate::gpu::buffers as gpu_buffers;
use crate::gpu::formats::*;
use crate::image::output::{
    PackedWgpuImageOutput, PlaneView, WgpuMappedImageReadback, WgpuOutputSelection,
    WgpuOwnedMappedImageReadback,
};
use crate::image::pipelines::{create_image_pipeline_resources, ImagePipelineResources};
use crate::image::prepared_scene::{prepare_image_scene_input, WgpuPreparedScene};
use crate::image::projection::prepare_prepared_view;
use crate::image::readback::{
    into_owned_mapped_readback, schedule_readback_maps, with_mapped_readback,
};
use crate::image::render_pass;
use crate::image::resources::{
    create_packed_output_buffers, create_render_textures, enqueue_packed_output_copies,
    WgpuReadbackLeaseGuard, WgpuReadbackPool, WgpuReadbackPoolHandle,
};
use crate::image::ssao;
use crate::image::timings::WgpuImageTimings;
use crate::point_cloud::pipeline as point_cloud_pipeline;
use mesh_projector_core::config::LightingConfig;
use mesh_projector_core::config::{MsaaSampleCount, PointCloudConfig, ProjectionConfig};
use mesh_projector_core::types::image_batch::ImageBatch;
use mesh_projector_core::types::mesh::{Scene, ShadingMode};
use ndarray::Array2;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use wgpu::util::DeviceExt;

static NEXT_PROJECTOR_ID: AtomicU64 = AtomicU64::new(1);

/// GPU projector facade.
pub struct WgpuProjector {
    id: u64,
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    image_pipeline: ImagePipelineResources,
    supported_msaa_sample_counts: Vec<u32>,
    readback_pool: WgpuReadbackPoolHandle,
}

impl WgpuProjector {
    /// Creates a native/headless WGPU projector.
    pub async fn new() -> WgpuResult<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let mut required_features = wgpu::Features::empty();
        if adapter
            .features()
            .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
        {
            required_features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        }

        let supported_msaa_sample_counts = supported_image_msaa_sample_counts(&adapter);
        let mut required_limits = adapter.limits();
        required_limits.max_storage_buffer_binding_size = required_limits
            .max_storage_buffer_binding_size
            .max(wgpu::Limits::default().max_storage_buffer_binding_size);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("mesh-projector-wgpu device"),
                required_features,
                required_limits,
                ..Default::default()
            })
            .await?;

        let image_pipeline =
            create_image_pipeline_resources(&device, &supported_msaa_sample_counts);

        Ok(Self {
            id: NEXT_PROJECTOR_ID.fetch_add(1, Ordering::Relaxed),
            instance,
            adapter,
            device,
            queue,
            image_pipeline,
            supported_msaa_sample_counts,
            readback_pool: Arc::new(Mutex::new(WgpuReadbackPool::default())),
        })
    }

    /// Synchronous constructor for non-async applications and examples.
    pub fn new_blocking() -> WgpuResult<Self> {
        pollster::block_on(Self::new())
    }

    /// Returns the selected adapter metadata for diagnostics and benchmarks.
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    /// Returns the MSAA sample counts supported by the current image pipeline.
    ///
    /// `X1` is the baseline path and `X4` is the portable WebGPU multisample target used by
    /// the custom G-buffer resolve. Adapter-specific `X2`/`X8` support will be exposed here
    /// once the render pipeline cache is extended to create those variants from format features.
    pub fn supported_msaa_sample_counts(&self) -> Vec<u32> {
        self.supported_msaa_sample_counts.clone()
    }

    fn msaa_sample_count(&self, sample_count: &MsaaSampleCount) -> WgpuResult<u32> {
        validate_msaa_sample_count(&self.supported_msaa_sample_counts, sample_count)
    }

    pub fn prepare_scene(&self, mesh: &Scene) -> WgpuResult<WgpuPreparedScene> {
        let input = prepare_image_scene_input(mesh)?;
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh-projector prepared image vertices"),
                contents: bytemuck::cast_slice(&input.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let texel_buffer = self.create_storage_buffer(
            "mesh-projector prepared image texels",
            bytemuck::cast_slice(&input.texels),
        );
        let material_buffer = self.create_storage_buffer(
            "mesh-projector prepared image materials",
            bytemuck::cast_slice(&input.materials),
        );
        Ok(WgpuPreparedScene {
            projector_id: self.id,
            vertex_buffer,
            material_buffer,
            texel_buffer,
            triangles: input.triangles,
            shading_mode: mesh.mode,
            corner_count: input.vertices.len() as u32,
        })
    }

    /// Image projection entry point.
    ///
    /// Rendering is GPU-only; unsupported settings return `WgpuProjectorError` instead of
    /// switching to the CPU reference renderer.
    pub fn project_to_image(
        &self,
        mesh: &Scene,
        settings: &ProjectionConfig,
        lighting_config: &LightingConfig,
    ) -> WgpuResult<ImageBatch> {
        if settings.filtering_window_radius > 0 {
            return Err(WgpuProjectorError::UnsupportedOutlierFiltering);
        }
        self.msaa_sample_count(&settings.msaa.sample_count)?;

        self.project_to_image_fast(mesh, settings, lighting_config)
    }

    /// Fast offscreen GPU image projection for the base rasterization path.
    pub fn project_to_image_fast(
        &self,
        mesh: &Scene,
        settings: &ProjectionConfig,
        lighting_config: &LightingConfig,
    ) -> WgpuResult<ImageBatch> {
        self.project_to_image_fast_with_timings(mesh, settings, lighting_config)
            .map(|(image_batch, _timings)| image_batch)
    }

    /// Fast offscreen GPU image projection with stage timings for benchmarking.
    pub fn project_to_image_fast_with_timings(
        &self,
        mesh: &Scene,
        settings: &ProjectionConfig,
        lighting_config: &LightingConfig,
    ) -> WgpuResult<(ImageBatch, WgpuImageTimings)> {
        let (packed, mut timings) = self.project_to_packed_image_with_timings(
            mesh,
            settings,
            lighting_config,
            WgpuOutputSelection::all(),
        )?;
        let convert_started = Instant::now();
        let image_batch = packed.into_image_batch();
        let convert_elapsed = convert_started.elapsed();
        timings.cpu_copy += convert_elapsed;
        timings.image_batch_conversion += convert_elapsed;
        timings.total += convert_elapsed;
        Ok((image_batch, timings))
    }

    pub fn project_to_packed_image(
        &self,
        mesh: &Scene,
        settings: &ProjectionConfig,
        lighting_config: &LightingConfig,
        selection: WgpuOutputSelection,
    ) -> WgpuResult<PackedWgpuImageOutput> {
        self.project_to_packed_image_with_timings(mesh, settings, lighting_config, selection)
            .map(|(output, _timings)| output)
    }

    pub fn project_to_packed_image_with_timings(
        &self,
        mesh: &Scene,
        settings: &ProjectionConfig,
        lighting_config: &LightingConfig,
        selection: WgpuOutputSelection,
    ) -> WgpuResult<(PackedWgpuImageOutput, WgpuImageTimings)> {
        let prepare_started = Instant::now();
        let prepared = self.prepare_scene(mesh)?;
        let prepare_elapsed = prepare_started.elapsed();
        let (packed, mut timings) = self.project_prepared_to_packed_image_with_timings(
            &prepared,
            settings,
            lighting_config,
            selection,
        )?;
        timings.prepare_scene += prepare_elapsed;
        timings.setup += prepare_elapsed;
        timings.total += prepare_elapsed;
        Ok((packed, timings))
    }

    pub fn project_prepared_to_packed_image_with_timings(
        &self,
        prepared: &WgpuPreparedScene,
        settings: &ProjectionConfig,
        lighting_config: &LightingConfig,
        selection: WgpuOutputSelection,
    ) -> WgpuResult<(PackedWgpuImageOutput, WgpuImageTimings)> {
        let mut copy_elapsed = std::time::Duration::ZERO;
        let (packed, mut timings) = self.project_prepared_to_image_readback_with(
            prepared,
            settings,
            lighting_config,
            selection,
            |mapped| {
                let copy_started = Instant::now();
                let packed = mapped.to_packed_output();
                copy_elapsed = copy_started.elapsed();
                packed
            },
        )?;
        timings.cpu_copy += copy_elapsed;
        Ok((packed, timings))
    }

    pub fn project_to_image_readback_with<R>(
        &self,
        mesh: &Scene,
        settings: &ProjectionConfig,
        lighting_config: &LightingConfig,
        selection: WgpuOutputSelection,
        callback: impl FnOnce(WgpuMappedImageReadback<'_>) -> R,
    ) -> WgpuResult<(R, WgpuImageTimings)> {
        let prepare_started = Instant::now();
        let prepared = self.prepare_scene(mesh)?;
        let prepare_elapsed = prepare_started.elapsed();
        let (result, mut timings) = self.project_prepared_to_image_readback_with(
            &prepared,
            settings,
            lighting_config,
            selection,
            callback,
        )?;
        timings.prepare_scene += prepare_elapsed;
        timings.setup += prepare_elapsed;
        timings.total += prepare_elapsed;
        Ok((result, timings))
    }

    pub fn project_to_owned_image_readback(
        &self,
        mesh: &Scene,
        settings: &ProjectionConfig,
        lighting_config: &LightingConfig,
        selection: WgpuOutputSelection,
    ) -> WgpuResult<(WgpuOwnedMappedImageReadback, WgpuImageTimings)> {
        let prepare_started = Instant::now();
        let prepared = self.prepare_scene(mesh)?;
        let prepare_elapsed = prepare_started.elapsed();
        let (result, mut timings) = self.project_prepared_to_owned_image_readback(
            &prepared,
            settings,
            lighting_config,
            selection,
        )?;
        timings.prepare_scene += prepare_elapsed;
        timings.setup += prepare_elapsed;
        timings.total += prepare_elapsed;
        Ok((result, timings))
    }

    pub fn project_prepared_to_owned_image_readback(
        &self,
        prepared: &WgpuPreparedScene,
        settings: &ProjectionConfig,
        lighting_config: &LightingConfig,
        selection: WgpuOutputSelection,
    ) -> WgpuResult<(WgpuOwnedMappedImageReadback, WgpuImageTimings)> {
        if prepared.projector_id != self.id {
            return Err(WgpuProjectorError::PreparedSceneProjectorMismatch);
        }
        if selection.is_empty() {
            return Err(WgpuProjectorError::NoOutputsSelected);
        }
        let sample_count = self.msaa_sample_count(&settings.msaa.sample_count)?;
        if settings.ssao.enabled && prepared.shading_mode != ShadingMode::BlinnPhong {
            log::warn!(
                "SSAO enabled but shading mode is {:?}. GPU SSAO is only applied with Blinn-Phong lighting.",
                prepared.shading_mode
            );
        }
        let apply_ssao = settings.ssao.enabled
            && prepared.shading_mode == ShadingMode::BlinnPhong
            && selection.color;

        let total_started = Instant::now();
        let setup_started = Instant::now();
        let resolution = settings.resolution as usize;
        let num_views = settings.view_angles.len();
        let pixels_per_view = resolution * resolution;

        if prepared.corner_count == 0 || settings.view_angles.is_empty() {
            let setup_elapsed = setup_started.elapsed();
            return Ok((
                WgpuOwnedMappedImageReadback::zeroed(selection, num_views, resolution),
                WgpuImageTimings {
                    total: total_started.elapsed(),
                    setup: setup_elapsed,
                    ..Default::default()
                },
            ));
        }

        let setup_elapsed = setup_started.elapsed();
        log::info!(
            "wgpu image setup: {:.3?} (prepared corners: {})",
            setup_elapsed,
            prepared.corner_count
        );

        let mut cull_elapsed = std::time::Duration::ZERO;
        let mut upload_indices_elapsed = std::time::Duration::ZERO;
        let mut render_elapsed = std::time::Duration::ZERO;
        let mut msaa_resolve_elapsed = std::time::Duration::ZERO;
        let mut pack_elapsed = std::time::Duration::ZERO;
        let mut readback_elapsed = std::time::Duration::ZERO;
        let mut copy_to_readback_elapsed = std::time::Duration::ZERO;
        let mut map_wait_elapsed = std::time::Duration::ZERO;
        let mut ssao_elapsed = std::time::Duration::ZERO;
        let packed_buffers =
            create_packed_output_buffers(&self.device, num_views, pixels_per_view, selection);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mesh-projector batched image encoder"),
            });
        let mut textures_to_keep_alive = Vec::new();

        for (view_index, orientation) in settings.view_angles.iter().enumerate() {
            let cull_started = Instant::now();
            let view_geometry =
                prepare_prepared_view(prepared, settings, lighting_config, orientation)?;
            cull_elapsed += cull_started.elapsed();
            if view_geometry.indices.is_empty() {
                continue;
            }

            let upload_started = Instant::now();
            let index_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mesh-projector image visible indices"),
                    contents: bytemuck::cast_slice(&view_geometry.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
            let params_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mesh-projector image params"),
                    contents: bytemuck::bytes_of(&view_geometry.params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mesh-projector image bind group"),
                layout: &self.image_pipeline.bind_group_layout,
                entries: &[
                    bind_entry(0, &prepared.texel_buffer),
                    bind_entry(1, &params_buffer),
                    bind_entry(2, &prepared.material_buffer),
                ],
            });
            let mut textures =
                create_render_textures(&self.device, settings.resolution, sample_count);
            upload_indices_elapsed += upload_started.elapsed();
            let render_selection = WgpuOutputSelection {
                color: selection.color || apply_ssao,
                depth: selection.depth,
                normal: selection.normal || apply_ssao,
                occupancy: selection.occupancy || apply_ssao,
            };

            let render_started = Instant::now();
            render_pass::encode_image_view(
                &mut encoder,
                self.image_pipeline
                    .pipeline_for(render_selection, sample_count),
                &bind_group,
                &prepared.vertex_buffer,
                &index_buffer,
                view_geometry.indices.len() as u32,
                settings.background.to_srgba8(),
                render_selection,
                &textures,
            );
            render_elapsed += render_started.elapsed();

            if sample_count > 1 {
                let resolve_started = Instant::now();
                render_pass::encode_msaa_resolve(
                    &self.device,
                    &self.image_pipeline,
                    &mut encoder,
                    &textures,
                    settings.resolution,
                    sample_count,
                );
                msaa_resolve_elapsed += resolve_started.elapsed();
            }

            if apply_ssao {
                let ssao_started = Instant::now();
                let color = ssao::encode_ssao(
                    &self.device,
                    &self.image_pipeline.ssao,
                    &mut encoder,
                    &textures,
                    settings.resolution,
                    &settings.ssao,
                    view_geometry.params.projection_kind,
                    view_geometry.params.projection[3],
                    view_geometry.params.projection[2],
                );
                ssao_elapsed += ssao_started.elapsed();
                textures.ssao_color = Some(color);
            }

            let pack_started = Instant::now();
            render_pass::encode_pack_outputs(
                &self.device,
                &self.image_pipeline,
                &mut encoder,
                &textures,
                settings.resolution,
                view_index,
                selection,
                apply_ssao,
                &packed_buffers,
            );
            pack_elapsed += pack_started.elapsed();

            textures_to_keep_alive.push(textures);
        }

        let readback_lease = self
            .readback_pool
            .lock()
            .expect("readback pool mutex poisoned")
            .acquire(&self.device, &packed_buffers, selection);

        let copy_to_readback_started = Instant::now();
        enqueue_packed_output_copies(&mut encoder, &packed_buffers, &readback_lease, selection);
        let scheduled = schedule_readback_maps(&encoder, &readback_lease, selection);
        copy_to_readback_elapsed += copy_to_readback_started.elapsed();

        let submit_started = Instant::now();
        self.queue.submit(Some(encoder.finish()));
        render_elapsed += submit_started.elapsed();

        let readback_guard =
            WgpuReadbackLeaseGuard::new(Arc::clone(&self.readback_pool), readback_lease);
        let (owned_readback, readback_wait_timings) = into_owned_mapped_readback(
            &self.instance,
            &packed_buffers,
            readback_guard,
            selection,
            num_views,
            resolution,
            scheduled,
        )?;
        readback_elapsed += readback_wait_timings.map_wait;
        map_wait_elapsed += readback_wait_timings.map_wait;

        drop(textures_to_keep_alive);

        let timings = WgpuImageTimings {
            total: total_started.elapsed(),
            setup: setup_elapsed,
            prepare_scene: std::time::Duration::ZERO,
            prepare_vertices: cull_elapsed,
            cull_indices: cull_elapsed,
            upload: upload_indices_elapsed,
            upload_indices: upload_indices_elapsed,
            render_submit: render_elapsed,
            msaa_resolve: msaa_resolve_elapsed,
            ssao: ssao_elapsed,
            copy_to_staging: pack_elapsed,
            pack_output: pack_elapsed,
            copy_to_readback: copy_to_readback_elapsed,
            map_wait: map_wait_elapsed,
            readback: readback_elapsed,
            cpu_copy: std::time::Duration::ZERO,
            image_batch_conversion: std::time::Duration::ZERO,
        };
        log::info!(
            "wgpu image timings: total={:.3?}, setup={:.3?}, prep={:.3?}, upload={:.3?}, render_submit={:.3?}, pack_output={:.3?}, copy_to_readback={:.3?}, map_wait={:.3?}, readback={:.3?}, cpu_copy={:.3?}",
            timings.total,
            timings.setup,
            timings.prepare_vertices,
            timings.upload,
            timings.render_submit,
            timings.pack_output,
            timings.copy_to_readback,
            timings.map_wait,
            timings.readback,
            timings.cpu_copy
        );

        Ok((owned_readback, timings))
    }

    pub fn project_prepared_to_image_readback_with<R>(
        &self,
        prepared: &WgpuPreparedScene,
        settings: &ProjectionConfig,
        lighting_config: &LightingConfig,
        selection: WgpuOutputSelection,
        callback: impl FnOnce(WgpuMappedImageReadback<'_>) -> R,
    ) -> WgpuResult<(R, WgpuImageTimings)> {
        if prepared.projector_id != self.id {
            return Err(WgpuProjectorError::PreparedSceneProjectorMismatch);
        }
        if selection.is_empty() {
            return Err(WgpuProjectorError::NoOutputsSelected);
        }
        let sample_count = self.msaa_sample_count(&settings.msaa.sample_count)?;
        if settings.ssao.enabled && prepared.shading_mode != ShadingMode::BlinnPhong {
            log::warn!(
                "SSAO enabled but shading mode is {:?}. GPU SSAO is only applied with Blinn-Phong lighting.",
                prepared.shading_mode
            );
        }
        let apply_ssao = settings.ssao.enabled
            && prepared.shading_mode == ShadingMode::BlinnPhong
            && selection.color;

        let total_started = Instant::now();
        let setup_started = Instant::now();
        let resolution = settings.resolution as usize;
        let num_views = settings.view_angles.len();
        let pixels_per_view = resolution * resolution;

        if prepared.corner_count == 0 || settings.view_angles.is_empty() {
            let setup_elapsed = setup_started.elapsed();
            let result = with_zeroed_readback(selection, num_views, resolution, callback);
            return Ok((
                result,
                WgpuImageTimings {
                    total: total_started.elapsed(),
                    setup: setup_elapsed,
                    ..Default::default()
                },
            ));
        }

        let setup_elapsed = setup_started.elapsed();
        log::info!(
            "wgpu image setup: {:.3?} (prepared corners: {})",
            setup_elapsed,
            prepared.corner_count
        );

        let mut cull_elapsed = std::time::Duration::ZERO;
        let mut upload_indices_elapsed = std::time::Duration::ZERO;
        let mut render_elapsed = std::time::Duration::ZERO;
        let mut msaa_resolve_elapsed = std::time::Duration::ZERO;
        let mut pack_elapsed = std::time::Duration::ZERO;
        let mut readback_elapsed = std::time::Duration::ZERO;
        let mut copy_to_readback_elapsed = std::time::Duration::ZERO;
        let mut map_wait_elapsed = std::time::Duration::ZERO;
        let mut ssao_elapsed = std::time::Duration::ZERO;
        let packed_buffers =
            create_packed_output_buffers(&self.device, num_views, pixels_per_view, selection);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mesh-projector batched image encoder"),
            });
        let mut textures_to_keep_alive = Vec::new();

        for (view_index, orientation) in settings.view_angles.iter().enumerate() {
            let cull_started = Instant::now();
            let view_geometry =
                prepare_prepared_view(prepared, settings, lighting_config, orientation)?;
            cull_elapsed += cull_started.elapsed();
            if view_geometry.indices.is_empty() {
                continue;
            }

            let upload_started = Instant::now();
            let index_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mesh-projector image visible indices"),
                    contents: bytemuck::cast_slice(&view_geometry.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
            let params_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mesh-projector image params"),
                    contents: bytemuck::bytes_of(&view_geometry.params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mesh-projector image bind group"),
                layout: &self.image_pipeline.bind_group_layout,
                entries: &[
                    bind_entry(0, &prepared.texel_buffer),
                    bind_entry(1, &params_buffer),
                    bind_entry(2, &prepared.material_buffer),
                ],
            });
            let mut textures =
                create_render_textures(&self.device, settings.resolution, sample_count);
            upload_indices_elapsed += upload_started.elapsed();
            let render_selection = WgpuOutputSelection {
                color: selection.color || apply_ssao,
                depth: selection.depth,
                normal: selection.normal || apply_ssao,
                occupancy: selection.occupancy || apply_ssao,
            };

            let render_started = Instant::now();
            render_pass::encode_image_view(
                &mut encoder,
                self.image_pipeline
                    .pipeline_for(render_selection, sample_count),
                &bind_group,
                &prepared.vertex_buffer,
                &index_buffer,
                view_geometry.indices.len() as u32,
                settings.background.to_srgba8(),
                render_selection,
                &textures,
            );
            render_elapsed += render_started.elapsed();

            if sample_count > 1 {
                let resolve_started = Instant::now();
                render_pass::encode_msaa_resolve(
                    &self.device,
                    &self.image_pipeline,
                    &mut encoder,
                    &textures,
                    settings.resolution,
                    sample_count,
                );
                msaa_resolve_elapsed += resolve_started.elapsed();
            }

            if apply_ssao {
                let ssao_started = Instant::now();
                let color = ssao::encode_ssao(
                    &self.device,
                    &self.image_pipeline.ssao,
                    &mut encoder,
                    &textures,
                    settings.resolution,
                    &settings.ssao,
                    view_geometry.params.projection_kind,
                    view_geometry.params.projection[3],
                    view_geometry.params.projection[2],
                );
                ssao_elapsed += ssao_started.elapsed();
                textures.ssao_color = Some(color);
            }

            let pack_started = Instant::now();
            render_pass::encode_pack_outputs(
                &self.device,
                &self.image_pipeline,
                &mut encoder,
                &textures,
                settings.resolution,
                view_index,
                selection,
                apply_ssao,
                &packed_buffers,
            );
            pack_elapsed += pack_started.elapsed();

            textures_to_keep_alive.push(textures);
        }

        let readback_lease = self
            .readback_pool
            .lock()
            .expect("readback pool mutex poisoned")
            .acquire(&self.device, &packed_buffers, selection);

        let copy_to_readback_started = Instant::now();
        enqueue_packed_output_copies(&mut encoder, &packed_buffers, &readback_lease, selection);
        let scheduled = schedule_readback_maps(&encoder, &readback_lease, selection);
        copy_to_readback_elapsed += copy_to_readback_started.elapsed();

        let submit_started = Instant::now();
        self.queue.submit(Some(encoder.finish()));
        render_elapsed += submit_started.elapsed();

        let mapped_result = with_mapped_readback(
            &self.instance,
            &packed_buffers,
            &readback_lease,
            selection,
            num_views,
            resolution,
            scheduled,
            callback,
        );

        self.readback_pool
            .lock()
            .expect("readback pool mutex poisoned")
            .release(readback_lease);
        let (result, readback_wait_timings) = mapped_result?;
        readback_elapsed += readback_wait_timings.map_wait;
        map_wait_elapsed += readback_wait_timings.map_wait;

        drop(textures_to_keep_alive);

        let timings = WgpuImageTimings {
            total: total_started.elapsed(),
            setup: setup_elapsed,
            prepare_scene: std::time::Duration::ZERO,
            prepare_vertices: cull_elapsed,
            cull_indices: cull_elapsed,
            upload: upload_indices_elapsed,
            upload_indices: upload_indices_elapsed,
            render_submit: render_elapsed,
            msaa_resolve: msaa_resolve_elapsed,
            ssao: ssao_elapsed,
            copy_to_staging: pack_elapsed,
            pack_output: pack_elapsed,
            copy_to_readback: copy_to_readback_elapsed,
            map_wait: map_wait_elapsed,
            readback: readback_elapsed,
            cpu_copy: std::time::Duration::ZERO,
            image_batch_conversion: std::time::Duration::ZERO,
        };
        log::info!(
            "wgpu image timings: total={:.3?}, setup={:.3?}, prep={:.3?}, upload={:.3?}, render_submit={:.3?}, pack_output={:.3?}, copy_to_readback={:.3?}, map_wait={:.3?}, readback={:.3?}, cpu_copy={:.3?}",
            timings.total,
            timings.setup,
            timings.prepare_vertices,
            timings.upload,
            timings.render_submit,
            timings.pack_output,
            timings.copy_to_readback,
            timings.map_wait,
            timings.readback,
            timings.cpu_copy
        );

        Ok((result, timings))
    }

    /// Generates a point cloud using a deterministic WGPU compute pipeline.
    pub fn project_to_point_cloud_by_sampling(
        &self,
        mesh: &Scene,
        settings: &PointCloudConfig,
        lighting_config: &LightingConfig,
    ) -> WgpuResult<(Array2<f32>, Array2<u8>)> {
        point_cloud_pipeline::project_to_point_cloud_by_sampling(
            &self.instance,
            &self.device,
            &self.queue,
            mesh,
            settings,
            lighting_config,
        )
    }

    fn create_storage_buffer(&self, label: &'static str, contents: &[u8]) -> wgpu::Buffer {
        gpu_buffers::create_storage_buffer(&self.device, label, contents)
    }
}

fn validate_msaa_sample_count(
    supported_msaa_sample_counts: &[u32],
    sample_count: &MsaaSampleCount,
) -> WgpuResult<u32> {
    let samples = sample_count.samples() as u32;
    if supported_msaa_sample_counts.contains(&samples) {
        Ok(samples)
    } else {
        Err(WgpuProjectorError::UnsupportedMsaaSampleCount {
            sample_count: samples,
        })
    }
}

fn with_zeroed_readback<R>(
    selection: WgpuOutputSelection,
    num_views: usize,
    resolution: usize,
    callback: impl FnOnce(WgpuMappedImageReadback<'_>) -> R,
) -> R {
    let pixels_per_view = resolution * resolution;
    let color_bytes_per_view = pixels_per_view * 4;
    let color = selection
        .color
        .then(|| vec![0; num_views * color_bytes_per_view]);
    let depth = selection
        .depth
        .then(|| vec![0; num_views * pixels_per_view]);
    let normal = selection.normal.then(|| {
        let mut bytes = vec![0; num_views * color_bytes_per_view];
        for pixel in bytes.chunks_exact_mut(4) {
            pixel[3] = 255;
        }
        bytes
    });
    let occupancy = selection
        .occupancy
        .then(|| vec![0; num_views * pixels_per_view]);

    callback(WgpuMappedImageReadback {
        selection,
        num_views,
        resolution,
        color: color.as_ref().map(|bytes| PlaneView {
            bytes,
            bytes_per_view: color_bytes_per_view,
            stride_bytes: color_bytes_per_view,
        }),
        depth: depth.as_ref().map(|bytes| PlaneView {
            bytes,
            bytes_per_view: pixels_per_view,
            stride_bytes: pixels_per_view,
        }),
        normal: normal.as_ref().map(|bytes| PlaneView {
            bytes,
            bytes_per_view: color_bytes_per_view,
            stride_bytes: color_bytes_per_view,
        }),
        occupancy: occupancy.as_ref().map(|bytes| PlaneView {
            bytes,
            bytes_per_view: pixels_per_view,
            stride_bytes: pixels_per_view,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_projector_core::config::{
        BackgroundConfig, CameraConfig, LightComponents, LightSource, MsaaConfig, SsaoConfig,
    };
    use mesh_projector_core::types::color::Color;

    fn empty_scene() -> Scene {
        Scene::new(Vec::new(), ShadingMode::VertexColor, None)
    }

    fn projection_settings() -> ProjectionConfig {
        ProjectionConfig {
            resolution: 8,
            filtering_window_radius: 0,
            view_angles: Vec::new(),
            camera: CameraConfig::default(),
            msaa: MsaaConfig {
                sample_count: MsaaSampleCount::X1,
            },
            ssao: SsaoConfig::disabled(),
            background: BackgroundConfig::default(),
        }
    }

    fn lighting_config() -> LightingConfig {
        LightingConfig {
            source: LightSource::Ambient,
            components: LightComponents {
                global_ambient: Color::white(),
                global_diffuse: Color::white(),
                global_specular: Color::black(),
            },
        }
    }

    #[test]
    fn msaa_validation_returns_direct_unsupported_error() {
        let err = validate_msaa_sample_count(&[1, 4], &MsaaSampleCount::X8)
            .expect_err("X8 should be rejected when unsupported");

        assert!(matches!(
            err,
            WgpuProjectorError::UnsupportedMsaaSampleCount { sample_count: 8 }
        ));
    }

    #[test]
    fn project_to_image_errors_for_outlier_filtering() {
        let projector = WgpuProjector::new_blocking().expect("wgpu projector should initialize");
        let mut settings = projection_settings();
        settings.filtering_window_radius = 1;

        match projector.project_to_image(&empty_scene(), &settings, &lighting_config()) {
            Err(WgpuProjectorError::UnsupportedOutlierFiltering) => {}
            Err(err) => panic!("expected outlier filtering error, got {err}"),
            Ok(_) => panic!("outlier filtering should not be supported by the GPU-only image path"),
        }
    }
}
