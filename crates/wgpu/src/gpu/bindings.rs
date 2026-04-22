pub(crate) fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    storage_entry_for(wgpu::ShaderStages::COMPUTE, binding, read_only)
}

pub(crate) fn storage_entry_for(
    visibility: wgpu::ShaderStages,
    binding: u32,
    read_only: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(crate) fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    uniform_entry_for(wgpu::ShaderStages::COMPUTE, binding)
}

pub(crate) fn uniform_entry_for(
    visibility: wgpu::ShaderStages,
    binding: u32,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(crate) fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

pub(crate) fn texture_bind_entry(
    binding: u32,
    view: &wgpu::TextureView,
) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

pub(crate) fn storage_texture_bind_entry(
    binding: u32,
    view: &wgpu::TextureView,
) -> wgpu::BindGroupEntry<'_> {
    texture_bind_entry(binding, view)
}

pub(crate) fn texture_entry(
    binding: u32,
    format: wgpu::TextureFormat,
) -> wgpu::BindGroupLayoutEntry {
    let sample_type = if format == crate::gpu::formats::RAW_DEPTH_FORMAT {
        wgpu::TextureSampleType::Float { filterable: false }
    } else {
        wgpu::TextureSampleType::Float { filterable: true }
    };
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

pub(crate) fn multisampled_texture_entry(
    binding: u32,
    _format: wgpu::TextureFormat,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: true,
        },
        count: None,
    }
}

pub(crate) fn storage_texture_entry(
    binding: u32,
    format: wgpu::TextureFormat,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

pub(crate) fn color_target(format: wgpu::TextureFormat, selected: bool) -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format,
        blend: None,
        write_mask: if selected {
            wgpu::ColorWrites::ALL
        } else {
            wgpu::ColorWrites::empty()
        },
    }
}

pub(crate) fn render_attachment<'a>(
    view: &'a wgpu::TextureView,
    resolve_target: Option<&'a wgpu::TextureView>,
    clear: wgpu::Color,
    selected: bool,
) -> wgpu::RenderPassColorAttachment<'a> {
    wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target,
        ops: wgpu::Operations {
            load: if selected {
                wgpu::LoadOp::Clear(clear)
            } else {
                wgpu::LoadOp::Load
            },
            store: if selected {
                wgpu::StoreOp::Store
            } else {
                wgpu::StoreOp::Discard
            },
        },
    }
}
