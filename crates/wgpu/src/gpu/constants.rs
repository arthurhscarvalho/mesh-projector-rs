use mesh_projector_core::types::mesh::ShadingMode;

pub(crate) const MAX_POINT_LIGHTS: usize = 16;
pub(crate) const SHADING_VERTEX_COLOR: u32 = 0;
pub(crate) const SHADING_TEXTURE_MAPPED: u32 = 1;
pub(crate) const SHADING_BLINN_PHONG: u32 = 2;
pub(crate) const LIGHT_AMBIENT: u32 = 0;
pub(crate) const LIGHT_DIRECTIONAL: u32 = 1;
pub(crate) const LIGHT_HEADLIGHT: u32 = 2;
pub(crate) const LIGHT_POINT: u32 = 3;

pub(crate) fn shading_mode_id(mode: ShadingMode) -> u32 {
    match mode {
        ShadingMode::VertexColor => SHADING_VERTEX_COLOR,
        ShadingMode::TextureMapped => SHADING_TEXTURE_MAPPED,
        ShadingMode::BlinnPhong => SHADING_BLINN_PHONG,
    }
}

pub(crate) fn vec4(v: [f32; 3]) -> [f32; 4] {
    [v[0], v[1], v[2], 0.0]
}
