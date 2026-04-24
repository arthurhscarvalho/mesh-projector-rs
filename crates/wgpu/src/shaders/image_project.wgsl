const MAX_POINT_LIGHTS: u32 = 16u;
const SHADING_VERTEX_COLOR: u32 = 0u;
const SHADING_TEXTURE_MAPPED: u32 = 1u;
const SHADING_BLINN_PHONG: u32 = 2u;
const LIGHT_AMBIENT: u32 = 0u;
const LIGHT_DIRECTIONAL: u32 = 1u;
const LIGHT_HEADLIGHT: u32 = 2u;
const LIGHT_POINT: u32 = 3u;
const PROJECTION_ORTHOGRAPHIC: u32 = 0u;
const PROJECTION_PERSPECTIVE: u32 = 1u;

struct Params {
    global_ambient: vec4<f32>,
    global_diffuse: vec4<f32>,
    global_specular: vec4<f32>,
    directional_light: vec4<f32>,
    view_direction: vec4<f32>,
    u_axis: vec4<f32>,
    v_axis: vec4<f32>,
    camera_position: vec4<f32>,
    point_lights: array<vec4<f32>, 16>,
    background: vec4<f32>,
    projection: vec4<f32>,
    depth: vec4<f32>,
    light_kind: u32,
    light_count: u32,
    pad0: u32,
    projection_kind: u32,
};

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) uv_color_rg: vec4<f32>,
    @location(3) color_b_material: vec4<f32>,
};

struct Material {
    ka: vec4<f32>,
    kd: vec4<f32>,
    ks_ns: vec4<f32>,
    tex: vec4<f32>,
    mode_illum_opacity: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) position_ws: vec3<f32>,
    @location(1) normal_ws: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
    @location(4) @interpolate(flat) material_id: u32,
    @location(5) depth: f32,
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) depth_byte: vec4<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) occupancy: vec4<f32>,
    @location(4) raw_depth: f32,
};

@group(0) @binding(0) var<storage, read> texels: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> materials: array<Material>;

fn srgb_byte_to_linear(b: u32) -> f32 {
    let x = f32(b) / 255.0;
    return pow(x, 2.2);
}

fn linear_to_srgb(x: f32) -> f32 {
    return pow(clamp(x, 0.0, 1.0), 1.0 / 2.2);
}

fn unpack_texel_linear(packed: u32) -> vec3<f32> {
    let r = packed & 255u;
    let g = (packed >> 8u) & 255u;
    let b = (packed >> 16u) & 255u;
    return vec3<f32>(
        srgb_byte_to_linear(r),
        srgb_byte_to_linear(g),
        srgb_byte_to_linear(b),
    );
}

fn normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len2 = dot(v, v);
    if len2 > 0.0 {
        return v * inverseSqrt(len2);
    }
    return fallback;
}

fn sample_texture_bilinear(tex: vec4<f32>, uv: vec2<f32>) -> vec3<f32> {
    let offset = u32(tex.x);
    let width = u32(tex.y);
    let height = u32(tex.z);
    let channels = u32(tex.w);
    if width == 0u || height == 0u || channels < 3u {
        return vec3<f32>(1.0, 1.0, 1.0);
    }

    let width_f = f32(width);
    let height_f = f32(height);
    let tex_xf = uv.x * width_f;
    let tex_yf = uv.y * height_f;
    let x0f = clamp(tex_xf, 0.0, width_f - 1.0001);
    let y0f = clamp(tex_yf, 0.0, height_f - 1.0001);
    let x0 = u32(x0f);
    let y0 = u32(y0f);
    let x1 = min(x0 + 1u, width - 1u);
    let y1 = min(y0 + 1u, height - 1u);
    let wx = x0f - f32(x0);
    let wy = y0f - f32(y0);
    let row0 = offset + y0 * width;
    let row1 = offset + y1 * width;
    let c00 = unpack_texel_linear(texels[row0 + x0]);
    let c10 = unpack_texel_linear(texels[row0 + x1]);
    let c01 = unpack_texel_linear(texels[row1 + x0]);
    let c11 = unpack_texel_linear(texels[row1 + x1]);
    return c00 * (1.0 - wx) * (1.0 - wy)
        + c10 * wx * (1.0 - wy)
        + c01 * (1.0 - wx) * wy
        + c11 * wx * wy;
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let pos = input.position.xyz;
    let u_axis = params.u_axis.xyz;
    let v_axis = params.v_axis.xyz;
    let view_dir = params.view_direction.xyz;
    let resolution = params.depth.z;
    let cube_center = params.projection.z;
    var col = 0.0;
    var row = 0.0;
    var depth_value = 0.0;

    if params.projection_kind == PROJECTION_PERSPECTIVE {
        let to_vertex = pos - params.camera_position.xyz;
        let x_cam = dot(to_vertex, u_axis);
        let y_cam = dot(to_vertex, v_axis);
        let z_cam = dot(to_vertex, view_dir);
        let scale = params.projection.w / max(z_cam, 0.000001);
        col = scale * x_cam + cube_center;
        row = scale * y_cam + cube_center;
        depth_value = z_cam;
    } else {
        col = dot(pos, u_axis) - params.projection.x;
        row = dot(pos, v_axis) - params.projection.y;
        depth_value = dot(pos, view_dir);
    }

    let flipped_col = resolution - col;
    let flipped_row = resolution - row;
    let clip_x = flipped_col / resolution * 2.0 - 1.0;
    let clip_y = 1.0 - flipped_row / resolution * 2.0;
    let near = min(params.depth.x, params.depth.y);
    let far = max(params.depth.x, params.depth.y);
    let depth_norm = clamp((depth_value - near) / max(far - near, 0.000001), 0.0, 1.0);

    out.clip_position = vec4<f32>(clip_x, clip_y, depth_norm, 1.0);
    out.position_ws = pos;
    out.normal_ws = normalize_or(input.normal.xyz, vec3<f32>(0.0, 0.0, 1.0));
    out.uv = input.uv_color_rg.xy;
    out.color = vec3<f32>(
        input.uv_color_rg.z,
        input.uv_color_rg.w,
        input.color_b_material.x,
    );
    out.material_id = u32(input.color_b_material.y);
    out.depth = depth_value;
    return out;
}

fn light_direction(normal_vs: vec3<f32>) -> vec3<f32> {
    if params.light_kind == LIGHT_AMBIENT {
        return normal_vs;
    }
    if params.light_kind == LIGHT_DIRECTIONAL {
        return normalize_or(params.directional_light.xyz, vec3<f32>(0.0, 0.0, 1.0));
    }
    if params.light_kind == LIGHT_HEADLIGHT {
        return -normalize_or(params.view_direction.xyz, vec3<f32>(0.0, 0.0, 1.0));
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

fn to_view_space(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(v, params.u_axis.xyz),
        dot(v, params.v_axis.xyz),
        dot(v, params.view_direction.xyz),
    );
}

fn lambert(base: vec3<f32>, normal_ws: vec3<f32>, ka: vec3<f32>, kd: vec3<f32>) -> vec3<f32> {
    let normal_vs = normalize_or(to_view_space(normal_ws), vec3<f32>(0.0, 0.0, 1.0));
    let ambient = base * ka * params.global_ambient.xyz;
    var diffuse = vec3<f32>(0.0);
    if params.light_kind == LIGHT_POINT {
        let count = min(params.light_count, MAX_POINT_LIGHTS);
        for (var i = 0u; i < count; i = i + 1u) {
            let light = normalize_or(to_view_space(params.point_lights[i].xyz), vec3<f32>(0.0, 0.0, 1.0));
            diffuse = diffuse + base * kd * max(dot(normal_vs, light), 0.0);
        }
    } else {
        let light = light_direction(normal_vs);
        diffuse = base * kd * max(dot(normal_vs, light), 0.0);
    }
    return ambient + diffuse;
}

fn blinn_phong(
    base: vec3<f32>,
    position_ws: vec3<f32>,
    normal_ws: vec3<f32>,
    ka: vec3<f32>,
    kd: vec3<f32>,
    ks_ns: vec4<f32>,
) -> vec3<f32> {
    let normal_vs = normalize_or(to_view_space(normal_ws), vec3<f32>(0.0, 0.0, 1.0));
    let position_vs = to_view_space(position_ws);
    let view_dir = normalize_or(-position_vs, vec3<f32>(0.0, 0.0, 1.0));
    let ambient = base * ka * params.global_ambient.xyz;
    var diffuse = vec3<f32>(0.0);
    var specular = vec3<f32>(0.0);
    if params.light_kind == LIGHT_POINT {
        let count = min(params.light_count, MAX_POINT_LIGHTS);
        for (var i = 0u; i < count; i = i + 1u) {
            let light = normalize_or(to_view_space(params.point_lights[i].xyz), vec3<f32>(0.0, 0.0, 1.0));
            let dot_nl = max(dot(normal_vs, light), 0.0);
            diffuse = diffuse + base * kd * dot_nl * params.global_diffuse.xyz;
            let halfway = normalize_or(light + view_dir, normal_vs);
            let dot_nh = max(dot(normal_vs, halfway), 0.0);
            let spec = select(0.0, pow(dot_nh, ks_ns.w), ks_ns.w > 0.0);
            specular = specular + ks_ns.xyz * spec * params.global_specular.xyz;
        }
    } else {
        let light = light_direction(normal_vs);
        let dot_nl = max(dot(normal_vs, light), 0.0);
        diffuse = base * kd * dot_nl * params.global_diffuse.xyz;
        let halfway = normalize_or(light + view_dir, normal_vs);
        let dot_nh = max(dot(normal_vs, halfway), 0.0);
        let spec = select(0.0, pow(dot_nh, ks_ns.w), ks_ns.w > 0.0);
        specular = ks_ns.xyz * spec * params.global_specular.xyz;
    }
    return ambient + diffuse + specular;
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    let material = materials[input.material_id];
    let mode = u32(material.mode_illum_opacity.x);
    let illum = u32(material.mode_illum_opacity.y);
    let opacity = clamp(material.mode_illum_opacity.z, 0.0, 1.0);
    let depth_value = input.depth;

    var base = vec3<f32>(1.0);
    if mode == SHADING_VERTEX_COLOR {
        base = max(input.color, vec3<f32>(0.0));
    } else if mode == SHADING_TEXTURE_MAPPED || mode == SHADING_BLINN_PHONG {
        base = sample_texture_bilinear(material.tex, input.uv);
    }

    var color = base;
    if illum == 1u {
        color = lambert(base, input.normal_ws, material.ka.xyz, material.kd.xyz);
    } else if illum > 1u {
        color = blinn_phong(base, input.position_ws, input.normal_ws, material.ka.xyz, material.kd.xyz, material.ks_ns);
    }
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    let srgb = vec3<f32>(
        linear_to_srgb(color.x),
        linear_to_srgb(color.y),
        linear_to_srgb(color.z),
    );

    let near = min(params.depth.x, params.depth.y);
    let far = max(params.depth.x, params.depth.y);
    let depth_byte = clamp((far - depth_value) / max(far - near, 0.000001), 0.0, 1.0);
    let normal_vs = normalize_or(to_view_space(input.normal_ws), vec3<f32>(0.0, 0.0, 1.0));
    let normal_encoded = clamp((normal_vs + vec3<f32>(1.0)) * 0.5, vec3<f32>(0.0), vec3<f32>(1.0));

    out.color = vec4<f32>(srgb, opacity);
    out.depth_byte = vec4<f32>(depth_byte, 0.0, 0.0, 1.0);
    out.normal = vec4<f32>(normal_encoded, 1.0);
    out.occupancy = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    out.raw_depth = depth_value;
    return out;
}
