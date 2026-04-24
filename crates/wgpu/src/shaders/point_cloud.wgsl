const MAX_POINT_LIGHTS: u32 = 16u;
const SHADING_VERTEX_COLOR: u32 = 0u;
const SHADING_TEXTURE_MAPPED: u32 = 1u;
const SHADING_BLINN_PHONG: u32 = 2u;
const LIGHT_AMBIENT: u32 = 0u;
const LIGHT_DIRECTIONAL: u32 = 1u;
const LIGHT_HEADLIGHT: u32 = 2u;
const LIGHT_POINT: u32 = 3u;

struct Triangle {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    n0: vec4<f32>,
    n1: vec4<f32>,
    n2: vec4<f32>,
    uv0: vec4<f32>,
    uv1: vec4<f32>,
    uv2: vec4<f32>,
    c0: vec4<f32>,
    c1: vec4<f32>,
    c2: vec4<f32>,
    ka: vec4<f32>,
    kd: vec4<f32>,
    ks_ns: vec4<f32>,
};

struct TriangleMeta {
    tex_offset: u32,
    tex_width: u32,
    tex_height: u32,
    tex_channels: u32,
    n: u32,
    shading_mode: u32,
    illum: u32,
    _pad0: u32,
};

struct Params {
    global_ambient: vec4<f32>,
    global_diffuse: vec4<f32>,
    global_specular: vec4<f32>,
    directional_light: vec4<f32>,
    view_direction: vec4<f32>,
    point_lights: array<vec4<f32>, 16>,
    light_kind: u32,
    light_count: u32,
    triangle_count: u32,
    total_points: u32,
};

@group(0) @binding(0) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(1) var<storage, read> metas: array<TriangleMeta>;
@group(0) @binding(2) var<storage, read> texels: array<u32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_positions: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> out_colors: array<u32>;
@group(0) @binding(6) var<uniform> params: Params;

fn srgb_byte_to_linear(b: u32) -> f32 {
    let x = f32(b) / 255.0;
    return pow(x, 2.2);
}

fn linear_to_srgb_byte(x: f32) -> u32 {
    let clamped = clamp(x, 0.0, 1.0);
    let srgb = pow(clamped, 1.0 / 2.2) * 255.0;
    return u32(round(clamp(srgb, 0.0, 255.0)));
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

fn pack_srgb(color: vec3<f32>) -> u32 {
    let c = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    let r = linear_to_srgb_byte(c.x);
    let g = linear_to_srgb_byte(c.y);
    let b = linear_to_srgb_byte(c.z);
    return r | (g << 8u) | (b << 16u);
}

fn sample_texture_bilinear(tri_meta: TriangleMeta, uv: vec2<f32>) -> vec3<f32> {
    if tri_meta.tex_width == 0u || tri_meta.tex_height == 0u || tri_meta.tex_channels < 3u {
        return vec3<f32>(1.0, 1.0, 1.0);
    }

    let width_f = f32(tri_meta.tex_width);
    let height_f = f32(tri_meta.tex_height);
    let tex_xf = uv.x * width_f;
    let tex_yf = uv.y * height_f;
    let x0f = clamp(tex_xf, 0.0, width_f - 1.0001);
    let y0f = clamp(tex_yf, 0.0, height_f - 1.0001);
    let x0 = u32(x0f);
    let y0 = u32(y0f);
    let x1 = min(x0 + 1u, tri_meta.tex_width - 1u);
    let y1 = min(y0 + 1u, tri_meta.tex_height - 1u);
    let wx = x0f - f32(x0);
    let wy = y0f - f32(y0);

    let row0 = tri_meta.tex_offset + y0 * tri_meta.tex_width;
    let row1 = tri_meta.tex_offset + y1 * tri_meta.tex_width;
    let c00 = unpack_texel_linear(texels[row0 + x0]);
    let c10 = unpack_texel_linear(texels[row0 + x1]);
    let c01 = unpack_texel_linear(texels[row1 + x0]);
    let c11 = unpack_texel_linear(texels[row1 + x1]);
    return c00 * (1.0 - wx) * (1.0 - wy)
        + c10 * wx * (1.0 - wy)
        + c01 * (1.0 - wx) * wy
        + c11 * wx * wy;
}

fn normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len2 = dot(v, v);
    if len2 > 0.0 {
        return v * inverseSqrt(len2);
    }
    return fallback;
}

fn single_light_direction(normal_vs: vec3<f32>) -> vec3<f32> {
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

fn lambert(base: vec3<f32>, normal_ws: vec3<f32>, tri: Triangle) -> vec3<f32> {
    let ambient = base * tri.ka.xyz * params.global_ambient.xyz;
    var diffuse = vec3<f32>(0.0);
    if params.light_kind == LIGHT_POINT {
        let count = min(params.light_count, MAX_POINT_LIGHTS);
        for (var i = 0u; i < count; i = i + 1u) {
            let light_dir = normalize_or(params.point_lights[i].xyz, vec3<f32>(0.0, 0.0, 1.0));
            diffuse = diffuse + base * tri.kd.xyz * max(dot(normal_ws, light_dir), 0.0);
        }
    } else {
        let light_dir = single_light_direction(normal_ws);
        diffuse = base * tri.kd.xyz * max(dot(normal_ws, light_dir), 0.0);
    }
    return ambient + diffuse;
}

fn blinn_phong(base: vec3<f32>, position_ws: vec3<f32>, normal_ws: vec3<f32>, tri: Triangle) -> vec3<f32> {
    let n = normalize_or(normal_ws, vec3<f32>(0.0, 0.0, 1.0));
    let view_dir = normalize_or(-position_ws, vec3<f32>(0.0, 0.0, 1.0));
    let ambient = base * tri.ka.xyz * params.global_ambient.xyz;
    var diffuse = vec3<f32>(0.0);
    var specular = vec3<f32>(0.0);
    let shininess = tri.ks_ns.w;

    if params.light_kind == LIGHT_POINT {
        let count = min(params.light_count, MAX_POINT_LIGHTS);
        for (var i = 0u; i < count; i = i + 1u) {
            let light_dir = normalize_or(params.point_lights[i].xyz, vec3<f32>(0.0, 0.0, 1.0));
            let dot_nl = max(dot(n, light_dir), 0.0);
            diffuse = diffuse + base * tri.kd.xyz * dot_nl * params.global_diffuse.xyz;
            let halfway = normalize_or(light_dir + view_dir, n);
            let dot_nh = max(dot(n, halfway), 0.0);
            let spec = select(0.0, pow(dot_nh, shininess), shininess > 0.0);
            specular = specular + tri.ks_ns.xyz * spec * params.global_specular.xyz;
        }
    } else {
        let light_dir = single_light_direction(n);
        let dot_nl = max(dot(n, light_dir), 0.0);
        diffuse = base * tri.kd.xyz * dot_nl * params.global_diffuse.xyz;
        let halfway = normalize_or(light_dir + view_dir, n);
        let dot_nh = max(dot(n, halfway), 0.0);
        let spec = select(0.0, pow(dot_nh, shininess), shininess > 0.0);
        specular = tri.ks_ns.xyz * spec * params.global_specular.xyz;
    }

    return ambient + diffuse + specular;
}

fn find_triangle_index(point_index: u32) -> u32 {
    var lo = 0u;
    var hi = params.triangle_count;
    loop {
        if lo + 1u >= hi {
            break;
        }
        let mid = (lo + hi) / 2u;
        if offsets[mid] <= point_index {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

fn find_i_for_local(local_index: u32, n: u32) -> u32 {
    var lo = 0u;
    var hi = n + 1u;
    loop {
        if lo + 1u >= hi {
            break;
        }
        let mid = (lo + hi) / 2u;
        let before_mid = mid * (n + 1u) - (mid * (mid - 1u)) / 2u;
        if before_mid <= local_index {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let point_index = global_id.x;
    if point_index >= params.total_points {
        return;
    }

    let tri_index = find_triangle_index(point_index);
    let tri = triangles[tri_index];
    let tri_meta = metas[tri_index];
    let local_index = point_index - offsets[tri_index];
    let n = tri_meta.n;
    let i = find_i_for_local(local_index, n);
    let before_i = i * (n + 1u) - (i * (i - 1u)) / 2u;
    let j = local_index - before_i;
    let k = n - i - j;
    let denom = f32(n) + 1.0;
    let w0 = (f32(i) + 1.0 / 3.0) / denom;
    let w1 = (f32(j) + 1.0 / 3.0) / denom;
    let w2 = (f32(k) + 1.0 / 3.0) / denom;

    let position = tri.v0.xyz * w0 + tri.v1.xyz * w1 + tri.v2.xyz * w2;
    let normal = normalize_or(tri.n0.xyz * w0 + tri.n1.xyz * w1 + tri.n2.xyz * w2, vec3<f32>(0.0, 0.0, 1.0));
    let uv = tri.uv0.xy * w0 + tri.uv1.xy * w1 + tri.uv2.xy * w2;

    var base = vec3<f32>(1.0);
    if tri_meta.shading_mode == SHADING_VERTEX_COLOR {
        base = tri.c0.xyz * w0 + tri.c1.xyz * w1 + tri.c2.xyz * w2;
        base = max(base, vec3<f32>(0.0));
    } else if tri_meta.shading_mode == SHADING_TEXTURE_MAPPED || tri_meta.shading_mode == SHADING_BLINN_PHONG {
        base = sample_texture_bilinear(tri_meta, uv);
    }

    var lit = base;
    if tri_meta.illum == 1u {
        lit = lambert(base, normal, tri);
    } else if tri_meta.illum > 1u {
        lit = blinn_phong(base, position, normal, tri);
    }

    out_positions[point_index] = vec4<f32>(position, 0.0);
    out_colors[point_index] = pack_srgb(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)));
}
