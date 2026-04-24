const WORKGROUP_SIZE: u32 = 16u;
const PROJECTION_ORTHOGRAPHIC: u32 = 0u;

struct SsaoParams {
    resolution: u32,
    num_samples: u32,
    blur_radius: u32,
    projection_kind: u32,
    sample_radius: f32,
    bias: f32,
    intensity: f32,
    max_distance: f32,
    focal_length: f32,
    cube_center: f32,
    pad0: vec2<f32>,
};

@group(0) @binding(0) var raw_depth: texture_2d<f32>;
@group(0) @binding(1) var normal_map: texture_2d<f32>;
@group(0) @binding(2) var occupancy_map: texture_2d<f32>;
@group(0) @binding(3) var occlusion_out: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var<uniform> occlusion_params: SsaoParams;

@group(1) @binding(0) var blur_src: texture_2d<f32>;
@group(1) @binding(1) var blur_out: texture_storage_2d<r32float, write>;
@group(1) @binding(2) var<uniform> blur_params: SsaoParams;

@group(2) @binding(0) var color_src: texture_2d<f32>;
@group(2) @binding(1) var darken_occupancy: texture_2d<f32>;
@group(2) @binding(2) var darken_occlusion: texture_2d<f32>;
@group(2) @binding(3) var color_out: texture_storage_2d<rgba8unorm, write>;
@group(2) @binding(4) var<uniform> darken_params: SsaoParams;

fn srgb_to_linear(x: f32) -> f32 {
    return pow(max(x, 0.0), 2.2);
}

fn linear_to_srgb(x: f32) -> f32 {
    return pow(clamp(x, 0.0, 1.0), 1.0 / 2.2);
}

fn hash_noise(coord: vec2<u32>) -> f32 {
    var h = coord.x * 73856093u ^ coord.y * 19349663u;
    h = h * 83492791u;
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    h = h * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return f32(h) / 4294967295.0 * 6.28318530718;
}

fn sample_vector(i: u32, count: u32) -> vec3<f32> {
    let fi = f32(i);
    let fc = max(f32(count), 1.0);
    let u1 = (fi + 0.5) / fc;
    let u2 = fract(fi * 0.618034);
    let cos_theta = sqrt(u1);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let phi = 6.28318530718 * u2;
    let scale = 0.1 + 0.9 * u1;
    return vec3<f32>(sin_theta * cos(phi) * scale, sin_theta * sin(phi) * scale, cos_theta * scale);
}

fn normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len2 = dot(v, v);
    if len2 > 0.000001 {
        return v * inverseSqrt(len2);
    }
    return fallback;
}

fn reconstruct_position(coord: vec2<u32>, depth: f32, params: SsaoParams) -> vec3<f32> {
    let pixel = vec2<f32>(f32(coord.x) + 0.5, f32(coord.y) + 0.5);
    let resolution = f32(params.resolution);
    if params.projection_kind == PROJECTION_ORTHOGRAPHIC {
        return vec3<f32>(
            pixel.x / resolution * 2.0 - 1.0,
            pixel.y / resolution * 2.0 - 1.0,
            depth,
        );
    }
    let x_cam = (pixel.x - params.cube_center) * depth / max(params.focal_length, 0.000001);
    let y_cam = (pixel.y - params.cube_center) * depth / max(params.focal_length, 0.000001);
    return vec3<f32>(x_cam, y_cam, depth);
}

fn project_position(position: vec3<f32>, params: SsaoParams) -> vec2<i32> {
    let resolution = f32(params.resolution);
    if params.projection_kind == PROJECTION_ORTHOGRAPHIC {
        return vec2<i32>(
            i32(round((position.x + 1.0) * 0.5 * resolution)),
            i32(round((position.y + 1.0) * 0.5 * resolution)),
        );
    }
    if position.z <= 0.0 {
        return vec2<i32>(-1, -1);
    }
    let scale = params.focal_length / position.z;
    return vec2<i32>(
        i32(round(scale * position.x + params.cube_center)),
        i32(round(scale * position.y + params.cube_center)),
    );
}

fn tangent_basis(normal: vec3<f32>) -> mat3x3<f32> {
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(normal.z) < 0.9);
    let tangent = normalize_or(cross(normal, up), vec3<f32>(1.0, 0.0, 0.0));
    let bitangent = cross(normal, tangent);
    return mat3x3<f32>(tangent, bitangent, normal);
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn compute_occlusion(@builtin(global_invocation_id) id: vec3<u32>) {
    let params = occlusion_params;
    if id.x >= params.resolution || id.y >= params.resolution {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    if textureLoad(occupancy_map, coord, 0).r <= 0.0 {
        textureStore(occlusion_out, coord, vec4<f32>(0.0));
        return;
    }
    let depth = textureLoad(raw_depth, coord, 0).r;
    if depth <= 0.0 || params.num_samples == 0u || params.sample_radius <= 0.0 || params.max_distance <= 0.0 {
        textureStore(occlusion_out, coord, vec4<f32>(0.0));
        return;
    }
    let view_pos = reconstruct_position(id.xy, depth, params);
    let normal_encoded = textureLoad(normal_map, coord, 0).rgb;
    let normal = normalize_or(normal_encoded * 2.0 - vec3<f32>(1.0), vec3<f32>(0.0, 0.0, 1.0));
    let basis = tangent_basis(normal);
    let rotation = hash_noise(id.xy);
    let cs = cos(rotation);
    let sn = sin(rotation);
    var occlusion = 0.0;
    var valid = 0u;
    for (var i = 0u; i < params.num_samples; i = i + 1u) {
        var sample = sample_vector(i, params.num_samples);
        sample = vec3<f32>(sample.x * cs - sample.y * sn, sample.x * sn + sample.y * cs, sample.z);
        let sample_pos = view_pos + basis * sample * params.sample_radius;
        let sample_coord = project_position(sample_pos, params);
        if sample_coord.x < 0 || sample_coord.y < 0 || sample_coord.x >= i32(params.resolution) || sample_coord.y >= i32(params.resolution) {
            continue;
        }
        if textureLoad(occupancy_map, sample_coord, 0).r <= 0.0 {
            continue;
        }
        let sample_depth = textureLoad(raw_depth, sample_coord, 0).r;
        let dist = length(sample_pos - view_pos);
        if sample_depth <= 0.0 || dist > params.max_distance {
            continue;
        }
        if sample_depth < sample_pos.z - params.bias {
            let d = dist / params.max_distance;
            occlusion = occlusion + max(1.0 - d * d, 0.0);
        }
        valid = valid + 1u;
    }
    let value = select(0.0, occlusion / f32(valid), valid > 0u);
    textureStore(occlusion_out, coord, vec4<f32>(clamp(value, 0.0, 1.0)));
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn blur_occlusion(@builtin(global_invocation_id) id: vec3<u32>) {
    let params = blur_params;
    if id.x >= params.resolution || id.y >= params.resolution {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    let radius = i32(params.blur_radius);
    let center = textureLoad(blur_src, coord, 0).r;
    var sum = 0.0;
    var weight_sum = 0.0;
    let sigma_color_sq = 0.25;
    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            let pos = coord + vec2<i32>(dx, dy);
            if pos.x < 0 || pos.y < 0 || pos.x >= i32(params.resolution) || pos.y >= i32(params.resolution) {
                continue;
            }
            let v = textureLoad(blur_src, pos, 0).r;
            let spatial = f32(dx * dx + dy * dy);
            let r = max(f32(radius), 1.0);
            let sw = exp(-spatial / (2.0 * r * r));
            let diff = center - v;
            let cw = exp(-(diff * diff) / (2.0 * sigma_color_sq));
            let w = sw * cw;
            sum = sum + v * w;
            weight_sum = weight_sum + w;
        }
    }
    textureStore(blur_out, coord, vec4<f32>(select(center, sum / weight_sum, weight_sum > 0.0)));
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn darken_color(@builtin(global_invocation_id) id: vec3<u32>) {
    let params = darken_params;
    if id.x >= params.resolution || id.y >= params.resolution {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    var color = textureLoad(color_src, coord, 0);
    if textureLoad(darken_occupancy, coord, 0).r > 0.0 {
        let occ = clamp(textureLoad(darken_occlusion, coord, 0).r * params.intensity, 0.0, 1.0);
        let factor = 1.0 - occ;
        let lin = vec3<f32>(
            srgb_to_linear(color.r) * factor,
            srgb_to_linear(color.g) * factor,
            srgb_to_linear(color.b) * factor,
        );
        color = vec4<f32>(linear_to_srgb(lin.r), linear_to_srgb(lin.g), linear_to_srgb(lin.b), color.a);
    }
    textureStore(color_out, coord, color);
}
