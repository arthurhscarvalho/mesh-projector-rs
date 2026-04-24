const WORKGROUP_SIZE: u32 = 16u;

struct ResolveParams {
    resolution: u32,
    sample_count: u32,
    pad0: vec2<u32>,
};

@group(0) @binding(0) var color_msaa: texture_multisampled_2d<f32>;
@group(0) @binding(1) var depth_byte_msaa: texture_multisampled_2d<f32>;
@group(0) @binding(2) var normal_msaa: texture_multisampled_2d<f32>;
@group(0) @binding(3) var occupancy_msaa: texture_multisampled_2d<f32>;
@group(0) @binding(4) var raw_depth_msaa: texture_multisampled_2d<f32>;
@group(0) @binding(5) var color_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var depth_byte_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(7) var normal_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(8) var occupancy_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(9) var<uniform> params: ResolveParams;

fn srgb_to_linear(x: f32) -> f32 {
    return pow(clamp(x, 0.0, 1.0), 2.2);
}

fn linear_to_srgb(x: f32) -> f32 {
    return pow(clamp(x, 0.0, 1.0), 1.0 / 2.2);
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn resolve_msaa_gbuffer(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution || id.y >= params.resolution {
        return;
    }

    let coord = vec2<i32>(i32(id.x), i32(id.y));
    var color_linear_sum = vec3<f32>(0.0);
    var alpha_sum = 0.0;
    var occupied_count = 0u;
    var best_depth = 3.402823e38;
    var best_depth_byte = 0.0;
    var best_normal = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    for (var i = 0u; i < params.sample_count; i = i + 1u) {
        let occ = textureLoad(occupancy_msaa, coord, i).r;
        if occ <= 0.0 {
            continue;
        }
        occupied_count = occupied_count + 1u;

        let color = textureLoad(color_msaa, coord, i);
        let alpha = clamp(color.a, 0.0, 1.0);
        color_linear_sum = color_linear_sum + vec3<f32>(
            srgb_to_linear(color.r),
            srgb_to_linear(color.g),
            srgb_to_linear(color.b),
        ) * alpha;
        alpha_sum = alpha_sum + alpha;

        let raw_depth = textureLoad(raw_depth_msaa, coord, i).r;
        if raw_depth < best_depth {
            best_depth = raw_depth;
            best_depth_byte = textureLoad(depth_byte_msaa, coord, i).r;
            best_normal = textureLoad(normal_msaa, coord, i);
        }
    }

    let sample_count_f = max(f32(params.sample_count), 1.0);
    var resolved_color = vec4<f32>(0.0);
    if alpha_sum > 0.0 {
        let rgb = color_linear_sum / alpha_sum;
        resolved_color = vec4<f32>(
            linear_to_srgb(rgb.r),
            linear_to_srgb(rgb.g),
            linear_to_srgb(rgb.b),
            clamp(alpha_sum / sample_count_f, 0.0, 1.0),
        );
    }

    let occupancy = select(0.0, 1.0, occupied_count > 0u);
    if best_depth == 3.402823e38 {
        best_depth = 0.0;
        best_depth_byte = 0.0;
        best_normal = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    textureStore(color_out, coord, resolved_color);
    textureStore(depth_byte_out, coord, vec4<f32>(best_depth_byte, 0.0, 0.0, 1.0));
    textureStore(normal_out, coord, best_normal);
    textureStore(occupancy_out, coord, vec4<f32>(occupancy, 0.0, 0.0, 1.0));
}
