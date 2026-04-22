const WORKGROUP_SIZE: u32 = 16u;

struct ResolveParams {
    resolution: u32,
    sample_count: u32,
    pad0: vec2<u32>,
};

@group(0) @binding(0) var raw_depth_msaa: texture_multisampled_2d<f32>;
@group(0) @binding(1) var occupancy_msaa: texture_multisampled_2d<f32>;
@group(0) @binding(2) var raw_depth_out: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> params: ResolveParams;

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn resolve_raw_depth(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution || id.y >= params.resolution {
        return;
    }

    let coord = vec2<i32>(i32(id.x), i32(id.y));
    var best_depth = 3.402823e38;
    for (var i = 0u; i < params.sample_count; i = i + 1u) {
        if textureLoad(occupancy_msaa, coord, i).r <= 0.0 {
            continue;
        }
        let raw_depth = textureLoad(raw_depth_msaa, coord, i).r;
        if raw_depth < best_depth {
            best_depth = raw_depth;
        }
    }
    if best_depth == 3.402823e38 {
        best_depth = 0.0;
    }

    textureStore(raw_depth_out, coord, vec4<f32>(best_depth, 0.0, 0.0, 1.0));
}
