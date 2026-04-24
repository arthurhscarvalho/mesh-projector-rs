const WORKGROUP_SIZE: u32 = 256u;

struct PackParams {
    resolution: u32,
    pixels_per_view: u32,
    view_index: u32,
    selection_bits: u32,
    dispatch_width: u32,
    color_stride_words: u32,
    scalar_words_per_view: u32,
    scalar_stride_words: u32,
};

@group(0) @binding(0) var color_tex: texture_2d<f32>;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var occupancy_tex: texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> color_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> depth_out: array<u32>;
@group(0) @binding(6) var<storage, read_write> normal_out: array<u32>;
@group(0) @binding(7) var<storage, read_write> occupancy_out: array<u32>;
@group(0) @binding(8) var<uniform> params: PackParams;

fn byte_from_unit(x: f32) -> u32 {
    return u32(round(clamp(x, 0.0, 1.0) * 255.0)) & 255u;
}

fn pack_rgba(v: vec4<f32>) -> u32 {
    return byte_from_unit(v.r)
        | (byte_from_unit(v.g) << 8u)
        | (byte_from_unit(v.b) << 16u)
        | (byte_from_unit(v.a) << 24u);
}

fn scalar_coord(pixel: u32) -> vec2<i32> {
    let x = pixel % params.resolution;
    let y = pixel / params.resolution;
    return vec2<i32>(i32(x), i32(y));
}

fn depth_at(pixel: u32) -> u32 {
    if pixel >= params.pixels_per_view {
        return 0u;
    }
    return byte_from_unit(textureLoad(depth_tex, scalar_coord(pixel), 0).r);
}

fn occupancy_at(pixel: u32) -> u32 {
    if pixel >= params.pixels_per_view {
        return 0u;
    }
    return byte_from_unit(textureLoad(occupancy_tex, scalar_coord(pixel), 0).r);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn pack_outputs(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel = id.x + id.y * params.dispatch_width;
    if pixel >= params.pixels_per_view {
        return;
    }
    let x = pixel % params.resolution;
    let y = pixel / params.resolution;
    let coord = vec2<i32>(i32(x), i32(y));
    let color_base = 0u;
    let scalar_base = 0u;

    if (params.selection_bits & 1u) != 0u {
        color_out[color_base + pixel] = pack_rgba(textureLoad(color_tex, coord, 0));
    }
    if (params.selection_bits & 4u) != 0u {
        normal_out[color_base + pixel] = pack_rgba(textureLoad(normal_tex, coord, 0));
    }

    if (pixel & 3u) == 0u {
        let word = pixel / 4u;
        if (params.selection_bits & 2u) != 0u {
            let packed_depth = depth_at(pixel)
                | (depth_at(pixel + 1u) << 8u)
                | (depth_at(pixel + 2u) << 16u)
                | (depth_at(pixel + 3u) << 24u);
            depth_out[scalar_base + word] = packed_depth;
        }
        if (params.selection_bits & 8u) != 0u {
            let packed_occupancy = occupancy_at(pixel)
                | (occupancy_at(pixel + 1u) << 8u)
                | (occupancy_at(pixel + 2u) << 16u)
                | (occupancy_at(pixel + 3u) << 24u);
            occupancy_out[scalar_base + word] = packed_occupancy;
        }
    }
}
