//! Per-view image projection preparation.

use crate::error::{WgpuProjectorError, WgpuResult};
use crate::gpu::constants::{
    vec4, LIGHT_AMBIENT, LIGHT_DIRECTIONAL, LIGHT_HEADLIGHT, LIGHT_POINT, MAX_POINT_LIGHTS,
};
use crate::image::prepared_scene::WgpuPreparedScene;
use bytemuck::{Pod, Zeroable};
use mesh_projector_core::config::{
    CameraConfig, LightSource, LightingConfig, ProjectionConfig, ProjectionType,
};
use mesh_projector_core::geometry::{create_view_matrix, is_triangle_visible};
use mesh_projector_core::types::coordinate::Coordinate;
use mesh_projector_core::types::orientation::Orientation;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuImageParams {
    pub(crate) global_ambient: [f32; 4],
    pub(crate) global_diffuse: [f32; 4],
    pub(crate) global_specular: [f32; 4],
    pub(crate) directional_light: [f32; 4],
    pub(crate) view_direction: [f32; 4],
    pub(crate) u_axis: [f32; 4],
    pub(crate) v_axis: [f32; 4],
    pub(crate) camera_position: [f32; 4],
    pub(crate) point_lights: [[f32; 4]; MAX_POINT_LIGHTS],
    pub(crate) background: [f32; 4],
    pub(crate) projection: [f32; 4],
    pub(crate) depth: [f32; 4],
    pub(crate) light_kind: u32,
    pub(crate) light_count: u32,
    pub(crate) _pad0: u32,
    pub(crate) projection_kind: u32,
}

pub(crate) struct PreparedImageView {
    pub(crate) indices: Vec<u32>,
    pub(crate) params: GpuImageParams,
}

pub(crate) fn prepare_prepared_view(
    prepared: &WgpuPreparedScene,
    settings: &ProjectionConfig,
    lighting_config: &LightingConfig,
    orientation: &Orientation,
) -> WgpuResult<PreparedImageView> {
    let (params, view_direction, camera_position, projection_kind) =
        build_image_view_params(settings, lighting_config, orientation)?;
    let mut indices = Vec::with_capacity(prepared.triangles.len() * 3);
    for triangle in &prepared.triangles {
        let visible = if projection_kind == 0 {
            is_triangle_visible(triangle.face_normal, view_direction)
        } else {
            let view_vec = triangle.centroid - camera_position;
            triangle.face_normal.dot(&view_vec) < 0.0
        };
        if visible {
            indices.extend_from_slice(&[
                triangle.first_corner,
                triangle.first_corner + 1,
                triangle.first_corner + 2,
            ]);
        }
    }
    Ok(PreparedImageView { indices, params })
}

fn build_image_view_params(
    settings: &ProjectionConfig,
    lighting_config: &LightingConfig,
    orientation: &Orientation,
) -> WgpuResult<(GpuImageParams, Coordinate, Coordinate, u32)> {
    let rotation_matrix = orientation.to_rotation_matrix();
    let view_matrix = create_view_matrix(&rotation_matrix);
    let view_direction = Coordinate::new(
        rotation_matrix[[0, 2]],
        rotation_matrix[[1, 2]],
        rotation_matrix[[2, 2]],
    );
    let u_axis = Coordinate::new(
        view_matrix[[0, 0]],
        view_matrix[[1, 0]],
        view_matrix[[2, 0]],
    );
    let v_axis = Coordinate::new(
        view_matrix[[0, 1]],
        view_matrix[[1, 1]],
        view_matrix[[2, 1]],
    );
    let max_bound = settings.resolution as f32;
    let cube_center = (max_bound - 1.0) * 0.5;
    let cube_center_coordinate = Coordinate::new(cube_center, cube_center, cube_center);
    let offset_u = cube_center_coordinate.dot(&u_axis) - cube_center;
    let offset_v = cube_center_coordinate.dot(&v_axis) - cube_center;
    let perspective = perspective_params(&settings.camera, view_direction, max_bound);
    let (projection_kind, focal_length, camera_position, z_near, z_far) = match perspective {
        Some(params) => (1, params.1, params.0, params.2, params.3),
        None => {
            let max_depth = max_bound
                * (view_direction.x().max(0.0)
                    + view_direction.y().max(0.0)
                    + view_direction.z().max(0.0));
            let min_depth = max_bound
                * (view_direction.x().min(0.0)
                    + view_direction.y().min(0.0)
                    + view_direction.z().min(0.0));
            (0, 0.0, Coordinate::new(0.0, 0.0, 0.0), min_depth, max_depth)
        }
    };

    let mut params = build_image_params(lighting_config)?;
    params.view_direction = vec4(view_direction.0);
    params.u_axis = vec4(u_axis.0);
    params.v_axis = vec4(v_axis.0);
    params.camera_position = vec4(camera_position.0);
    params.background = background_vec4(settings.background.to_srgba8());
    params.projection = [offset_u, offset_v, cube_center, focal_length];
    params.depth = [z_near, z_far, max_bound, 0.0];
    params.projection_kind = projection_kind;
    Ok((params, view_direction, camera_position, projection_kind))
}

fn perspective_params(
    camera: &CameraConfig,
    view_direction: Coordinate,
    max_bound: f32,
) -> Option<(Coordinate, f32, f32, f32)> {
    let ProjectionType::Perspective(perspective_cfg) = &camera.projection else {
        return None;
    };
    let cube_center = (max_bound - 1.0) * 0.5;
    let cube_center_coordinate = Coordinate::new(cube_center, cube_center, cube_center);
    let half_fov_rad = 0.5 * perspective_cfg.vertical_fov_degrees.to_radians();
    let tan_half_fov = half_fov_rad.tan();
    let focal_length = if tan_half_fov.abs() < 1e-6 {
        max_bound * 0.5
    } else {
        (max_bound * 0.5) / tan_half_fov
    };
    let camera_position = cube_center_coordinate - view_direction * (cube_center + focal_length);
    let z_near = focal_length;
    let z_far = focal_length + 2.0 * cube_center;
    Some((camera_position, focal_length, z_near, z_far))
}

fn build_image_params(lighting_config: &LightingConfig) -> WgpuResult<GpuImageParams> {
    let mut params = GpuImageParams {
        global_ambient: vec4(lighting_config.components.global_ambient.0),
        global_diffuse: vec4(lighting_config.components.global_diffuse.0),
        global_specular: vec4(lighting_config.components.global_specular.0),
        directional_light: [0.0, 0.0, 1.0, 0.0],
        view_direction: [0.0, 0.0, 1.0, 0.0],
        u_axis: [1.0, 0.0, 0.0, 0.0],
        v_axis: [0.0, 1.0, 0.0, 0.0],
        camera_position: [0.0, 0.0, 0.0, 0.0],
        point_lights: [[0.0; 4]; MAX_POINT_LIGHTS],
        background: [1.0, 1.0, 1.0, 1.0],
        projection: [0.0, 0.0, 0.0, 0.0],
        depth: [0.0, 1.0, 1.0, 0.0],
        light_kind: LIGHT_AMBIENT,
        light_count: 0,
        _pad0: 0,
        projection_kind: 0,
    };

    match &lighting_config.source {
        LightSource::Ambient => {
            params.light_kind = LIGHT_AMBIENT;
        }
        LightSource::Directional(direction) => {
            params.light_kind = LIGHT_DIRECTIONAL;
            params.directional_light = vec4(direction.0);
        }
        LightSource::Headlight => {
            params.light_kind = LIGHT_HEADLIGHT;
        }
        LightSource::Point(points) => {
            if points.len() > MAX_POINT_LIGHTS {
                return Err(WgpuProjectorError::TooManyPointLights {
                    count: points.len(),
                    max: MAX_POINT_LIGHTS,
                });
            }
            params.light_kind = LIGHT_POINT;
            params.light_count = points.len() as u32;
            for (index, point) in points.iter().enumerate() {
                params.point_lights[index] = vec4(point.0);
            }
        }
    }
    Ok(params)
}

fn background_vec4(background: [u8; 4]) -> [f32; 4] {
    [
        background[0] as f32 / 255.0,
        background[1] as f32 / 255.0,
        background[2] as f32 / 255.0,
        background[3] as f32 / 255.0,
    ]
}
