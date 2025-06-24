from typing import Optional, Tuple

import numpy as np

from .mesh_projector import MeshWrapper
from .mesh_projector import (
    project_to_image_by_rotation as _internal_project_to_image_by_rotation,
)
from .mesh_projector import (
    project_to_image_by_rotation_to_file as _internal_project_to_image_by_rotation_to_file,
)
from .mesh_projector import (
    project_to_image_on_bounding_cube as _internal_project_to_image_on_bounding_cube,
)
from .mesh_projector import (
    project_to_point_cloud_by_sampling as _internal_project_to_point_cloud_by_sampling,
)
from .mesh_projector import (
    read_mesh as _internal_read_mesh,
)


def read_mesh(
    obj_path: str,
    texture_path: Optional[str] = None,
    mtl_path: Optional[str] = None,
    precision: int = 12,
    flip_uv: bool = True,
    ignore_opacity: bool = False,
    verbose: bool = False,
) -> MeshWrapper:
    try:
        return _internal_read_mesh(
            obj_path,
            texture_path,
            mtl_path,
            precision,
            flip_uv,
            ignore_opacity,
            verbose,
        )
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during mesh reading: {e}")


def project_to_image_on_bounding_cube(
    mesh: MeshWrapper,
    filtering: int = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        return _internal_project_to_image_on_bounding_cube(
            mesh,
            filtering,
            verbose,
        )
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during projection generation: {e}")


def project_to_image_by_rotation(
    mesh: MeshWrapper,
    rotation_steps: int,
    angle_delta_x: float,
    angle_delta_y: float,
    angle_delta_z: float,
    filtering: int = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        return _internal_project_to_image_by_rotation(
            mesh,
            rotation_steps,
            angle_delta_x,
            angle_delta_y,
            angle_delta_z,
            filtering,
            verbose,
        )
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during projection generation: {e}")


def project_to_image_by_rotation_to_file(
    mesh: MeshWrapper,
    rotation_steps: int,
    angle_delta_x: float,
    angle_delta_y: float,
    angle_delta_z: float,
    filtering: int = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        return _internal_project_to_image_by_rotation_to_file(
            mesh,
            rotation_steps,
            angle_delta_x,
            angle_delta_y,
            angle_delta_z,
            filtering,
            verbose,
        )
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during projection generation: {e}")


def project_to_point_cloud_by_sampling(
    mesh: MeshWrapper,
    point_cloud_density_factor: float = 1.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        return _internal_project_to_point_cloud_by_sampling(
            mesh,
            point_cloud_density_factor,
            verbose,
        )
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during point cloud generation: {e}")
