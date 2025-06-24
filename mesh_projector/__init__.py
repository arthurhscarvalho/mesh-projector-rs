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


class Mesh:
    """
    Represents a mesh object for use with the mesh projector.

    This class serves as a Pythonic wrapper around the internal mesh representation used by the Rust backend.
    It is intended to provide a convenient interface for mesh operations within the mesh projector package.
    Users should not instantiate this class directly; instead, obtain instances via the `read_mesh` function.
    This class is intended for internal use within the mesh projector and is not designed for use outside this context.
    """

    def __init__(self, mesh: MeshWrapper):
        self._internal_mesh = mesh


def read_mesh(
    obj_path: str,
    texture_path: Optional[str] = None,
    mtl_path: Optional[str] = None,
    precision: int = 12,
    flip_uv: bool = True,
    ignore_opacity: bool = False,
    verbose: bool = False,
) -> Mesh:
    try:
        internal_mesh = _internal_read_mesh(
            obj_path,
            texture_path,
            mtl_path,
            precision,
            flip_uv,
            ignore_opacity,
            verbose,
        )
        return Mesh(internal_mesh)
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during mesh reading: {e}")


def project_to_image_on_bounding_cube(
    mesh: Mesh,
    filtering: int = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        return _internal_project_to_image_on_bounding_cube(
            mesh._internal_mesh,
            filtering,
            verbose,
        )
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during projection generation: {e}")


def project_to_image_by_rotation(
    mesh: Mesh,
    rotation_steps: int,
    angle_delta_x: float,
    angle_delta_y: float,
    angle_delta_z: float,
    filtering: int = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        return _internal_project_to_image_by_rotation(
            mesh._internal_mesh,
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
    mesh: Mesh,
    rotation_steps: int,
    angle_delta_x: float,
    angle_delta_y: float,
    angle_delta_z: float,
    filtering: int = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        return _internal_project_to_image_by_rotation_to_file(
            mesh._internal_mesh,
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
    mesh: Mesh,
    point_cloud_density_factor: float = 1.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        return _internal_project_to_point_cloud_by_sampling(
            mesh._internal_mesh,
            point_cloud_density_factor,
            verbose,
        )
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during point cloud generation: {e}")
