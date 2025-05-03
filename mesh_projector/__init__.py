import os
from enum import Enum

import cv2
import numpy as np
import open3d as o3d
from loguru import logger

from .mesh_projector import (
    generate_projections as _internal_generate_projections,
)


# Enum for the different modes of projection
class ProjectionMode(Enum):
    OBJ_ONLY = "obj_only"
    OBJ_TEXTURE = "obj_texture"
    FULL = "full"  # Corresponds to OBJ+Texture+MTL


class PyMesh:
    """Python representation of the Rust PyMesh struct defined in src/projector/mesh.rs"""

    def __init__(
        self,
        # Geometry
        vertices: np.ndarray = None,
        triangles: np.ndarray = None,
        vertex_normals: np.ndarray = None,
        triangle_normals: np.ndarray = None,
        # Texture
        vertex_colors: np.ndarray = None,
        texture: np.ndarray = None,
        triangle_uvs: np.ndarray = None,
        # Material
        ka: list = [0.0, 0.0, 0.0],
        kd: list = [1.0, 1.0, 1.0],
        ks: list = [0.0, 0.0, 0.0],
        ns: float = 10.0,
        opacity: float = 1.0,
        illum: int = 1,
    ):
        # Geometry
        self.vertices = vertices
        self.triangles = triangles
        self.vertex_normals = vertex_normals
        self.triangle_normals = triangle_normals
        # Texture
        self.vertex_colors = vertex_colors
        self.texture = texture
        self.triangle_uvs = triangle_uvs
        # Material
        self.ka = ka
        self.kd = kd
        self.ks = ks
        self.ns = ns
        self.opacity = opacity
        self.illum = illum


def info(msg):
    logger.info(msg)


def _parse_mtl(mtl_path, ignore_opacity: bool = False):
    """Parses relevant properties from an MTL file."""
    logger.debug(f"Attempting to parse MTL: {mtl_path}")
    # Default material if file not found or not provided
    default_material = {
        "ka": [0.0, 0.0, 0.0],
        "kd": [1.0, 1.0, 1.0],
        "ks": [0.0, 0.0, 0.0],
        "ns": 10.0,
        "opacity": 1.0,
        "illum": 1,
    }
    if not mtl_path or not os.path.exists(mtl_path):
        logger.warning(
            f"MTL file not provided or not found: {mtl_path}. Using default material values."
        )
        return default_material
    # Start with reasonable defaults before parsing
    material = {
        "ka": [0.1, 0.1, 0.1],
        "kd": [0.8, 0.8, 0.8],
        "ks": [0.1, 0.1, 0.1],
        "ns": 10.0,
        "opacity": 1.0,
        "illum": 2,
    }
    parsed_successfully = False
    try:
        with open(mtl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                key = parts[0].lower()
                values = parts[1:]

                if key == "ka" and len(values) == 3:
                    material["ka"] = [float(v) for v in values]
                elif key == "kd" and len(values) == 3:
                    material["kd"] = [float(v) for v in values]
                elif key == "ks" and len(values) == 3:
                    material["ks"] = [float(v) for v in values]
                elif key == "ns" and len(values) == 1:
                    material["ns"] = float(values[0])
                elif key == "tr" and len(values) == 1:
                    material["opacity"] = 1.0 - float(values[0])
                elif key == "d" and len(values) == 1:
                    material["opacity"] = float(values[0])
                elif key == "illum" and len(values) == 1:
                    material["illum"] = int(values[0])
        material["opacity"] = max(0.0, min(1.0, material["opacity"]))
        parsed_successfully = True
        # Override opacity if requested
        if ignore_opacity:
            logger.info("Ignoring MTL opacity and setting it to 1.0.")
            material["opacity"] = 1.0
    except Exception as e:
        logger.error(f"Error parsing MTL file {mtl_path}: {e}. Using default values.")
        material = default_material
    # Log final parsed/default values
    if parsed_successfully:
        logger.info(f"Successfully parsed MTL '{mtl_path}': {material}")
    else:
        logger.warning(f"Failed to parse MTL '{mtl_path}', using defaults: {material}")
    return material


def read_mesh(
    obj_location,
    texture_file=None,
    mtl_location=None,
    enable_post_processing=False,
    mode=ProjectionMode.FULL,
    verbose=False,
):
    """
    Loads a mesh and optionally its texture based on the mode.

    Args:
        obj_location (str): Path to the OBJ mesh file.
        texture_file (str, optional): Path to the texture image file. Required for OBJ_TEXTURE and FULL modes. Defaults to None.
        mtl_location (str, optional): Path to the MTL material file. Used only in FULL mode. Defaults to None.
        enable_post_processing (bool): Enables Open3D post-processing.
        mode (ProjectionMode): The processing mode (OBJ_ONLY, OBJ_TEXTURE, FULL).
        verbose (bool): Enables detailed logging output.

    Returns:
        o3d.geometry.TriangleMesh: The loaded mesh, possibly with texture attached.

    Raises:
        ValueError, FileNotFoundError: On failure to load or missing components required by the mode.
    """
    if verbose is not True:
        logger.remove()
    info(f"Loading mesh: {obj_location} in mode: {mode.value}")
    info(f"Post-processing enabled: {enable_post_processing}")
    mesh = o3d.io.read_triangle_mesh(
        obj_location,
        enable_post_processing=enable_post_processing,
        print_progress=verbose,
    )
    if not mesh.has_vertices():
        raise ValueError(f"Mesh '{obj_location}' has no vertices!")
    # Normals are generally useful for all modes if available, compute if missing.
    if not mesh.has_vertex_normals():
        logger.warning(
            f"Mesh '{obj_location}' does not have vertex normals. "
            "Attempting to compute vertex normals."
        )
        mesh.compute_vertex_normals()
        if not mesh.has_vertex_normals():
            # Decide if this is critical depending on the mode or downstream usage
            logger.warning(
                f"Failed to compute vertex normals for mesh '{obj_location}'. Shading might be affected."
            )
        else:
            logger.info("Successfully computed vertex normals.")
    # Handle texture and UVs based on mode
    if mode in [ProjectionMode.OBJ_TEXTURE, ProjectionMode.FULL]:
        if not texture_file or not os.path.exists(texture_file):
            raise FileNotFoundError(
                f"Texture file '{texture_file}' is required for mode '{mode.value}' but not found."
            )
        if not mesh.has_triangle_uvs():
            raise ValueError(
                f"Mesh '{obj_location}' has no UV coordinates, required for mode '{mode.value}'."
            )
        try:
            texture = cv2.cvtColor(cv2.imread(texture_file), cv2.COLOR_BGR2RGB)
            mesh.textures = [o3d.geometry.Image(texture)]
            info(f"Texture load successful: {texture_file}")
        except Exception as e:
            raise ValueError(f"Failed to load texture '{texture_file}': {e}")
    elif mode == ProjectionMode.OBJ_ONLY:
        # In OBJ_ONLY mode, texture is not loaded. Check for vertex colors.
        if mesh.has_vertex_colors():
            info("Mesh has vertex colors.")
        else:
            logger.warning(
                f"Mesh '{obj_location}' does not have vertex colors for OBJ_ONLY mode. Will need default color."
            )
        # Clear any textures that might have been loaded automatically by open3d based on MTL reference
        mesh.textures = []
    # MTL file is only relevant for FULL mode, parsed later in generate_projections
    return mesh


def preprocess_mesh(
    mesh,
    py_mesh,
    precision,
    mode=ProjectionMode.FULL,
    flip_v_coordinate=True,
    verbose=False,
):
    """
    Processes a mesh by scaling vertices and preparing data based on the mode.

    Args:
        mesh (o3d.geometry.TriangleMesh): Loaded mesh.
        py_mesh (PyMesh): PyMesh object to update with processed mesh data.
        precision (int): Output image resolution exponent (2^precision).
        mode (ProjectionMode): The processing mode.
        flip_v_coordinate (bool): Whether to flip the V coordinate (1.0 - v).
                                Defaults to True, assuming texture V=0 is top.
        verbose (bool): Enables detailed logging.

    Returns:
        PyMesh: The updated PyMesh object.
    """
    if verbose is not True:
        logger.remove()

    # --- Vertex Data (Common) ---
    vertices = np.asarray(mesh.vertices, dtype=np.double)
    triangle_indices = np.asarray(mesh.triangles, dtype=np.uint64)
    if triangle_indices.size == 0:
        raise ValueError("Mesh has vertices but no triangles!")
    # --- Normals (Common, if available) ---
    vertex_normals = None
    if mesh.has_vertex_normals():
        vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.double)
        if vertices.shape[0] != vertex_normals.shape[0]:
            raise ValueError(
                f"Vertex count mismatch: Vertices ({vertices.shape[0]}) vs. Normals ({vertex_normals.shape[0]})"
            )
        # Ensure normals are normalized
        norm_norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        zero_norms_mask = norm_norms.flatten() < 1e-9
        if np.any(zero_norms_mask):
            logger.warning(
                f"Found {np.sum(zero_norms_mask)} zero-length normals. Setting them to [0, 0, 1]."
            )
            vertex_normals[zero_norms_mask] = [0.0, 0.0, 1.0]
            norm_norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
            norm_norms[norm_norms < 1e-9] = (
                1.0  # Avoid division by zero for default normals
            )
        # Normalize the normals (result is still float64)
        vertex_normals = vertex_normals / norm_norms
    else:
        # Handle case where normals couldn't be computed/loaded
        logger.warning("Mesh lacks vertex normals. Proceeding without them.")
    # Initialize optional texture-related variables
    texture_np = None
    triangle_uvs = None
    vertex_colors = None
    # --- Mode-Specific Data ---
    if mode in [ProjectionMode.OBJ_TEXTURE, ProjectionMode.FULL]:
        if not mesh.textures:
            raise ValueError(
                f"Mode '{mode.value}' requires textures, but none are loaded on the mesh."
            )
        if not mesh.has_triangle_uvs():
            raise ValueError(
                f"Mode '{mode.value}' requires UVs, but none are found on the mesh."
            )

        texture_np = np.asarray(mesh.textures[0], dtype=np.uint8)
        if texture_np.ndim != 3 or texture_np.shape[2] < 3:
            raise ValueError(
                f"Texture has shape {texture_np.shape}. Expected 3 channels (RGB)."
            )
        tex_height, tex_width, tex_channels = texture_np.shape
        info(f"Texture shape: H={tex_height}, W={tex_width}, C={tex_channels}")
        triangle_uvs = np.asarray(mesh.triangle_uvs, dtype=np.double)
        if triangle_uvs.shape[1] != 2:
            raise ValueError(
                f"Expected triangle_uvs shape to be (*, 2), but got {triangle_uvs.shape}"
            )
        if triangle_uvs.shape[0] != triangle_indices.size:
            raise ValueError(
                f"UV mismatch: UVs ({triangle_uvs.shape[0]}) vs. triangles ({triangle_indices.size})"
            )
        triangle_uvs = np.clip(triangle_uvs, 0.0, 1.0)
        if flip_v_coordinate:
            info("Flipping V coordinate (1.0 - v) [Default Behavior]")
            triangle_uvs[:, 1] = 1.0 - triangle_uvs[:, 1]
        else:
            info("Using original V coordinates [Flipping Disabled]")
    elif mode == ProjectionMode.OBJ_ONLY:
        if mesh.has_vertex_colors():
            vertex_colors = np.asarray(mesh.vertex_colors, dtype=np.double)
            if vertex_colors.shape[0] != vertices.shape[0]:
                raise ValueError(
                    f"Vertex color mismatch: Vertices ({vertices.shape[0]}) vs. Colors ({vertex_colors.shape[0]})"
                )
            info(f"Vertex colors shape: {vertex_colors.shape}")
        else:
            vertex_colors = np.zeros_like(vertices, dtype=np.double)
            info("No vertex colors found for OBJ_ONLY mode. Using default black.")
    # --- Vertex Scaling (Common) ---
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    info(f"Original BBox Min: {min_bound}, Max: {max_bound}")
    shift = np.maximum(0, -min_bound)
    if np.any(shift > 0):
        info(f"Shifting vertices by: {shift}")
        vertices += shift
        min_bound = vertices.min(axis=0)
        max_bound = vertices.max(axis=0)
        info(f"Shifted BBox Min: {min_bound}, Max: {max_bound}")
    bbox_size = max_bound - min_bound
    bbox_size[bbox_size == 0] = 1e-9
    max_dim = np.max(bbox_size)
    target_max_val = (1 << precision) - 1.0
    scale_factor = target_max_val / max_dim
    info(f"Scaling vertices. Max dimension: {max_dim}, Scale factor: {scale_factor}")
    vertices *= scale_factor
    min_bound_scaled = vertices.min(axis=0)
    max_bound_scaled = vertices.max(axis=0)
    info(f"Scaled BBox Min: {min_bound_scaled}, Max: {max_bound_scaled}")
    # Update the PyMesh object with the processed data
    py_mesh.vertices = vertices
    py_mesh.triangles = triangle_indices
    py_mesh.vertex_normals = vertex_normals
    py_mesh.vertex_colors = vertex_colors
    py_mesh.texture = texture_np
    py_mesh.triangle_uvs = triangle_uvs
    return py_mesh


def generate_projections(
    py_mesh,
    precision,
    mtl_location=None,
    mode=ProjectionMode.FULL,
    filtering=0,
    verbose=False,
    ignore_opacity: bool = False,
):
    """
    Generates 6 orthogonal projections and corresponding outlier occlusion maps of a mesh using the Rust backend.

    Args:
        py_mesh (PyMesh): PyMesh object containing mesh data.
        precision (int): Output image resolution exponent (2^precision).
        mtl_location (str, optional): Path to the MTL file. Used only in FULL mode. Defaults to None.
        mode (ProjectionMode): The processing mode (OBJ_ONLY, OBJ_TEXTURE, FULL).
        filtering (int, optional): Radius for outlier filtering. If > 0, occlusion maps are generated.
                                  Defaults to 0 (disabled).
        verbose (bool): Enables detailed logging.
        ignore_opacity (bool): If True, forces opacity to 1.0, ignoring the MTL value.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]:
            - A list of 6 numpy arrays, each representing a projection image (H, W, 3).
            - A list of 6 numpy arrays, each representing an outlier occlusion map (H, W).
              Value is 1 for filtered outlier pixels, 0 otherwise. Returns empty maps if filtering=0.

    Raises:
        ValueError: If required data for the selected mode is missing.
        RuntimeError: If the Rust backend fails.
    """
    if verbose is not True:
        logger.remove()
    # --- Material Handling based on Mode ---
    if mode == ProjectionMode.FULL:
        info(f"Using FULL mode. Parsing MTL: {mtl_location}")
        material_props = _parse_mtl(mtl_location, ignore_opacity=ignore_opacity)
        py_mesh.ka = material_props["ka"]
        py_mesh.kd = material_props["kd"]
        py_mesh.ks = material_props["ks"]
        py_mesh.ns = material_props["ns"]
        py_mesh.opacity = material_props["opacity"]
        py_mesh.illum = material_props["illum"]
    elif mode == ProjectionMode.OBJ_TEXTURE:
        info("Using OBJ_TEXTURE mode. Applying default material for texturing.")
        py_mesh.ka = [0.0, 0.0, 0.0]  # No Ambient
        py_mesh.kd = [1.0, 1.0, 1.0]  # Base color modulation (texture is base)
        py_mesh.ks = [0.0, 0.0, 0.0]  # No Specular
        py_mesh.ns = 10.0  # Default Shininess (unused if Ks=0)
        py_mesh.opacity = 1.0  # Fully Opaque
        py_mesh.illum = 0  # Color = ShadingColor (no lighting calc)
    elif mode == ProjectionMode.OBJ_ONLY:
        info("Using OBJ_ONLY mode. Applying default material for vertex colors.")
        py_mesh.ka = [0.0, 0.0, 0.0]  # No Ambient term from material
        py_mesh.kd = [1.0, 1.0, 1.0]  # Base color modulation (vertex color is base)
        py_mesh.ks = [0.0, 0.0, 0.0]  # No Specular
        py_mesh.ns = 10.0
        py_mesh.opacity = 1.0
        py_mesh.illum = 0  # Color = ShadingColor (no lighting calculation)
    else:
        raise ValueError(f"Unknown ProjectionMode: {mode}")
    # --- Validate Data Requirements per Mode ---
    if py_mesh.vertices is None or py_mesh.triangles is None:
        raise ValueError("Vertices and triangle_indices are always required.")
    # Validate array shapes for the Rust backend
    if py_mesh.vertices.ndim != 2 or py_mesh.vertices.shape[1] != 3:
        raise ValueError(
            f"Vertices must be a 2D array with shape (N, 3), got {py_mesh.vertices.shape}"
        )
    if py_mesh.triangles.ndim != 2 or py_mesh.triangles.shape[1] != 3:
        raise ValueError(
            f"Triangle indices must be a 2D array with shape (M, 3), got {py_mesh.triangles.shape}"
        )
    # Validate optional arrays if they exist
    if py_mesh.vertex_normals is not None and (
        py_mesh.vertex_normals.ndim != 2 or py_mesh.vertex_normals.shape[1] != 3
    ):
        raise ValueError(
            f"Vertex normals must be a 2D array with shape (N, 3), got {py_mesh.vertex_normals.shape}"
        )
    if py_mesh.vertex_colors is not None and (
        py_mesh.vertex_colors.ndim != 2 or py_mesh.vertex_colors.shape[1] != 3
    ):
        raise ValueError(
            f"Vertex colors must be a 2D array with shape (N, 3), got {py_mesh.vertex_colors.shape}"
        )
    if py_mesh.triangle_uvs is not None and (
        py_mesh.triangle_uvs.ndim != 2 or py_mesh.triangle_uvs.shape[1] != 2
    ):
        raise ValueError(
            f"Triangle UVs must be a 2D array with shape (K, 2), got {py_mesh.triangle_uvs.shape}"
        )
    if py_mesh.texture is not None and py_mesh.texture.ndim != 3:
        raise ValueError(
            f"Texture must be a 3D array with shape (H, W, C), got {py_mesh.texture.shape}"
        )
    # Normals are preferred for lighting, but might be optional depending on Rust impl.
    if (
        py_mesh.vertex_normals is None
        and mode != ProjectionMode.OBJ_ONLY
        and py_mesh.illum > 0
    ):
        logger.warning(
            "Vertex normals are missing. Lighting/Shading might be incorrect."
        )
    if mode in [ProjectionMode.OBJ_TEXTURE, ProjectionMode.FULL]:
        if py_mesh.texture is None or py_mesh.triangle_uvs is None:
            raise ValueError(f"Mode '{mode.value}' requires texture and triangle_uvs.")
    elif mode == ProjectionMode.OBJ_ONLY:
        if py_mesh.vertex_colors is None:
            logger.warning(
                "Vertex colors are missing for OBJ_ONLY mode. Using default black."
            )
            # Create default black vertex colors if none exist and ensure float64 type
            py_mesh.vertex_colors = np.zeros_like(py_mesh.vertices, dtype=np.double)
    # Convert mode enum to string for Rust FFI
    mode_str = mode.value
    # <<< --- Log final properties before Rust call --- >>>
    logger.info("--- Calling Rust Backend ---")
    logger.info(f"Mode: {mode_str}")
    logger.info(
        f"Material Properties: ka={py_mesh.ka}, kd={py_mesh.kd}, ks={py_mesh.ks}, ns={py_mesh.ns}, opacity={py_mesh.opacity}, illum={py_mesh.illum}"
    )
    logger.info(f"Texture provided: {py_mesh.texture is not None}")
    logger.info(f"UVs provided: {py_mesh.triangle_uvs is not None}")
    logger.info(f"Vertex Normals provided: {py_mesh.vertex_normals is not None}")
    logger.info(f"Vertex Colors provided: {py_mesh.vertex_colors is not None}")
    logger.info("--------------------------")
    # <<< --- End logging --- >>>
    try:
        # Call the Rust backend - now returns a tuple
        projection_array, occlusion_array = _internal_generate_projections(
            py_mesh,
            precision,
            filtering,  # Pass filtering radius
            mode_str,
            verbose,
        )
        # Convert the 4D arrays to lists of images/maps
        # Projection Images (6, H, W, 3)
        projection_images = [
            projection_array[i] for i in range(projection_array.shape[0])
        ]
        # Occlusion Maps (6, H, W, 1) -> need to squeeze to (6, H, W)
        occlusion_maps = [
            np.squeeze(occlusion_array[i], axis=-1)
            for i in range(occlusion_array.shape[0])
        ]
    except Exception as e:
        # Log the specific error from the Rust backend if possible
        logger.error(f"Error calling Rust backend: {e}")
        raise RuntimeError(f"Rust backend failed during projection generation: {e}")
    if len(projection_images) != 6:
        logger.warning(f"Expected 6 projections, but received {len(projection_images)}")
    if len(occlusion_maps) != 6:
        logger.warning(f"Expected 6 occlusion maps, but received {len(occlusion_maps)}")
    info(
        f"Generated {len(projection_images)} projections and {len(occlusion_maps)} occlusion maps."
    )
    return projection_images, occlusion_maps


def project_mesh(
    obj_location,
    precision,
    texture_file=None,
    mtl_location=None,
    mode=None,  # If None, will be inferred
    filtering=0,
    flip_uv=True,
    verbose=False,
    enable_post_processing=True,
    ignore_opacity: bool = False,
):
    """
    High-level function to load, preprocess, and generate projections and occlusion maps for a mesh.

    Args:
        obj_location (str): Path to the OBJ mesh file.
        precision (int): Output image resolution exponent (2^precision).
        texture_file (str, optional): Path to the texture image file. Defaults to None.
        mtl_location (str, optional): Path to the MTL material file. Defaults to None.
        mode (ProjectionMode or str, optional): The processing mode ('obj_only', 'obj_texture', 'full').
                                                If None, it will be inferred based on provided files.
                                                Defaults to None.
        filtering (int, optional): Radius for outlier filtering. If > 0, occlusion maps are generated.
                                  Defaults to 0 (disabled).
        flip_uv (bool): Whether to flip the V coordinate (1.0 - v) during preprocessing.
                        Defaults to True.
        verbose (bool): Enables detailed logging. Defaults to False.
        enable_post_processing (bool): Enables Open3D post-processing during mesh loading.
                                      Defaults to False.
        ignore_opacity (bool): If True, forces opacity to 1.0, ignoring the MTL value. Defaults to False.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]:
            - A list of 6 numpy arrays, each representing a projection image (H, W, 3).
            - A list of 6 numpy arrays, each representing an outlier occlusion map (H, W).
              Value is 1 for filtered outlier pixels, 0 otherwise. Returns empty maps if filtering=0.
            Returns None for both lists if any step fails (deprecated, now raises exceptions).

    Raises:
        FileNotFoundError: If required files (OBJ, or Texture/MTL based on mode) are not found.
        ValueError: If the mesh data is invalid or required components are missing for the mode.
        RuntimeError: If the Rust backend fails during projection generation.
    """
    if verbose is not True:
        logger.remove()

    # --- Determine Mode ---
    if mode:
        if isinstance(mode, str):
            try:
                resolved_mode = ProjectionMode(mode)
            except ValueError:
                raise ValueError(
                    f"Invalid mode string specified: {mode}. Choices are: {[e.value for e in ProjectionMode]}"
                )
        elif isinstance(mode, ProjectionMode):
            resolved_mode = mode
        else:
            raise ValueError(
                f"Invalid type for mode: {type(mode)}. Must be str or ProjectionMode."
            )
        logger.info(f"Using specified mode: {resolved_mode.value}")
    else:
        # Infer mode if not explicitly set
        if texture_file and mtl_location:
            resolved_mode = ProjectionMode.FULL
            logger.info("Inferred mode: FULL (OBJ, Texture, MTL provided)")
        elif texture_file:
            resolved_mode = ProjectionMode.OBJ_TEXTURE
            logger.info("Inferred mode: OBJ_TEXTURE (OBJ, Texture provided)")
        else:
            resolved_mode = ProjectionMode.OBJ_ONLY
            logger.info("Inferred mode: OBJ_ONLY (Only OBJ provided)")

    # --- Validate required files for chosen mode ---
    if not os.path.exists(obj_location):
        raise FileNotFoundError(f"OBJ file not found: {obj_location}")
    if resolved_mode in [ProjectionMode.OBJ_TEXTURE, ProjectionMode.FULL]:
        if not texture_file:
            raise FileNotFoundError(
                f"Mode '{resolved_mode.value}' requires a texture file, but none was provided."
            )
        if not os.path.exists(texture_file):
            raise FileNotFoundError(f"Texture file not found: {texture_file}")
    if resolved_mode == ProjectionMode.FULL:
        if not mtl_location:
            raise FileNotFoundError(
                f"Mode '{resolved_mode.value}' requires an MTL file, but none was provided."
            )
        if not os.path.exists(mtl_location):
            # Don't raise error, _parse_mtl handles missing file with defaults
            logger.warning(
                f"MTL file not found: {mtl_location}. Proceeding with default material properties."
            )
    # --- Load Mesh ---
    # Errors (FileNotFoundError, ValueError) will propagate up
    mesh = read_mesh(
        obj_location=obj_location,
        texture_file=texture_file,
        mtl_location=mtl_location,  # Pass it even if None/missing, read_mesh handles it
        mode=resolved_mode,
        enable_post_processing=enable_post_processing,
        verbose=verbose,
    )
    # --- Initialize PyMesh with default values ---
    # Everything will be populated in the preprocess_mesh step
    py_mesh = PyMesh(
        vertices=None,
        triangles=None,
        # Default material properties
        ka=[0.0, 0.0, 0.0],
        kd=[1.0, 1.0, 1.0],
        ks=[0.0, 0.0, 0.0],
        ns=10.0,
        opacity=1.0,
        illum=1,
    )
    # --- Preprocess Mesh ---
    # This will populate the py_mesh with geometry and texture data
    preprocess_mesh(
        mesh,
        py_mesh,
        precision=precision,
        mode=resolved_mode,
        flip_v_coordinate=flip_uv,
        verbose=verbose,
    )
    # --- Generate Projections ---
    # This will update material properties in py_mesh and then generate projections
    projection_images, occlusion_maps = generate_projections(
        py_mesh=py_mesh,
        precision=precision,
        mtl_location=mtl_location
        if resolved_mode == ProjectionMode.FULL
        else None,  # Only pass MTL for FULL mode
        mode=resolved_mode,
        filtering=filtering,  # Pass filtering radius
        verbose=verbose,
        ignore_opacity=ignore_opacity,
    )
    return projection_images, occlusion_maps
