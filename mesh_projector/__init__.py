import os
from enum import Enum

import cv2
import numpy as np
import open3d as o3d
from loguru import logger

from .mesh_projector import (
    generate_projections as _internal_generate_projections,
)


# Define an Enum for the modes
class ProjectionMode(Enum):
    OBJ_ONLY = "obj_only"
    OBJ_TEXTURE = "obj_texture"
    FULL = "full"  # Corresponds to OBJ+Texture+MTL


def info(msg):
    logger.info(msg)


def _parse_mtl(mtl_path):
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
    mesh, precision, mode=ProjectionMode.FULL, flip_v_coordinate=True, verbose=False
):
    """
    Processes a mesh by scaling vertices and preparing data based on the mode.

    Args:
        mesh (o3d.geometry.TriangleMesh): Loaded mesh.
        precision (int): Output image resolution exponent (2^precision).
        mode (ProjectionMode): The processing mode.
        flip_v_coordinate (bool): Whether to flip the V coordinate (1.0 - v).
                                Defaults to True, assuming texture V=0 is top.
        verbose (bool): Enables detailed logging.

    Returns:
        dict: A dictionary containing processed mesh data:
            - 'vertices': Scaled vertex positions (N, 3).
            - 'triangle_indices': Triangle vertex indices (M, 3).
            - 'vertex_normals': Vertex normal vectors (N, 3).
            - 'texture_np' (optional): Texture image data (H, W, C), if mode requires texture.
            - 'triangle_uvs' (optional): Per-triangle UV coordinates (M*3, 2), if mode requires texture.
            - 'vertex_colors' (optional): Vertex color data (N, 3), if mode is OBJ_ONLY and colors exist.
    """
    if verbose is not True:
        logger.remove()

    processed_data = {}

    # --- Vertex Data (Common) ---
    vertices = np.asarray(mesh.vertices)
    triangle_indices = np.asarray(
        mesh.triangles, dtype=np.uint32
    )  # Changed to uint32 based on previous check
    if triangle_indices.size == 0:
        raise ValueError("Mesh has vertices but no triangles!")

    # --- Normals (Common, if available) ---
    if mesh.has_vertex_normals():
        vertex_normals = np.asarray(mesh.vertex_normals)
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

        processed_data["vertex_normals"] = vertex_normals / norm_norms
    else:
        # Handle case where normals couldn't be computed/loaded
        logger.warning("Mesh lacks vertex normals. Proceeding without them.")
        # Create placeholder normals if needed by Rust, or handle absence there
        # For now, just don't include them in the dict
        # processed_data['vertex_normals'] = np.zeros_like(vertices, dtype=np.float32) # Example placeholder
        processed_data["vertex_normals"] = None  # Indicate absence

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

        texture_np = np.asarray(mesh.textures[0])
        if texture_np.ndim != 3 or texture_np.shape[2] < 3:
            raise ValueError(
                f"Texture has shape {texture_np.shape}. Expected 3 channels (RGB)."
            )
        tex_height, tex_width, tex_channels = texture_np.shape
        info(f"Texture shape: H={tex_height}, W={tex_width}, C={tex_channels}")
        processed_data["texture_np"] = texture_np

        triangle_uvs = np.asarray(mesh.triangle_uvs)
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
        processed_data["triangle_uvs"] = triangle_uvs.astype(
            np.float32
        )  # Ensure float32

    elif mode == ProjectionMode.OBJ_ONLY:
        if mesh.has_vertex_colors():
            vertex_colors = np.asarray(mesh.vertex_colors)
            if vertex_colors.shape[0] != vertices.shape[0]:
                raise ValueError(
                    f"Vertex color mismatch: Vertices ({vertices.shape[0]}) vs. Colors ({vertex_colors.shape[0]})"
                )
            processed_data["vertex_colors"] = vertex_colors.astype(
                np.float32
            )  # Ensure float32
            info(f"Vertex colors shape: {vertex_colors.shape}")
        else:
            processed_data["vertex_colors"] = None  # Indicate absence
            info("No vertex colors found for OBJ_ONLY mode.")
        # Ensure texture and UVs are not included
        processed_data["texture_np"] = None
        processed_data["triangle_uvs"] = None

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
    bbox_size[bbox_size == 0] = 1e-6
    max_dim = np.max(bbox_size)
    target_max_val = (1 << precision) - 1.0
    scale_factor = target_max_val / max_dim
    info(f"Scaling vertices. Max dimension: {max_dim}, Scale factor: {scale_factor}")
    vertices *= scale_factor
    min_bound_scaled = vertices.min(axis=0)
    max_bound_scaled = vertices.max(axis=0)
    info(f"Scaled BBox Min: {min_bound_scaled}, Max: {max_bound_scaled}")

    processed_data["vertices"] = vertices.astype(np.float32)  # Ensure float32
    processed_data["triangle_indices"] = triangle_indices  # Already uint32

    # Final Logging
    final_shapes = {
        k: (v.shape if hasattr(v, "shape") else type(v))
        for k, v in processed_data.items()
        if v is not None
    }
    info(f"Final Processed Data Shapes: {final_shapes}")

    return processed_data


def generate_projections(
    processed_data,
    precision,
    mtl_location=None,
    mode=ProjectionMode.FULL,
    filtering=0,
    verbose=False,
):
    """
    Generates 6 orthogonal projections of a mesh using the Rust backend.

    Args:
        processed_data (dict): Result from preprocess_mesh, containing vertices,
                               indices, normals, and mode-specific data (texture/UVs or colors).
        precision (int): Output image resolution exponent (2^precision).
        mtl_location (str, optional): Path to the MTL file. Used only in FULL mode. Defaults to None.
        mode (ProjectionMode): The processing mode (OBJ_ONLY, OBJ_TEXTURE, FULL).
        filtering (int, optional): Radius for outlier filtering. Defaults to 0 (disabled).
        verbose (bool): Enables detailed logging.

    Returns:
        List[np.ndarray]: A list of 6 numpy arrays, each representing a projection image.

    Raises:
        ValueError: If required data for the selected mode is missing in processed_data.
        RuntimeError: If the Rust backend fails.
    """
    if verbose is not True:
        logger.remove()

    # --- Material Handling based on Mode ---
    if mode == ProjectionMode.FULL:
        info(f"Using FULL mode. Parsing MTL: {mtl_location}")
        material_props = _parse_mtl(mtl_location)
    elif mode == ProjectionMode.OBJ_TEXTURE:
        info("Using OBJ_TEXTURE mode. Applying default material for texturing.")
        material_props = {
            "ka": [0.0, 0.0, 0.0],  # No Ambient
            "kd": [1.0, 1.0, 1.0],  # Base color modulation (texture is base)
            "ks": [0.0, 0.0, 0.0],  # No Specular
            "ns": 10.0,  # Default Shininess (unused if Ks=0)
            "opacity": 1.0,  # Fully Opaque
            # Use illum 0: Color = ShadingColor (base color, no lighting calc)
            "illum": 0,  # CHANGED from 1 to 0
        }
    elif mode == ProjectionMode.OBJ_ONLY:
        info("Using OBJ_ONLY mode. Applying default material for vertex colors.")
        material_props = {
            "ka": [0.0, 0.0, 0.0],  # No Ambient term from material
            "kd": [1.0, 1.0, 1.0],  # Base color modulation (vertex color is base)
            "ks": [0.0, 0.0, 0.0],  # No Specular
            "ns": 10.0,
            "opacity": 1.0,
            "illum": 0,  # Color = ShadingColor (no lighting calculation)
        }
    else:
        raise ValueError(f"Unknown ProjectionMode: {mode}")

    # --- Unpack Processed Data ---
    vertices = processed_data.get("vertices")
    triangle_indices = processed_data.get("triangle_indices")
    vertex_normals = processed_data.get("vertex_normals")  # Can be None
    texture_np = processed_data.get("texture_np")  # Can be None
    triangle_uvs = processed_data.get("triangle_uvs")  # Can be None
    vertex_colors = processed_data.get("vertex_colors")  # Can be None

    # --- Validate Data Requirements per Mode ---
    if vertices is None or triangle_indices is None:
        raise ValueError("Vertices and triangle_indices are always required.")
    # Normals are preferred for lighting, but might be optional depending on Rust impl.
    if (
        vertex_normals is None
        and mode != ProjectionMode.OBJ_ONLY
        and material_props["illum"] > 0
    ):
        logger.warning(
            "Vertex normals are missing. Lighting/Shading might be incorrect."
        )
        # Consider generating default normals if Rust requires them:
        # vertex_normals = np.zeros_like(vertices, dtype=np.float32)
        # vertex_normals[:, 2] = 1.0 # Point up Z-axis

    if mode in [ProjectionMode.OBJ_TEXTURE, ProjectionMode.FULL]:
        if texture_np is None or triangle_uvs is None:
            raise ValueError(
                f"Mode '{mode.value}' requires texture_np and triangle_uvs."
            )
    elif mode == ProjectionMode.OBJ_ONLY:
        if vertex_colors is None:
            logger.warning(
                "Vertex colors are missing for OBJ_ONLY mode. Using default black."
            )
            # Create default black vertex colors if none exist
            vertex_colors = np.zeros_like(vertices, dtype=np.float32)

    # Convert mode enum to string for Rust FFI
    mode_str = mode.value

    # <<< --- Log final properties before Rust call --- >>>
    logger.info("--- Calling Rust Backend ---")
    logger.info(f"Mode: {mode_str}")
    logger.info(f"Material Properties Sent: {material_props}")
    logger.info(f"Texture provided: {texture_np is not None}")
    logger.info(f"UVs provided: {triangle_uvs is not None}")
    logger.info(f"Vertex Normals provided: {vertex_normals is not None}")
    logger.info(f"Vertex Colors provided: {vertex_colors is not None}")
    logger.info("--------------------------")
    # <<< --- End logging --- >>>

    try:
        results = _internal_generate_projections(
            vertices.flatten().tolist(),  # Flattened vertex coords
            triangle_indices.flatten().tolist(),  # Flattened triangle indices
            vertex_normals.flatten().tolist() if vertex_normals is not None else None,
            vertex_colors.flatten().tolist() if vertex_colors is not None else None,
            texture_np.flatten().tolist() if texture_np is not None else None,
            texture_np.shape[0] if texture_np is not None else 0,
            texture_np.shape[1] if texture_np is not None else 0,
            triangle_uvs.flatten().tolist() if triangle_uvs is not None else None,
            material_props["ka"],
            material_props["kd"],
            material_props["ks"],
            material_props["ns"],
            material_props["opacity"],
            material_props["illum"],
            precision,
            filtering,
            mode_str,
        )
    except Exception as e:
        # Log the specific error from the Rust backend if possible
        logger.error(f"Error calling Rust backend: {e}")
        raise RuntimeError(f"Rust backend failed during projection generation: {e}")

    # Process results (assuming Rust returns list of numpy arrays or similar)
    projection_images = []
    img_size = 1 << precision
    expected_shape = (img_size, img_size, 3)  # Assuming RGB output

    for i, res_bytes in enumerate(results):
        try:
            # Assuming res_bytes is bytes data for a numpy array
            img = np.frombuffer(res_bytes, dtype=np.uint8).reshape(expected_shape)
            projection_images.append(img)
        except Exception as e:
            logger.error(
                f"Error processing result {i}: {e}. Data length: {len(res_bytes)}, Expected shape: {expected_shape}"
            )
            # Handle error, maybe append a placeholder image or re-raise
            placeholder = np.zeros(expected_shape, dtype=np.uint8)
            projection_images.append(placeholder)

    if len(projection_images) != 6:
        logger.warning(f"Expected 6 projections, but received {len(projection_images)}")

    info(f"Generated {len(projection_images)} projections.")
    return projection_images


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
):
    """
    High-level function to load, preprocess, and generate projections for a mesh.

    Args:
        obj_location (str): Path to the OBJ mesh file.
        precision (int): Output image resolution exponent (2^precision).
        texture_file (str, optional): Path to the texture image file. Defaults to None.
        mtl_location (str, optional): Path to the MTL material file. Defaults to None.
        mode (ProjectionMode or str, optional): The processing mode ('obj_only', 'obj_texture', 'full').
                                                If None, it will be inferred based on provided files.
                                                Defaults to None.
        filtering (int, optional): Radius for outlier filtering. Defaults to 0 (disabled).
        flip_uv (bool): Whether to flip the V coordinate (1.0 - v) during preprocessing.
                        Defaults to True.
        verbose (bool): Enables detailed logging. Defaults to False.
        enable_post_processing (bool): Enables Open3D post-processing during mesh loading.
                                      Defaults to False.

    Returns:
        List[np.ndarray]: A list of 6 numpy arrays, each representing a projection image.
                          Returns None if any step fails.

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
            logger.warning(
                f"MTL file not found: {mtl_location}. Proceeding with default material properties."
            )
            # Don't raise error, _parse_mtl handles missing file with defaults

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

    # --- Preprocess Mesh ---
    # Errors (ValueError) will propagate up
    processed_data = preprocess_mesh(
        mesh,
        precision=precision,
        mode=resolved_mode,
        flip_v_coordinate=flip_uv,
        verbose=verbose,
    )

    # --- Generate Projections ---
    # Errors (ValueError, RuntimeError) will propagate up
    projection_images = generate_projections(
        processed_data=processed_data,
        precision=precision,
        mtl_location=mtl_location
        if resolved_mode == ProjectionMode.FULL
        else None,  # Only pass MTL for FULL mode
        mode=resolved_mode,
        filtering=filtering,
        verbose=verbose,
    )

    return projection_images
