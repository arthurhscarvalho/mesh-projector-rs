import argparse
import os

import cv2
import numpy as np
from loguru import logger

import mesh_projector as mp
from mesh_projector import ProjectionMode


def main():
    parser = argparse.ArgumentParser(
        description="""
        Generate 6 projections of a mesh using the mesh-projector library,
        supporting different rendering modes.

        Modes:
        - obj_only: Uses OBJ vertices and vertex colors (if available). Requires --obj.
        - obj_texture: Uses OBJ vertices/UVs and texture map. Requires --obj and --texture.
        - full: Uses OBJ vertices/UVs, texture map, and MTL properties. Requires --obj, --texture, and --mtl.

        Example (OBJ+Texture+MTL - Full):
        python examples/example_generate_projections.py \
                    --obj assets/spot/spot_triangulated.obj \
                    --texture assets/spot/spot_texture.png \
                    --mtl assets/spot/spot_triangulated.mtl \
                    --mode full --out projections_full/

        Example (OBJ+Texture):
        python examples/example_generate_projections.py \
                    --obj assets/spot/spot_triangulated.obj \
                    --texture assets/spot/spot_texture.png \
                    --mode obj_texture --out projections_tex/

        Example (OBJ Only - requires vertex colors in OBJ or will be white):
        python examples/example_generate_projections.py \
                    --obj assets/spot/spot_triangulated.obj \
                    --mode obj_only --out projections_obj/
        """,
        formatter_class=argparse.RawTextHelpFormatter,  # Keep formatting
    )
    parser.add_argument(
        "--obj", type=str, required=True, help="Path to the input OBJ file."
    )
    # Texture is optional depending on mode
    parser.add_argument(
        "--texture",
        type=str,
        required=False,
        default=None,
        help="Path to the input texture file (required for obj_texture and full modes).",
    )
    parser.add_argument(
        "--mtl",
        type=str,
        required=False,
        default=None,
        help="Path to the input MTL file (required for full mode).",
    )
    # Add mode argument
    parser.add_argument(
        "--mode",
        type=str,
        choices=[e.value for e in ProjectionMode],  # Use enum values for choices
        default=None,  # Default to None, will infer if not set
        help="Rendering mode: obj_only, obj_texture, or full. If not set, inferred from provided files.",
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Directory to save projection images."
    )
    parser.add_argument(
        "--precision", type=int, default=8, help="Precision P (image size 2^P x 2^P)."
    )
    parser.add_argument(
        "--filtering", type=int, default=0, help="Filtering window radius (0=disabled)."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    parser.add_argument(
        "--no-flip-uv",
        action="store_false",
        dest="flip_uv",
        help="Disable flipping the V texture coordinate (use if texture V=0 is bottom).",
    )
    parser.set_defaults(flip_uv=True)

    args = parser.parse_args()

    # Configure logging
    if not args.verbose:
        logger.remove()
        # logger.add(lambda _: None) # No need for null sink, just remove default

    # --- Determine Mode ---
    if args.mode:
        try:
            mode = ProjectionMode(args.mode)
        except ValueError:
            logger.error(
                f"Invalid mode specified: {args.mode}. Choices are: {[e.value for e in ProjectionMode]}"
            )
            return
    else:
        # Infer mode if not explicitly set
        if args.texture and args.mtl:
            mode = ProjectionMode.FULL
            logger.info("Inferred mode: FULL (OBJ, Texture, MTL provided)")
        elif args.texture:
            mode = ProjectionMode.OBJ_TEXTURE
            logger.info("Inferred mode: OBJ_TEXTURE (OBJ, Texture provided)")
        else:
            mode = ProjectionMode.OBJ_ONLY
            logger.info("Inferred mode: OBJ_ONLY (Only OBJ provided)")

    # --- Validate required files for chosen mode ---
    if mode in [ProjectionMode.OBJ_TEXTURE, ProjectionMode.FULL] and not args.texture:
        logger.error(f"Mode '{mode.value}' requires the --texture argument.")
        return
    if mode == ProjectionMode.FULL and not args.mtl:
        logger.error(f"Mode '{mode.value}' requires the --mtl argument.")
        return

    # --- Load Mesh ---
    try:
        mesh = mp.read_mesh(
            obj_location=args.obj,
            texture_file=args.texture,  # Pass None if not provided
            mtl_location=args.mtl,  # Pass None if not provided
            mode=mode,  # Pass the determined mode
            verbose=args.verbose,
        )
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Failed to load mesh: {e}")
        return

    # --- Preprocess Mesh ---
    try:
        processed_data = mp.preprocess_mesh(
            mesh,
            precision=args.precision,
            mode=mode,
            flip_v_coordinate=args.flip_uv,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error(f"Failed to preprocess mesh: {e}")
        return

    # --- Generate Projections ---
    try:
        # generate_projections now takes processed_data and mode
        # It returns only the images list
        projection_images = mp.generate_projections(
            processed_data=processed_data,
            precision=args.precision,
            mtl_location=args.mtl,  # Still needed for FULL mode's material parsing
            mode=mode,  # Pass the mode
            filtering=args.filtering,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error(f"Failed to generate projections: {e}")
        return

    # --- Save Output ---
    os.makedirs(args.out, exist_ok=True)
    view_names = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"]
    if len(projection_images) != len(view_names):
        logger.warning(
            f"Expected {len(view_names)} projections but got {len(projection_images)}. Saving available images."
        )

    for i, img in enumerate(projection_images):
        if i >= len(view_names):
            break  # Prevent index error if fewer images returned
        # Save image (convert RGB to BGR for OpenCV)
        try:
            cv2.imwrite(
                os.path.join(args.out, f"proj_{view_names[i]}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )
        except Exception as e:
            logger.error(f"Failed to save projection {view_names[i]}: {e}")

        # Occlusion maps are no longer returned or saved

    logger.info(f"Projections saved to {args.out}")


if __name__ == "__main__":
    main()
