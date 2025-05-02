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
                    --mode full --out projections_full/ --filtering 2

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
    parser.add_argument(
        "--mode",
        type=str,
        choices=[e.value for e in ProjectionMode],  # Use enum values for choices
        default=None,  # Default to None, will infer if not set
        help="Rendering mode: obj_only, obj_texture, or full. If not set, inferred from provided files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="projections",
        required=True,
        help="Directory to save projection images.",
    )
    parser.add_argument(
        "--precision", type=int, default=12, help="Precision P (image size 2^P x 2^P)."
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
    parser.add_argument(
        "--ignore-opacity",
        action="store_true",
        help="Ignore opacity from MTL file and force it to 1.0.",
    )
    args = parser.parse_args()

    # Configure logging
    if not args.verbose:
        logger.remove()

    # --- Generate Projections using the combined function ---
    try:
        # The project_mesh function now handles mode inference and validation internally
        # It now returns a tuple: (projection_images, occlusion_maps)
        projection_images, occlusion_maps = mp.project_mesh(
            obj_location=args.obj,
            precision=args.precision,
            texture_file=args.texture,
            mtl_location=args.mtl,
            mode=args.mode,  # Pass the mode arg directly (can be None)
            filtering=args.filtering,
            flip_uv=args.flip_uv,
            verbose=args.verbose,
            ignore_opacity=args.ignore_opacity,
            # enable_post_processing could be added as an arg if needed
        )
        if (
            projection_images is None or occlusion_maps is None
        ):  # Should not happen if exceptions are raised correctly
            logger.error("Projection generation failed, but no exception was raised.")
            return

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Failed to generate projections: {e}")
        return
    except Exception as e:  # Catch any unexpected errors
        logger.error(f"An unexpected error occurred: {e}")
        return

    # --- Save Output ---
    os.makedirs(args.out, exist_ok=True)
    view_names = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"]
    if len(projection_images) != len(view_names):
        logger.warning(
            f"Expected {len(view_names)} projections but got {len(projection_images)}. Saving available images."
        )
    # Check occlusion maps length as well if filtering was enabled
    save_occlusion = args.filtering > 0 and len(occlusion_maps) == len(view_names)
    if args.filtering > 0 and not save_occlusion:
        logger.warning(
            f"Filtering was enabled, but expected {len(view_names)} occlusion maps and got {len(occlusion_maps)}. Not saving occlusion maps."
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

        # Save occlusion map if available and filtering was enabled
        if save_occlusion:
            try:
                occ_map = occlusion_maps[i]
                # Scale 0/1 map to 0/255 for saving as visible grayscale image
                occ_map_scaled = (occ_map * 255).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(args.out, f"proj_{view_names[i]}_occ.png"),
                    occ_map_scaled,
                )
            except Exception as e:
                logger.error(f"Failed to save occlusion map {view_names[i]}: {e}")

    logger.info(f"Projections saved to {args.out}")
    if save_occlusion:
        logger.info(f"Occlusion maps saved to {args.out}")


if __name__ == "__main__":
    main()
