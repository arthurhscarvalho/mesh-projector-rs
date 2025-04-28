import argparse
import os
import time

import cv2
import numpy as np
from loguru import logger

import mesh_projector as mp


def save_projections(projections, output_dir="./projections"):
    """Helper function to save projection images."""
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(projections)):
        image = projections[i].astype(np.uint8)
        # Convert RGB (from library) to BGR (for OpenCV)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"projection_{i}.png"), image_bgr)
    print(f"Projections saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="""
        Generate 6 projections of a textured mesh using the mesh-projector library.
        Example: python examples/example_generate_projections.py \
                    --obj assets/just_a_girl_C0-L5_deq_tri.obj \
                    --texture assets/just_a_girl_C0-L5_deq_tri_0.png \
                    --mtl assets/just_a_girl_C0-L5_deq_tri.mtl \
                    --out projections/ \
                    --precision 10
        """
    )
    parser.add_argument(
        "--obj", type=str, required=True, help="Path to the input OBJ file."
    )
    parser.add_argument(
        "--texture", type=str, required=True, help="Path to the input texture file."
    )
    parser.add_argument(
        "--mtl",
        type=str,
        required=False,
        default=None,
        help="Path to the input MTL file (optional).",
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

    args = parser.parse_args()

    # Configure logging based on verbosity
    if not args.verbose:
        logger.remove()  # Remove default handler
        logger.add(lambda _: None)  # Add a null sink if not verbose

    # --- Load Mesh ---
    try:
        mesh = mp.read_mesh(
            args.obj,
            args.texture,
            mtl_location=args.mtl,
            verbose=args.verbose,  # Pass mtl path
        )
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Failed to load mesh: {e}")
        return

    # --- Generate Projections ---
    try:
        images, ocp_maps = mp.generate_projections(
            mesh,
            args.precision,
            mtl_location=args.mtl,
            filtering=args.filtering,
            verbose=args.verbose,  # Pass mtl path
        )
    except Exception as e:
        logger.error(f"Failed to generate projections: {e}")
        return

    # --- Save Output ---
    os.makedirs(args.out, exist_ok=True)
    view_names = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"]
    for i, (img, ocp) in enumerate(zip(images, ocp_maps)):
        # Save image (convert RGB to BGR for OpenCV)
        cv2.imwrite(
            os.path.join(args.out, f"proj_{view_names[i]}.png"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )
        # Save occlusion map (scale to 0-255 for visibility)
        cv2.imwrite(
            os.path.join(args.out, f"ocp_{view_names[i]}.png"),
            (ocp * 255).astype(np.uint8),
        )

    logger.info(f"Projections saved to {args.out}")


if __name__ == "__main__":
    main()
