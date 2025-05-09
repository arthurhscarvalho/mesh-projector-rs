import argparse
import os

import cv2

import mesh_projector as mp


def main():
    parser = argparse.ArgumentParser(
        description="""
        Generate 6 projections of a mesh using the mesh-projector library.
        The rendering mode is automatically inferred based on the files provided:
        
        - OBJ only: Uses OBJ vertices and vertex colors (if available). Requires --obj.
        - OBJ + Texture: Uses OBJ vertices/UVs and texture map. Requires --obj and --texture.
        - OBJ + Texture + MTL: Uses OBJ vertices/UVs, texture map, and MTL properties. Requires --obj, --texture, and --mtl.

        Example (OBJ+Texture+MTL):
        python examples/example_generate_projections.py \
                    --obj assets/spot/spot_triangulated.obj \
                    --texture assets/spot/spot_texture.png \
                    --mtl assets/spot/spot_triangulated.mtl \
                    --out projections_full/ --filtering 2

        Example (OBJ+Texture):
        python examples/example_generate_projections.py \
                    --obj assets/spot/spot_triangulated.obj \
                    --texture assets/spot/spot_texture.png \
                    --out projections_tex/

        Example (OBJ Only - requires vertex colors in OBJ or will be white):
        python examples/example_generate_projections.py \
                    --obj assets/spot/spot_triangulated.obj \
                    --out projections_obj/
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
        help="Path to the input texture file (for textured renderings).",
    )
    parser.add_argument(
        "--mtl",
        type=str,
        required=False,
        default=None,
        help="Path to the input MTL file (for material properties).",
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
    # --- Generate Projections using the combined function ---
    try:
        # Mode is automatically inferred based on the provided files
        projection_array, depth_array, normal_array, occlusion_array = mp.project_mesh(
            obj_location=args.obj,
            texture_file=args.texture,
            mtl_location=args.mtl,
            precision=args.precision,
            filtering=args.filtering,
            flip_uv=args.flip_uv,
            ignore_opacity=args.ignore_opacity,
            verbose=args.verbose,
        )
        if (
            projection_array is None
            or depth_array is None
            or normal_array is None
            or occlusion_array is None
        ):  # Should not happen if exceptions are raised correctly
            print("Projection generation failed, but no exception was raised.")
            return
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Failed to generate projections: {e}")
        return
    except Exception as e:  # Catch any unexpected errors
        print(f"An unexpected error occurred: {e}")
        return
    # --- Save Output ---
    os.makedirs(args.out, exist_ok=True)
    view_names = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"]
    images = zip(projection_array, occlusion_array, depth_array, normal_array)
    for i, (image, occlusion, depth, normal) in enumerate(images):
        # Save image (convert RGB to BGR for OpenCV)
        try:
            cv2.imwrite(
                os.path.join(args.out, f"proj_{view_names[i]}.png"),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(args.out, f"proj_{view_names[i]}_depth.png"),
                cv2.cvtColor(depth, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(args.out, f"proj_{view_names[i]}_normal.png"),
                cv2.cvtColor(normal, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(args.out, f"proj_{view_names[i]}_occ.png"),
                cv2.cvtColor(occlusion, cv2.COLOR_GRAY2BGR),
            )
        except Exception as e:
            print(f"Failed to save projection {view_names[i]}: {e}")
    print(f"Projections saved to {args.out}")


if __name__ == "__main__":
    main()
