import time

import open3d as o3d

import mesh_projector as mp

if __name__ == "__main__":
    # Configuration
    OBJ_PATH = "assets/just_a_girl_C0-L5_deq_tri.obj"
    PRECISION = 9
    FLIP_UV = True
    IGNORE_OPACITY = True
    VERBOSE = True
    POINT_CLOUD_DENSITY_FACTOR = 0.3
    # Read the mesh
    mesh = mp.read_mesh(
        obj_path=OBJ_PATH,
        mesh_mode="full",
        precision=PRECISION,
        flip_uv=FLIP_UV,
        ignore_opacity=IGNORE_OPACITY,
        verbose=VERBOSE,
    )
    t0 = time.time()
    # Project the mesh to a point cloud
    points, colors = mp.project_to_point_cloud_by_sampling(
        mesh,
        point_cloud_density_factor=POINT_CLOUD_DENSITY_FACTOR,
        verbose=VERBOSE,
    )
    t1 = time.time()
    print(f"Done. Time taken: {(t1 - t0):.2f} s")
    # Save the point cloud to a ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.io.write_point_cloud("output_point_cloud.ply", pcd)
