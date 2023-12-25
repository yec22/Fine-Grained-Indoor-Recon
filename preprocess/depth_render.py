import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import argparse, os
import cv2
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default='scene0050_00')
    parser.add_argument('--iter', type=str, default='50000')
    args = parser.parse_args()

    prefix = args.scene.split('_')[0]
    mesh_path = f'exps_backup/indoor/neus/{args.scene}/exp_{prefix}_full/meshes/000{args.iter}_reso512_{args.scene}_world.ply'
    intrinsic_path = f'dataset/indoor/intrinsic_depth.txt'
    pose_path = f'dataset/indoor/{args.scene}/pose'

    pose_list = sorted(os.listdir(pose_path))

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    for pose in tqdm(pose_list):

        intrinsic_matrix = np.loadtxt(intrinsic_path)[:3, :3]
        cam2world = np.loadtxt(os.path.join(pose_path, pose))
        world2cam = np.linalg.inv(cam2world)

        rays = scene.create_rays_pinhole(intrinsic_matrix=o3d.core.Tensor(intrinsic_matrix),
                                        extrinsic_matrix=o3d.core.Tensor(world2cam),
                                        width_px=640,
                                        height_px=480)

        # Compute the ray intersections.
        ans = scene.cast_rays(rays)
        depth = ans['t_hit'].numpy()

        # Visualize the hit distance (depth)
        # plt.imsave("depth.png", depth)
        np.save(f'tmp/pred_depth/{pose[:4]}.npy', depth)