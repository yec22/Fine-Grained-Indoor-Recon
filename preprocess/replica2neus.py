import argparse
import numpy as np
import os
from glob import glob
import cv2
from tqdm import tqdm
import open3d as o3d

def get_aabb(points, scale=1.0):
    '''
    Args:
        points; 1) numpy array (converted to '2)'; or 
                2) open3d cloud
    Return:
        min_bound
        max_bound
        center: bary center of geometry coordinates
    '''
    if isinstance(points, np.ndarray):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        points = point_cloud
    min_max_bounds = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(points)
    min_bound, max_bound = min_max_bounds.min_bound, min_max_bounds.max_bound
    center = (min_bound+max_bound)/2
    # center = points.get_center()
    if scale != 1.0:
        min_bound = center + scale * (min_bound-center)
        max_bound = center + scale * (max_bound-center)

    # logging.info(f"min_bound, max_bound, center: {min_bound, max_bound, center}")
    return min_bound, max_bound, center

def get_norm_matrix_from_point_cloud(pcd, radius_normalize_sphere = 1.0):
    '''Normalize point cloud into a sphere
    '''
    # if not checkExistence(path_cloud):
    #     logging.error(f"Path is not existent. [{path_cloud}]")
    #     exit()

    # pcd = read_point_cloud(path_cloud)
    min_bound, max_bound, pcd_center = get_aabb(pcd)

    edges_half =  np.linalg.norm(max_bound-min_bound, ord=2) / 2  #np.concatenate((max_bound -pcd_center, pcd_center -min_bound))
    max_edge_half = np.max(edges_half)
    scale = max_edge_half / radius_normalize_sphere
    scale_n2w = np.diag([scale, scale, scale, 1.0])
    translate_n2w = np.identity(4)
    translate_n2w[:3,3] = pcd_center

    trans_n2w = translate_n2w @ scale_n2w #@ rx_homo @ rx_homo_2

    return trans_n2w

def get_pose_inv(pose):
    # R = pose[:3, :3]
    # T = pose[:3, 3]
    # R_inv = R.transpose()
    # T_inv = -R_inv @ T
    # pose_inv = np.identity(4)
    # pose_inv[:3, :3] = R_inv
    # pose_inv[:3, 3] = T_inv
    return np.linalg.inv(pose)

def get_projection_matrix(root_dir, intrin, poses, trans_n2w):
    '''
    Args:
        poses: world to camera
    '''
    num_poses = poses.shape[0]
    
    projs = []
    poses_norm = []
    dir_pose_norm = os.path.join(root_dir, "extrin_norm")

    if not os.path.exists(dir_pose_norm):
        os.mkdir(dir_pose_norm)

    for i in range(num_poses):
        # pose_norm_i = poses[i] @ trans_n2w

        # Method 2
        pose = poses[i]
        rot = pose[:3,:3]
        trans = pose[:3,3]

        cam_origin_world = - np.linalg.inv(rot) @ trans.reshape(3,1)
        cam_origin_world_homo = np.concatenate([cam_origin_world,[[1]]], axis=0)
        cam_origin_norm = np.linalg.inv(trans_n2w) @ cam_origin_world_homo
        trans_norm = -rot @ cam_origin_norm[:3]

        pose[:3,3] = np.squeeze(trans_norm)
        poses_norm.append(pose)
        proj_norm = intrin @ pose
        projs.append(proj_norm)
        
        np.savetxt(f'{dir_pose_norm}/{i:04d}.txt', pose, fmt='%f') # world to camera
        np.savetxt(f'{dir_pose_norm}/{i:04d}_inv.txt', get_pose_inv(pose) , fmt='%f') # inv: camera to world
    return np.array(projs), np.array(poses_norm)

def get_camera_origins(poses_homo):
    '''
    Args:
        poses_homo: world to camera poses
    '''
    if not isinstance(poses_homo, np.ndarray):
        poses_homo = np.array(poses_homo)
    cam_centers = []
    poses_homo = np.array(poses_homo)
    num_cams = poses_homo.shape[0]
    for i in range(num_cams):
        rot = poses_homo[i, :3,:3]
        trans = poses_homo[i, :3,3]
        trans = trans.reshape(3,1)
        cam_center = - np.linalg.inv(rot) @ trans
        cam_centers.append(cam_center)
    cam_centers = np.array(cam_centers).squeeze(axis=-1)
    return cam_centers

def save_points(path_save, pts, colors = None, normals = None, BRG2RGB=False):
    '''save points to point cloud using open3d
    '''
    assert len(pts) > 0
    if colors is not None:
        assert colors.shape[1] == 3
    assert pts.shape[1] == 3
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        # Open3D assumes the color values are of float type and in range [0, 1]
        if np.max(colors) > 1:
            colors = colors / np.max(colors)
        if BRG2RGB:
            colors = np.stack([colors[:, 2], colors[:, 1], colors[:, 0]], axis=-1)
        cloud.colors = o3d.utility.Vector3dVector(colors) 
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals) 

    o3d.io.write_point_cloud(path_save, cloud)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='tmp/00')
    args = parser.parse_args()

    root_dir = args.root_dir

    skip = 4

    # pose
    traj_file = os.path.join(root_dir, "traj_w_c.txt")
    c2w = np.loadtxt(traj_file, delimiter=" ").reshape([-1, 4, 4])[::skip]
    
    pose_dir = os.path.join(args.root_dir, 'pose')
    if not os.path.exists(pose_dir):
        os.mkdir(pose_dir)
    
    for i in tqdm(range(c2w.shape[0])):
        np.savetxt(f'{pose_dir}/{i:04d}.txt', c2w[i], fmt='%f')

    # image crop
    image_dir = os.path.join(args.root_dir, 'image')
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    image_file = sorted(glob(os.path.join(root_dir, f'rgb/*png')))
    n_img = len(image_file)

    W_origin, H_origin = 1200, 680
    W_target, H_target = 896, 672

    img_idx = 0
    
    for idx in tqdm(range(n_img)):
        f = os.path.join(os.path.join(root_dir, f'rgb/rgb_{idx}.png'))
        img = cv2.imread(f)

        crop_width_half = (W_origin-W_target) // 2
        crop_height_half = (H_origin-H_target) // 2
        
        if idx == 0:
            intrinsic = np.loadtxt(os.path.join(args.root_dir, 'intrinsics.txt'))
            intrinsic[0, 2] = intrinsic[0, 2] - crop_width_half
            intrinsic[1, 2] = intrinsic[1, 2] - crop_height_half

            intrinsic[0, 0] = intrinsic[0, 0] / 1.4
            intrinsic[0, 2] = intrinsic[0, 2] / 1.4

            intrinsic[1, 1] = intrinsic[1, 1] / 1.4
            intrinsic[1, 2] = intrinsic[1, 2] / 1.4
            
            np.savetxt(os.path.join(args.root_dir, 'intrinsics_resize.txt'), intrinsic, fmt='%f')

        if idx % skip != 0:
            continue

        img_crop = img[crop_height_half:H_origin-crop_height_half, crop_width_half:W_origin-crop_width_half]
        img_crop = cv2.resize(img_crop, (640, 480))

        cv2.imwrite(f'{image_dir}/{img_idx:04d}.png', img_crop)
        img_idx += 1

    # generate neus data
    cloud_clean =o3d.io.read_point_cloud(os.path.join(root_dir, 'mesh.ply'))
    trans_n2w = get_norm_matrix_from_point_cloud(cloud_clean, radius_normalize_sphere=1.0)

    intrinsic_resize = np.loadtxt(os.path.join(args.root_dir, 'intrinsics_resize.txt'))

    w2c = np.stack([get_pose_inv(T) for T in c2w], axis=0)

    projs, poses_norm = get_projection_matrix(root_dir, intrinsic_resize, w2c, trans_n2w)
    path_trans_n2w = f'{root_dir}/trans_n2w.txt'
    np.savetxt(path_trans_n2w, trans_n2w, fmt = '%.04f')

    cloud_clean_trans = cloud_clean.transform(np.linalg.inv(trans_n2w))
    o3d.io.write_point_cloud(f'{root_dir}/point_cloud_scan_norm.ply', cloud_clean_trans)

    pts_cam_norm = get_camera_origins(poses_norm)
    save_points(f'{root_dir}/cam_norm.ply', pts_cam_norm)
    
    pts_cam = (trans_n2w[None, :3,:3] @ pts_cam_norm[:, :, None]).squeeze()  + trans_n2w[None, :3, 3]
    save_points(f'{root_dir}/cam_origin.ply', pts_cam)

    scale_mat = np.identity(4)
    num_cams = projs.shape[0]
    cams_neus = {}
    for i in range(num_cams):
        cams_neus[f"scale_mat_{i}"] = scale_mat
        cams_neus[f'world_mat_{i}'] = projs[i]
    
    np.savez(f'{root_dir}/cameras_sphere.npz', **cams_neus)