import torch
import numpy as np
import cv2
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

# copy from vis-mvsnet
def find_files(dir, exts=['*.png', '*.jpg']):
    if os.path.isdir(dir):
        # types should be ['*.png', '*.jpg']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []
            
# adatpted from https://github.com/dakshaau/ICP/blob/master/icp.py#L4 for rotation only 
def best_fit_transform(A, B):
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    AA = A
    BB = B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    return R


# align normal map in the x direction from left to right
def align_normal_x(normal1, normal2, s1, e1, s2, e2):
    assert normal1.shape[0] == normal2.shape[0]
    assert normal1.shape[1] == normal2.shape[1]

    assert (e1 - s1) == (e2 - s2)
    
    R = best_fit_transform(normal2[:, :, s2:e2].reshape(3, -1).T, normal1[:, :, s1:e1].reshape(3, -1).T)

    normal2_aligned = (R @ normal2.reshape(3, -1)).reshape(normal2.shape)
    result = np.ones((3, normal1.shape[1], normal1.shape[2] + normal2.shape[2] - (e1 - s1)))

    result[:, :, :s1] = normal1[:, :, :s1]
    result[:, :, normal1.shape[2]:] = normal2_aligned[:, :, e2:]

    weight = np.linspace(1, 0, (e1-s1))[None, None, :]
    
    result[:, :, s1:normal1.shape[2]] = normal1[:, :, s1:] * weight + normal2_aligned[:, :, :e2] * (1 - weight)
    result = result / (np.linalg.norm(result, axis=0) + 1e-15)[None]
    
    return result

# align normal map in the y direction from top to down
def align_normal_y(normal1, normal2, s1, e1, s2, e2):
    assert normal1.shape[0] == normal2.shape[0]
    assert normal1.shape[2] == normal2.shape[2]

    assert (e1 - s1) == (e2 - s2)
    
    R = best_fit_transform(normal2[:, s2:e2, :].reshape(3, -1).T, normal1[:, s1:e1, :].reshape(3, -1).T)

    normal2_aligned = (R @ normal2.reshape(3, -1)).reshape(normal2.shape)
    result = np.ones((3, normal1.shape[1] + normal2.shape[1] - (e1 - s1), normal1.shape[2]))

    result[:, :s1, :] = normal1[:, :s1, :]
    result[:, normal1.shape[1]:, :] = normal2_aligned[:, e2:, :]

    weight = np.linspace(1, 0, (e1-s1))[None, :, None]
    
    result[:, s1:normal1.shape[1], :] = normal1[:, s1:, :] * weight + normal2_aligned[:, :e2, :] * (1 - weight)
    result = result / (np.linalg.norm(result, axis=0) + 1e-15)[None]
    
    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate high resolution outputs')

    parser.add_argument('--mode', required=True, help="choose from creating patches or merge patches")
    args = parser.parse_args()

    assert args.mode in ["create_patches", "merge_patches"]

    # data-folder
    data_root = 'dataset/indoor'
    scenes = ['scene0050_00']

    # temporary folders
    out_path_prefix = "dataset/indoor/highres_tmp"

    # output folder for hihg-resolution cues 
    out_path_for_training = 'dataset/indoor/highres/'

    for scene in scenes:
        # temporary folders for overlapped images
        out_path = os.path.join(out_path_prefix, scene)
        os.makedirs(out_path, exist_ok=True)
        print(out_path)

        out_folder = os.path.join(out_path, "image")
        os.makedirs(out_folder, exist_ok=True)
        print(out_folder)

        # high-resolutin image used for training
        out_path_for_training = os.path.join(out_path_for_training, scene)
        os.makedirs(out_path_for_training, exist_ok=True)

        # load images
        images_dir = os.path.join(data_root, scene, "image_denoised_cv07211010")
        print(images_dir)
        rgbs = find_files(images_dir, exts=["*.png"])
        
        # only use 3 images for debug
        # rgbs = rgbs[:3]

        if args.mode == 'create_patches':
            for idx, image in enumerate(rgbs):
                idx = idx * 10
                image = cv2.imread(image)
            
                size = 384
                
                H, W = image.shape[:2]
                assert H == 480
                assert W == 640

                x = (W - size) // 64
                y = (H - size) // 48
                
                # crop images
                for j in range(0, y + 1):
                    for i in range(0, x + 1):
                        image_cur = image[j*48:j*48+size, i*64:i*64+size, :]
                        print(image_cur.shape)
                        target_file = os.path.join(out_folder, "%06d_%02d_%02d.jpg"%(idx, j, i))
                        print(target_file)
                        cv2.imwrite(target_file, image_cur)
                
                # save middle file for alignments
                image_cur = image[H//2-size//2:H//2+size//2, W//2-size//2:W//2+size//2]

                print(image_cur.shape)
                target_file = os.path.join(out_folder, "%06d_mid.jpg"%(idx))
                print(target_file)
                cv2.imwrite(target_file, image_cur)

        elif args.mode == 'merge_patches':    
            H, W = 480, 640
            size = 384
            x = (W - size) // 64
            y = (H - size) // 48
    
            for idx, image in enumerate(rgbs):
                idx = idx * 10
                # normal
                normals_row = []
                # align normal maps from left to right row by row  
                for j in range(0, y + 1):            
                    normals = []
                    for i in range(0, x + 1):
                        normal_path = os.path.join(out_path, "normal", "%06d_%02d_%02d_normal.png"%(idx, j, i))
                        normal = cv2.imread(normal_path)
                        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
                        normal = normal / 255.
                        normal = normal * 2. - 1.
                        normal = normal.transpose((2, 0, 1))
                        normal = normal / (np.linalg.norm(normal, axis=0) + 1e-15)[None]
                        normals.append(normal)
            
                    # align from left to right
                    normal_left = normals[0]
                    s1 = 64
                    s2 = 0
                    e2 = 320
                    for normal_right in normals[1:]:
                        normal_left = align_normal_x(normal_left, normal_right, s1, normal_left.shape[2], s2, e2)
                        s1 += 64
                    normals_row.append(normal_left)
                    

                normal_top = normals_row[0]
                # align normal maps from top to down
                s1 = 48
                s2 = 0
                e2 = 336
                for normal_bottom in normals_row[1:]:
                    print(normal_top.shape, normal_bottom.shape)
                    normal_top = align_normal_y(normal_top, normal_bottom, s1, normal_top.shape[1], s2, e2)
                    s1 += 48

                # align to middle part
                mid_file = os.path.join(out_path, "normal", "%06d_mid_normal.png"%(idx))
                mid_normal = cv2.imread(mid_file)
                mid_normal = cv2.cvtColor(mid_normal, cv2.COLOR_BGR2RGB)
                mid_normal = mid_normal / 255.
                mid_normal = mid_normal * 2. - 1.
                mid_normal = mid_normal.transpose((2, 0, 1))
                mid_normal = mid_normal / (np.linalg.norm(mid_normal, axis=0) + 1e-15)[None]
                
                R = best_fit_transform(normal_top[:, H//2-size//2:H//2+size//2, W//2-size//2:W//2+size//2].reshape(3, -1).T, mid_normal.reshape(3, -1).T)
                normal_top = (R @ normal_top.reshape(3, -1)).reshape(normal_top.shape)

                plt.imsave(os.path.join(out_path_for_training ,"%04d.png"%(idx)), np.moveaxis(normal_top, [0,1, 2], [2, 0, 1]) * 0.5 + 0.5)
                normal_top = normal_top.transpose((1, 2, 0)) # [H, W, 3]
                np.save(os.path.join(out_path_for_training ,"%04d.npy"%(idx)), normal_top)