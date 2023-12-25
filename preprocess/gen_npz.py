import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse
from shutil import copyfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_neus', default='dataset/indoor/scene0616_00', type=str)
    args = parser.parse_args()

    pred_save_dir = os.path.join(f'{args.dir_neus}/gt_normal_npz/')
    if not os.path.exists(pred_save_dir):
        os.makedirs(pred_save_dir)

    for name in tqdm(sorted(os.listdir(f'{args.dir_neus}/gt_normal/'))):
        norm_data = (cv2.imread(os.path.join(f'{args.dir_neus}/gt_normal/', name)) / 255.0 - 0.5) * 2
        np.savez(f"{pred_save_dir}/{name[:4]}.npz", arr_0=norm_data[...,::-1])
        copyfile(os.path.join(f'{args.dir_neus}/gt_normal/', name), os.path.join(pred_save_dir, f'{name[:4]}.png'))
