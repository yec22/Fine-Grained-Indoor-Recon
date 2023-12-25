import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse
from shutil import copyfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_neus', default='dataset/indoor/demo', type=str)
    args = parser.parse_args()

    pred_save_dir = os.path.join(f'{args.dir_neus}/pred_normal_refine/')
    if not os.path.exists(pred_save_dir):
        os.makedirs(pred_save_dir)
    
    # weight_save_dir = os.path.join(f'{args.dir_neus}/weight/')
    # if not os.path.exists(weight_save_dir):
    #     os.makedirs(weight_save_dir)

    for name in tqdm(sorted(os.listdir(f'{args.dir_neus}/pred_normal_png/'))):
        if name[-8:-4] == "norm":
            norm_data = (cv2.imread(os.path.join(f'{args.dir_neus}/pred_normal_png/', name)) / 255.0 - 0.5) * 2
            np.savez(f"{pred_save_dir}/{name[:4]}.npz", arr_0=norm_data[...,::-1])
            copyfile(os.path.join(f'{args.dir_neus}/pred_normal_png/', name), os.path.join(pred_save_dir, f'{name[:4]}.png'))
        # if name[-10:-4] =="weight":
        #     copyfile(os.path.join(f'{args.dir_neus}/pred_normal_png/', name), os.path.join(weight_save_dir, f'{name[:4]}.npy'))