import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default='scene0616_00')
    args = parser.parse_args()

    data_dir = f'dataset/indoor/{args.scene}/image'
    save_dir = f'dataset/indoor/{args.scene}/image_process'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    file_list = sorted(os.listdir(data_dir))

    for f in tqdm(file_list):
        f_path = os.path.join(data_dir, f)
        image = cv2.imread(f_path)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen = cv2.filter2D(image, -1, sharpen_kernel)
        smooth = cv2.medianBlur(sharpen, 3)
        
        cv2.imwrite(os.path.join(save_dir, f), smooth)