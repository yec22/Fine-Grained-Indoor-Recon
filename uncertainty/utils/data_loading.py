from glob import glob
import os
from torch.utils.data import Dataset
import torch
import numpy as np
import cv2

class BasicDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = os.path.join(data_dir)
        self.images_lis = sorted(glob(os.path.join(self.data_dir, f'image/*png')))
        self.normal_lis = sorted(glob(os.path.join(self.data_dir, f'pred_normal_refine/*png')))
        self.dino_lis = sorted(glob(os.path.join(self.data_dir, f'dino_feature/*npy')))

        print('num: ', len(self.images_lis))

    def __len__(self):
        return len(self.images_lis)

    def read_img(self, path, reso):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[0], img.shape[1]

        if reso > 1.0:
            img = cv2.resize(img, (int(W/reso), int(H/reso)), interpolation=cv2.INTER_LINEAR)

        return img

    def __getitem__(self, idx):
        image_np = self.read_img(self.images_lis[idx], reso=1) / 255.
        image = torch.from_numpy(image_np.astype(np.float32))

        normal_np = self.read_img(self.normal_lis[idx], reso=1) / 255.
        normal = torch.from_numpy(normal_np.astype(np.float32))

        dino_np = np.load(self.dino_lis[idx])
        dino = torch.from_numpy(dino_np.astype(np.float32))

        return {
            'image': image,
            'normal': normal,
            'dino': dino,
            'im_name': self.images_lis[idx]
        }