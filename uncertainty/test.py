import argparse
import torch
from unet import UNet
from utils.data_loading import BasicDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np

def test(
        model,
        device,
        data_dir
):
    # 1. Create dataset
    dst_test = BasicDataset(data_dir)
    test_loader = DataLoader(dst_test, shuffle=False, batch_size=1, num_workers=1, pin_memory=True)

    dir_test = Path(f'{data_dir}/weight')

    for batch in tqdm(test_loader):
        image, normal, dino = batch['image'], batch['normal'], batch['dino']
                
        image = image.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
        normal = normal.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
        dino = dino.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)

        model_input = torch.cat([normal, image, dino], dim=1)
        model_output = model(model_input).permute(0, 2, 3, 1).squeeze(-1)

        Path(dir_test).mkdir(parents=True, exist_ok=True)
        im_name = batch['im_name'][0].split('/')[-1]
        im_name = im_name[:-4]
        plt.imsave(os.path.join(dir_test, f"{im_name}.png"), model_output[0].detach().cpu().numpy())
        np.save(os.path.join(dir_test, f"{im_name}.npy"), 1 - model_output[0].detach().cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the UNet on images and target')
    parser.add_argument('--in_channel', type=int, default=14, help='Number of epochs')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--data_dir', type=str, default="./dataset", help='data directory')

    args =  parser.parse_args()

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    model = UNet(n_channels=args.in_channel, n_classes=1, bilinear=args.bilinear)
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
    
    model = model.to(device=device)

    test(
            model=model,
            device=device,
            data_dir=args.data_dir
        )