import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from data.dataloader_custom import CustomLoader
from models.NNET import NNET
import utils.utils as utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test(model, test_loader, device, results_dir):
    alpha_max = 60
    kappa_max = 30

    with torch.no_grad():
        for data_dict in tqdm(test_loader):

            img = data_dict['img'].to(device)
            norm_out_list, _, _ = model(img)
            norm_out = norm_out_list[-1]

            pred_norm = norm_out[:, :3, :, :]
            pred_kappa = norm_out[:, 3:, :, :]

            # to numpy arrays
            img = img.detach().cpu().permute(0, 2, 3, 1).numpy()                    # (B, H, W, 3)
            pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()        # (B, H, W, 3)
            pred_kappa = pred_kappa.cpu().permute(0, 2, 3, 1).numpy()

            # save results
            img_name = data_dict['img_name'][0]

            # 1. save input image
            img = utils.unnormalize(img[0, ...])

            target_path = '%s/%s_img.png' % (results_dir, img_name)
            # plt.imsave(target_path, img)

            # 2. predicted normal
            pred_norm_rgb = ((pred_norm + 1) * 0.5) * 255
            pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
            pred_norm_rgb = pred_norm_rgb.astype(np.uint8)                  # (B, H, W, 3)

            target_path = '%s/%s_norm.png' % (results_dir, img_name)
            plt.imsave(target_path, pred_norm_rgb[0, :, :, :])

            # 3. predicted uncertainty
            # pred_alpha = utils.kappa_to_alpha(pred_kappa)
            # uncertainty = pred_alpha[0, :, :, 0]
            # target_path = '%s/%s_uncertainty.png' % (results_dir, img_name)

            # plt.imsave(target_path, uncertainty, vmin=0.0, vmax=alpha_max, cmap='jet')
            
            # 4. normal weight
            # u_max = uncertainty.max()
            # u_min = uncertainty.min()

            # weight = (uncertainty - u_min) / (u_max - u_min)
            # plt.imsave(target_path, weight)

            # target_path = '%s/%s_weight.npy' % (results_dir, img_name)
            # np.save(target_path, 1 - weight)



if __name__ == '__main__':
    # Arguments ########################################################################################################
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.convert_arg_line_to_args = utils.convert_arg_line_to_args

    parser.add_argument('--architecture', required=True, type=str, help='{BN, GN}')
    parser.add_argument("--pretrained", required=True, type=str, help="{nyu, scannet}")
    parser.add_argument('--sampling_ratio', type=float, default=0.4)
    parser.add_argument('--importance_ratio', type=float, default=0.7)
    parser.add_argument('--input_height', default=480, type=int)
    parser.add_argument('--input_width', default=640, type=int)
    parser.add_argument('--imgs_dir', default='./examples', type=str)

    # read arguments from txt file
    if sys.argv.__len__() == 2 and '.txt' in sys.argv[1]:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    device = torch.device('cuda:7')

    # load checkpoint
    checkpoint = './checkpoints/%s.pt' % args.pretrained
    print('loading checkpoint... {}'.format(checkpoint))
    model = NNET(args).to(device)
    model = utils.load_checkpoint(checkpoint, model)
    model.eval()
    print('loading checkpoint... / done')

    # test the model
    results_dir = os.path.join(os.path.abspath(os.path.join(args.imgs_dir, "..")), 'pred_normal_png')
    print(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    test_loader = CustomLoader(args, args.imgs_dir).data
    test(model, test_loader, device, results_dir)
