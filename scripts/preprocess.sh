#!/bin/bash
scan="scene0084_00"

# image sharpening
python preprocess/img_preprocess.py --scene ${scan}

# pred normal
cd snucode
python test2.py --pretrained scannet_neuris_retrain --architecture BN --imgs_dir ../dataset/indoor/${scan}/image_process
python gen_npz.py --dir_neus ../dataset/indoor/${scan}

# dino feature
cd ..
python dino/extract_dino_feature.py --data_dir ./dataset/indoor/${scan}

# uncertainty weight
cd uncertainty
python test.py --load checkpoints/ckpt_epoch_5.pth --data_dir ../dataset/indoor/${scan}