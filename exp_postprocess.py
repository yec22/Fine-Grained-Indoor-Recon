import os, argparse, logging
from datetime import datetime
import numpy as np

import evaluation.EvalScanNet as EvalScanNet
from evaluation.renderer import render_depthmaps_pyrender

import utils.utils_geometry as GeoUtils
import utils.utils_image  as ImageUtils
import utils.utils_io as IOUtils
import utils.utils_normal as NormalUtils

# from confs.path import lis_name_scenes

if __name__ == '__main__':
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    dir_dataset = './dataset/indoor'
    path_intrin = f'{dir_dataset}/demo/intrinsics.txt'
    name_baseline = 'neus' # manhattansdf neuris
    exp_name = name_baseline
    eval_threshold = 0.05
    check_existence = True
    
    dir_results_baseline = f'./exps/indoor'

    lis_name_scenes = ["demo"]
    for scene_name in lis_name_scenes:
        logging.info(f'\n\nProcess: {scene_name}')

        path_mesh_pred = f'exps/indoor/neus/{scene_name}/exp_{scene_name}/meshes/00100000_reso1024_{scene_name}_world.ply'
        metrics_eval =  EvalScanNet.postprocess(path_mesh_pred, scene_name, dir_dataset = './dataset/indoor',
                                                            eval_threshold = 0.05, reso_level = 2, 
                                                            check_existence = check_existence)
    
    print('Done')
