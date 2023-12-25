import os, argparse, logging
from datetime import datetime
import numpy as np

import evaluation.EvalScanNet as EvalScanNet
from evaluation.renderer import render_depthmaps_pyrender

import utils.utils_geometry as GeoUtils
import utils.utils_image  as ImageUtils
import utils.utils_io as IOUtils
import utils.utils_normal as NormalUtils


if __name__ == '__main__':
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='eval_3D_mesh_metrics')
    parser.add_argument('--scene', type=str, default='scene0050_00')
    parser.add_argument('--iter', type=int, default=50000)
    parser.add_argument('--reso', type=int, default=512)

    args = parser.parse_args()
    
    if args.mode == 'eval_3D_mesh_metrics':
        dir_dataset = './dataset/indoor'
        path_intrin = f'{dir_dataset}/intrinsic_depth.txt'
        name_baseline = 'neus' # manhattansdf neuris
        exp_name = name_baseline
        eval_threshold = 0.05
        check_existence = True
        
        dir_results_baseline = f'./exps/indoor'

        metrics_eval_all = []
        scene_name = args.scene
        logging.info(f'\n\nProcess: {scene_name}')

        prefix = scene_name.split('_')[0]
        path_mesh_pred = f'exps/indoor/neus/{scene_name}/exp_{prefix}_full/meshes/000{args.iter}_reso{args.reso}_{scene_name}_world.ply'
        metrics_eval =  EvalScanNet.evaluate_3D_mesh(path_mesh_pred, scene_name, dir_dataset = './dataset/indoor',
                                                                eval_threshold = 0.05, reso_level = 2, 
                                                                check_existence = check_existence)
    
        metrics_eval_all.append(metrics_eval)
        metrics_eval_all = np.array(metrics_eval_all)
        str_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
        path_log = f'{dir_results_baseline}/eval_{name_baseline}_thres{eval_threshold}_{str_date}.txt'
        EvalScanNet.save_evaluation_results_to_latex(path_log, 
                                                        header = f'{exp_name}\n                     Accu.      Comp.      Prec.     Recall     F-score \n', 
                                                        results = metrics_eval_all, 
                                                        names_item = args.scene, 
                                                        save_mean = True, 
                                                        mode = 'w')
    
    print('Done')