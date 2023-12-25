# Indoor Scene Reconstruction with Fine-Grained Details
We propose a new method for high quality reconstruction of indoor scenes using Hybrid Representation and Normal Prior Enhancement.

## Usage

#### Data preparation
The data is organized as follows:
```
<scene_name>
|-- cameras_sphere.npz   # camera parameters
|-- image
    |-- 0000.png        # target image for each view
    |-- 0001.png
    ...
|-- weight
    |-- 0000.png        # uncertainty weight for each view
    |-- 0001.png
    ...
|-- pose
    |-- 0000.txt        # camera pose for each view
    |-- 0001.txt
    ...
|-- pred_normal_refine
    |-- 0000.npz        # predicted normal for each view
    |-- 0001.npz
    ...
|-- xxx.ply		# GT mesh or point cloud from MVS
|-- trans_n2w.txt       # transformation matrix from normalized coordinates to world coordinates
```

### Setup
```
conda create -n recon python=3.8
conda activate recon
conda install pytorch=1.9.0 torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

### Training

```
python ./exp_runner.py --mode train --conf ./confs/scannet_0050.conf --gpu 0
```

### Mesh extraction
```
python exp_runner.py --mode validate_mesh --conf ./confs/scannet_0050.conf --is_continue --gpu 0
```

### Evaluation
```
python ./exp_evaluation.py --scene scene0050_00 --iter 50000 --reso 512
```
