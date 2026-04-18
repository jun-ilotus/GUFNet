# GUFNet: Geo-Aware Texture Adaptation and Uncertainty-Guided Fusion for 3D Semantic Segmentation in Complex Water Conservancy Scenarios


## Hydro2D3D Dataset

Hydro2D3D is a specialized benchmark dataset for quantitative evaluation in multi-modal joint segmentation tasks, which is  a large-scale UAV 2D–3D semantic segmentation benchmark for water conservancy scenes.

OneDrive: TODO

Baiduyun: https://pan.baidu.com/s/17Bs7ZtVQuZ4v_xBa0iHXAg

## Environment
```bash
conda create -n gaufnet python=3.8 -y

conda activate gaufnet

pip install torch==2.0.1 torchvision==0.15.2  -f https://mirrors.aliyun.com/pytorch-wheels/cu118

pip install easydict==1.13 pyquaternion==0.9.9 torchmetrics==0.5 pytorch-lightning==1.3.8 wandb scipy

pip install spconv-cu118==2.3.4


torch_scatter：[pytorch-geometric.com/whl/](https://pytorch-geometric.com/whl/)，

pip install torch_scatter-2.1.1+pt20cu118-cp38-cp38-linux_x86_64.whl


[Releases · state-spaces/mamba](https://github.com/state-spaces/mamba/releases) 

pip install mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```


## Quick Start

Train on Hydro2D3D

```bash
python main.py --gpu0 --log_dir gufnet --config_path config/hydro2d3d.yaml

```

Test on Hydro2D3D

```bash
python main.py --gpu0 --test --num_vote 12 --config_path config/hydro2d3d.yaml --checkpoint <path>
```