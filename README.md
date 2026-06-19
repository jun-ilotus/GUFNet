# GUFNet: Geo-Aware Texture Adaptation and Uncertainty-Guided Fusion for 3D Semantic Segmentation in Complex Water Conservancy Scenarios

## Hydro2D3D Dataset

Hydro2D3D is a specialized benchmark dataset for quantitative evaluation in
multimodal joint segmentation tasks. It is a large-scale UAV 2D/3D semantic
segmentation benchmark for water conservancy scenes.

Baiduyun: https://pan.baidu.com/s/1KGKVbR70_uTK3On3aNwiZw, code: ctgk

After downloading and extracting the dataset, arrange the Hydro2D3D root as
follows this root directory:

```text
Hydro2D3D/
|-- 00/
|   |-- velodyne/
|   |   |-- 000000.pth
|   |   `-- ...
|   `-- image/
|       |-- 000000.png
|       `-- ...
|-- 01/
|   |-- velodyne/
|   `-- image/
|-- 02/
|   |-- velodyne/
|   `-- image/
|-- 03/
|   |-- velodyne/
|   `-- image/
|-- 04/
|   |-- velodyne/
|   `-- image/
`-- 05/
    |-- velodyne/
    `-- image/
```

Each point-cloud file in `velodyne/` should be a `.pth` sample whose columns are
`x, y, z, r, g, b, label`. The paired image must have the same basename and be
stored in the sibling `image/` folder, for example
`00/velodyne/000000.pth` and `00/image/000000.png`.

## Environment

```bash
conda create -n gufnet python=3.8 -y
conda activate gufnet

pip install torch==2.0.1 torchvision==0.15.2 -f https://mirrors.aliyun.com/pytorch-wheels/cu118
pip install easydict==1.13 pyquaternion==0.9.9 torchmetrics==0.5 pytorch-lightning==1.3.8 wandb scipy Pillow einops
pip install spconv-cu118==2.3.4
```

Install `torch_scatter` from the PyG wheel index matching your PyTorch/CUDA
version:

```bash
pip install torch_scatter-2.1.1+pt20cu118-cp38-cp38-linux_x86_64.whl
```

Optional: install `mamba_ssm` from the
[state-spaces/mamba releases](https://github.com/state-spaces/mamba/releases).
If it is not installed, GAUFNet falls back to the built-in GRU sequence block.

```bash
pip install mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

## Quick Start

Train on Hydro2D3D:

```bash
python main.py --gpu 0 --log_dir gufnet --config_path config/hydro2d3d.yaml
```

Test on Hydro2D3D:

```bash
python main.py --gpu 0 --test --num_vote 12 --config_path config/hydro2d3d.yaml --checkpoint <path>
```
