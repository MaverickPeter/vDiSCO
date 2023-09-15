# Leveraging BEV Representation for 360-degree Visual Place Recognition

[Paper: Leveraging BEV Representation for 360-degree Visual Place Recognition ](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/datasets)

Author: Peter XU (Xuecheng XU)

Affiliation: ZJU-Robotics Lab, Zhejiang University

Maintainer: Peter XU, xuechengxu@zju.edu.cn

## About

We investigates the advantages of using Bird's Eye View (BEV) representation in <b>360-degree visual place recognition (VPR)</b>. We propose a novel network architecture that utilizes the BEV representation in feature extraction, feature aggregation, and vision-LiDAR fusion, which bridges visual cues and spatial awareness. The proposed BEV-based method is evaluated in ablation and comparative studies on two datasets, including on-the-road and off-the-road scenarios. The experimental results verify the hypothesis that BEV can benefit VPR by its superior performance compared to baseline methods.

## Prerequisites

### CUDA 11.x and Pytorch 1.1x.x
```
# cuda 11.3.1
wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run
bash cuda_11.3.1_465.19.01_linux.run

# torch 1.12.0
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Conda env

```
# create conda env and install required packages
conda create --name vdisco python==3.8
pip install -r requirements.txt

# mmcv
pip install openmim
mim install mmcv-full

# spconv related to cuda version
pip install spconv-cu113

# install ops
cd prnet/ops
python setup.py develop

# install prnet
cd ../../
python setup.py develop

```

## Preprocess
Get dataset from [NCLT](http://robots.engin.umich.edu/nclt/) and [Oxford](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/datasets)


```
# Prepare data, check detailed params in files
cd prnet/datasets/nclt
python image_preprocess.py --dataset_root $YOUR_DATASET_FOLDER

# get training pickle
python generate_training_tuples.py --dataset_root $YOUR_DATASET_FOLDER

# get evaluation pickle
python generate_evaluation_sets.py --dataset_root $YOUR_DATASET_FOLDER
```

## Training
```
# define python path
export PYTHONPATH=`pwd`:$PYTHONPATH

# train with method's config
cd tools
python train.py --config ../prnet/config/deformable/config_deformable.txt --model_config ../prnet/config/deformable/deformable.txt
# !! you need to change the params in the config files
```

## Evaluation
```
# eval
cd tools
python evaluate.py --dataset_type nclt --eval_set test_xxx.pickle --model_config ../prnet/config/deformable/deformable.txt --weights ../weights/xxx.pth
```


## Our other projects ###
* DiSCO: Differentiable Scan Context with Orientation (RA-L 2021): [DiSCO](https://github.com/MaverickPeter/DiSCO-pytorch) 
* RING++: Roto-translation Invariant Gram for Global Localization on a Sparse Scan Map (TRO 2023): [MR_SLAM](https://github.com/MaverickPeter/MR_SLAM)


## Citation
```bibtex
@misc{xu2023leveraging,
      title={Leveraging BEV Representation for 360-degree Visual Place Recognition}, 
      author={Xuecheng Xu and Yanmei Jiao and Sha Lu and Xiaqing Ding and Rong Xiong and Yue Wang},
      year={2023},
      eprint={2305.13814},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgments

[DOLG-pytorch](https://github.com/dongkyuk/DOLG-pytorch): PyTorch Implementation of "DOLG: Single-Stage Image Retrieval with Deep Orthogonal Fusion of Local and Global Features"

[NetVLAD-pytorch](https://github.com/Nanne/pytorch-NetVlad): Implementation of NetVlad in PyTorch, including code for training the model on the Pittsburgh dataset.

[ASMK & HowNet](https://github.com/jenicek/asmk): A Python implementation of the ASMK approach.

[Simple-BEV](https://github.com/aharley/simple_bev): A Simple Baseline for BEV Perception.

## License
Our code is released under the MIT License (see LICENSE file for details).


