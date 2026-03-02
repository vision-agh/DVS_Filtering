# Event-Based Classification with Noise Filtering

This repository is intended for experimenting with and comparing different deep learning models for **event-based classification**, both **with** and **without noise filtering**.

The framework supports multiple model families and their corresponding event representations, allowing consistent comparison across different learning paradigms.

## Supported Models

The repository currently includes the following model families:

- **CNN** with **ResNet**
- **Vision Transformer** with **MaxViT**
- **Graph Neural Network** with **SplineConv**
- **Spiking Neural Network** with **SpikingJelly-based ResNet**

## Data Representations

Each model operates on a dedicated representation of event-based data:

- **CNN (ResNet)** → **event frame** representation
- **ViT (MaxViT)** → **event voxel** representation (spatiotemporal)
- **GNN (SplineConv)** → **event graph** representation
- **SNN (SpikingJelly ResNet)** → **spiking-compatible event** representation

## Dependencies

To create the environment and install the required packages:

```bash
conda create -y -n dvs_fil python=3.9
conda activate dvs_fil

conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install h5py
conda install blosc-hdf5-plugin lightning -c conda-forge

pip install matplotlib tqdm numba scikit-learn wandb pyyaml opencv-python pybind11 omegaconf
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install lightning
