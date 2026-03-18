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
```

## Noise Generator

A C++ shot-noise generator with Python bindings. It injects random noise events into a clean DVS event stream, simulating real sensor noise.

### Build

```bash
cd noise_generator
mkdir -p build && cd build
cmake ..
make
```

This produces `noise_generator_py.cpython-*.so` inside `build/`.

### Python usage

```python
import sys
sys.path.insert(0, "noise_generator/build")
import noise_generator_py as ng

# Create input events: (x, y, polarity, timestamp_us)
events = [ng.Event2d(x=i % 640, y=i % 480, p=i % 2, t=i * 10) for i in range(1000)]

# Create noise generator
noise_gen = ng.NoiseGeneratorAlgorithm(
    width=640,
    height=480,
    shot_noise_rate_hz=0.5,   # global shot-noise rate in Hz
    poisson_divider=20.0,     # controls noise density
    timestamp_resolution_us=1
)

# Add noise to the event stream
noisy_events = noise_gen.process_events(events)
print(f"Input: {len(events)} events → Output: {len(noisy_events)} events")
```

`shot_noise_rate_hz=0` disables noise injection and passes events through unchanged.

If you find the resources usefull, please cite the paper:

```
@InProceedings{Kowalczyk_2025_CVPR,
    author    = {Kowalczyk, Marcin and Jeziorek, Kamil and Kryjak, Tomasz},
    title     = {Learning from Noise: Enhancing DNNs for Event-Based Vision through Controlled Noise Injection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2025},
    pages     = {5131-5141}
}
```