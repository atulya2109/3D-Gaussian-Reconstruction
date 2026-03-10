# 3D Human Body Avatar from Monocular Video

An end-to-end pipeline for creating animatable 3D human body avatars from monocular video using 3D Gaussian Splatting and SMPL body models. Inspired by FlashAvatar (CVPR 2024), extended to full-body reconstruction.

## Overview

This project takes a monocular video of a person and produces an animatable 3D avatar through three main stages:

1. **Segmentation** — Uses SAM2 (Segment Anything Model v2) to isolate the person from the background
2. **Body Estimation** — Uses SMPLer (transformer-based body estimator) to fit SMPL parametric body mesh to each frame
3. **Gaussian Splatting** — Trains 3D Gaussians anchored to the SMPL mesh to learn appearance while motion is driven by SMPL

## Project Structure

```
.
├── gaussian_renderer/     # Rendering engine for 3D Gaussians
├── models/               # Core models (GaussianModel, FrameMeshDataset)
├── utils/                # Utility functions (camera conversion, data loading)
├── smpl/                 # SMPL mesh topology data
├── data/                 # Dataset storage (frames, meshes, checkpoints)
├── submodules/           # External dependencies (diff-gaussian-rasterization)
├── train.py              # Training script
├── segment.py            # Segmentation module
├── *.ipynb              # Jupyter notebooks for experimentation
└── INTERVIEW.md          # Detailed technical documentation

## Requirements

- NVIDIA GPU with CUDA support (tested on RTX 4070 Super)
- Python 3.8+
- CUDA 11.8+ (for diff-gaussian-rasterization compilation)
- CMake 3.18+ (for building CUDA extensions)

## Installation

### 1. Clone the repository with submodules

```bash
git clone --recursive <repository-url>
cd DL
```

If you already cloned without `--recursive`:
```bash
git submodule update --init --recursive
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install SAM2

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### 4. Compile diff-gaussian-rasterization

```bash
cd submodules/diff-gaussian-rasterization
pip install .
cd ../..
```

### 5. Set up SMPL and SMPLer

- Download SMPL model files from [SMPL official website](https://smpl.is.tue.mpg.de/)
- Download SMPLer weights (follow SMPLer repository instructions)
- Place model files in appropriate directories (see data preparation below)

## Usage

### Data Preparation

Organize your data in the following structure:

```
data/
└── <subject_id>/
    ├── frames/          # Original video frames (1080x1920 PNG)
    ├── processed/       # Preprocessed frames (256x256 PNG, segmented)
    └── meshes/          # SMPL mesh data per frame (PKL files)
```

### Segmentation

```bash
python segment.py --input data/<subject_id>/frames --output data/<subject_id>/processed
```

Or use the notebook:
```bash
jupyter notebook segment.ipynb
```

### Training

```bash
python train.py
```

Training parameters are currently hardcoded in `train.py`. Key settings:
- Max iterations: 5000
- Batch size: 32
- Learning rates: rotation=0.001, scaling=0.001, opacity=0.01, features=0.005
- Output resolution: 256x256

Checkpoints are saved to `data/<subject_id>/chkpnt/` every 100 iterations.

Or use the training notebook:
```bash
jupyter notebook train.ipynb
```

## Technical Details

### Camera System

The project handles the challenging conversion from weak-perspective cameras (output by SMPLer) to full perspective cameras (required by Gaussian rasterization). See `INTERVIEW.md` for detailed technical discussion of this challenge.

### Gaussian Initialization

- One 3D Gaussian per SMPL mesh triangle (~13,776 Gaussians)
- Positions updated to triangle centroids each frame
- Appearance parameters (rotation, opacity, SH features, scaling) are learned
- Uses spherical harmonics (SH degree 3, 16 bands) for view-dependent appearance

### Known Issues

See `INTERVIEW.md` for comprehensive documentation of current limitations:
- Missing activation functions on Gaussian parameters
- Hardcoded camera conversion hacks
- No densification implementation
- Single fixed camera per training session

## Development

This is a research prototype. Key areas for improvement:

1. Replace camera conversion hacks with proper weak-perspective handling
2. Add activation functions (exp, sigmoid, normalize) to Gaussian parameters
3. Implement densification for better rendering quality
4. Add multi-view training support
5. Improve Gaussian positioning (barycentric coordinates or UV grid)

## References

- FlashAvatar: High-fidelity Head Avatar with Efficient Gaussian Embedding (CVPR 2024)
- 3D Gaussian Splatting for Real-Time Radiance Field Rendering
- SMPL: A Skinned Multi-Person Linear Model
- SAM2: Segment Anything in Images and Videos

## License

[Specify license here]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation information]
```

## Acknowledgments

- Built on [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- Uses Meta's SAM2 for segmentation
- Uses SMPL body model
- Inspired by FlashAvatar methodology
