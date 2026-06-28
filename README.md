# DynaMode: Generative Protein Dynamics with Spectral Diffusion

[![ICML 2026 GenBio Workshop](https://img.shields.io/badge/ICML-2026-blue.svg)](https://openreview.net/forum?id=0Cy0I8B9O2) 

<p align="center">
  <img src="assets/DynaMode.png" alt="DynaMode" width="100%">
</p>

Official implementation of [**DynaMode** (Spectral Diffusion for Protein Dynamics)](https://openreview.net/forum?id=0Cy0I8B9O2) accepted at ICML 2026 GenBio workshop. DynaMode is a  diffusion model trained on mdCATH to sample temporally coherent 256 frame (256ns) $C_\alpha$ monomer protein trajectories given an input structure and temperature. Diffusion in the DCT transformed spectral domain over the time domain leads to faster improved dynamics prediction over existing methods.

---

## Installation

```bash
conda env create -f dynamode.yaml
conda activate dynamode
```

## Datasets

2. **mdCATH** — Use script `scripts/download_mdcath.py` to download from [HuggingFace](https://huggingface.co/datasets/compsciencelab/mdCATH) using `hugginface_hub`.
3. **ATLAS** - Use script `scripts/download_atlas.py` to download from [ATLAS](https://www.dsimb.inserm.fr/ATLAS/index.html) using their ftp server.

### OPTIONAL Prepare Zarr Dataset for Fast Training

Using zarr files can immensely increase both stability of training and speed by precomputing 
features into numerical arrays and not relying on maintaining open file streaming of .h5's or
temporary pdb file streams which can cause training.

```bash
python -m scripts.extract_mdcath_features_to_zarr.py --[args]
python -m scripts.extract_atlas_features_to_zarr.py --[args]
```

## Pre-trained Checkpoint

Available soon

## Inference

WIP

## Training

WIP




## Citation

```bibtex
@article{}
```