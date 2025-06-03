# Diffusion-AD Project

This repository explores **anomaly detection** using *denoising diffusion models* (DDMs). The codebase contains a modular pipeline for training diffusion models, running anomaly detection experiments and benchmarking on the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad).

## Features
- **Few‑shot training** – train a diffusion model on a small number of images using `scripts/train.py`.
- **Automated benchmarking** – evaluate anomaly detection results for multiple categories with `scripts/benchmark_ad.py`.
- **Modular components** – noise/denoise wrappers, error‑map generators and anomaly scorers located under `utils/` and `core/`.
- **Persistent results** – experiment results are stored as CSV files under `output/` for easy comparison.

## Setup
1. Create a Conda environment and install dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate diffusion-ad
   ```
2. Download the 256×256 unconditional diffusion model from [OpenAI](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) and place it at `models/256x256_diffusion_uncond.pt`.
3. Place the MVTec dataset under `extern/mvtec/` (folder should contain sub‑directories such as `bottle`, `cable`, …).

## Running Experiments
### Few‑shot Training
Train a model on a small subset of images:
```bash
python scripts/train.py --data_dir extern/mvtec --target bottle --few_shot_count 10 --val_size 2
```
The script creates PyTorch Lightning logs under `output/train_logs/`.

### Benchmarking
Run anomaly detection on selected categories using a pretrained model:
```bash
python scripts/benchmark_ad.py --model models/256x256_diffusion_uncond.pt \
    --targets bottle cable carpet --reconstruction-batch-size 16
```
Results are written to `output/results.csv` and per‑category folders under `output/`.

## Results
Example combined results can be found in [`results/results_combined.csv`](results/results_combined.csv). A shortened excerpt:
```csv
category,category_type,img_auc,pixel_auc
bottle,object,0.88,0.92312313
cable,object,0.56,0.855968617
capsule,object,0.76,0.9547285
...
```

A static HTML site summarising the methodology and achievements is available in the [`site/`](site) directory.

## Acknowledgements
This project is based on the guided diffusion implementation and extends it for anomaly detection tasks.
