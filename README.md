# Introduction

**Anomaly-detection (AD)**: is a task where the goal is to find anomalous samples at inference time while during training, only positive (good) samples are given.  

**Denoising Diffusion Model (DDM)**: DDMs are models trained to recover images from noisy versions of the same image; they have recently been proven useful for many tasks (with a focus on generative models).  


## Our Objective

**Goal:** In this project we will attempt to develop a POC for detecting anomalies in images based on the ability or inability of a DDM to reconstruct them.  

**Example:** Inspection of a product in a factory may take images of all products on the product line. The goal may be to find scratched or damaged products, while during training no such samples were given.  

![image](https://user-images.githubusercontent.com/63167980/202312808-85b91816-6e06-4660-a8d9-b3329cd439b6.png)


## High Level Methodology

- Add random gaussian noise to an image.

- Denoise the noised image using a DDM and reconstruct the image.

- Calculate the “difference” between the original image and the reconstructed image and use the result to determine an anomality score.


# Setup and Execution

Download the 256x256 class unconditional model from here: [256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)

Place it in the path `models/256x256x_diffusion_uncond.pt` (relative path from the root directory of the repository).

## Execution
Run all the cells of `main_experiment.ipynb`.</br>You can see the output in the following paths:
- `output/` - The root folder of the outputs. Everything will be in here.
- `output/results.csv` - A csv table containing the columns `category | category_type | img_auc | pixel_auc` with the final results per category.
Will have the results of the **last** execution for each class. Can optimize and set `overwrite = False` in `main_experiment.ipynb` in the proper places and that will make sure that only categories that are not already present in the csv will be evaluated.
- `output/<category_name>/lightning_logs/` - Can find many output files here. Will have a version per evaluation performed. Each version will have a `sample` sub-directory inside of it that will included visualizations.
