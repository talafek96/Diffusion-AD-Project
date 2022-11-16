# Introduction

**Anomaly-detection (AD)**: is a task where the goal is to find anomalous samples at inference time while during training, only positive (good) samples are given.  

**Denoising Diffusion Model (DDM)**: DDMs are models trained to recover images from noisy versions of the same image; they have recently been proven useful for many tasks (with a focus on generative models).  


# Our Objective

**Goal:** In this project we will attempt to develop a POC for detecting anomalies in images based on the ability or inability of a DDM to reconstruct them.  

**Example:** Inspection of a product in a factory may take images of all products on the product line. The goal may be to find scratched or damaged products, while during training no such samples were given.  

![image](https://user-images.githubusercontent.com/63167980/202312808-85b91816-6e06-4660-a8d9-b3329cd439b6.png)


# High Level Methodology

- Add noise to an image in a certain to be defined manner.

- Denoise the noised image using a DDM and reconstruct the image.

- Calculate the “difference” between the original image and the reconstructed image and use the result to determine an anomality score.
