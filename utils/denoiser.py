import torch
from time import time
from typing import Optional
if __name__ == '__main__':
    import import_fixer
else:
    from . import import_fixer
from guided_diffusion.gaussian_diffusion import GaussianDiffusion
from guided_diffusion.unet import UNetModel
from abc import ABC, abstractmethod


class Denoiser(ABC):
    """
    An abstract class that acts as an interface for image denoisers.
    """
    @abstractmethod
    def denoise(self, images: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Takes in a batch of images and denoises them.

        The denoising process is dependant on the final class implementation.

        Parameters:
        -----------
        images : Tensor
            A batch of images of the shape (B, C, H, W)
        *args : [OPTIONAL] list of arguments
        *kwargs : [OPTIONAL] list of keyword arguments

        Return:
        -------
        A reconstructed version of the images from a noise(y) input.
        """
        return

class ModelTimestepUniformDenoiser(Denoiser):
    """
    A denoiser instance that uses a trained 256x256 class unconditional DDM
    in order to undo noise from a batch of images for a number of desired 
    timesteps progressively thus denoising them.
    """

    model: UNetModel
    diffusion: GaussianDiffusion

    def __init__(self, model: UNetModel, diffusion: GaussianDiffusion):
        self.model: UNetModel = model
        self.diffusion: GaussianDiffusion = diffusion

    def denoise(self, 
                noisy_images: torch.Tensor, 
                num_timesteps: int, 
                clip_denoised: bool=True, 
                show_progress: bool=False) -> torch.Tensor:
        self.diffusion.num_timesteps = num_timesteps  # Set the number of time-steps.
        denoised_images = self.diffusion.p_sample_loop(
            model=self.model,
            shape=noisy_images.shape,
            noise=noisy_images,
            clip_denoised=clip_denoised,
            progress=show_progress
        )

        return denoised_images

class ModelTimestepDirectDenoiser(ModelTimestepUniformDenoiser):
    """
    A denoiser instance that uses a trained 256x256 class unconditional DDM
    in order to undo noise from a batch of images for a number of desired 
    timesteps by directly predicting the image at t = 0.
    """

    def denoise(self, 
                noisy_images: torch.Tensor, 
                num_timesteps: int, 
                clip_denoised: bool=True,
                show_progress: bool=False) -> torch.Tensor:
        if show_progress:
            tic = time()

        device = next(self.model.parameters()).device
        self.diffusion.num_timesteps = num_timesteps  # Set the number of time-steps.
        denoised_images = self.diffusion.p_sample(
            model=self.model,
            x=noisy_images,
            t=torch.tensor([num_timesteps] * noisy_images.shape[0], device=device),
            clip_denoised=clip_denoised,
            device=device
        )

        if show_progress:
            toc = time()
            print(f"Denoising took a total of {toc - tic:.3f} second(s).")

        return denoised_images['pred_xstart']
