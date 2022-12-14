import torch
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
    def denoise(images: torch.TensorType, *args, **kwargs) -> torch.TensorType:
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
    timesteps thus denoising them.
    """
    def __init__(self, model: UNetModel):
        self.model = model
        # TODO

    def apply_noise(images: torch.TensorType, num_timesteps: int) -> torch.TensorType:
        # TODO
        pass
