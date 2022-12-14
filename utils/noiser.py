import torch
if __name__ == '__main__':
  import import_fixer
else:
  from . import import_fixer
from guided_diffusion.gaussian_diffusion import GaussianDiffusion
from guided_diffusion.unet import UNetModel
from abc import ABC, abstractmethod


class Noiser(ABC):
    """
    An abstract class that acts as an interface for image noisers.
    """
    @abstractmethod
    def apply_noise(self, images: torch.TensorType, *args, **kwargs) -> torch.TensorType:
        """
        Takes in a batch of images and adds noise to them.

        Noise adding process is dependant on the final class implementation.

        Parameters:
        -----------
        images : Tensor
            A batch of images of the shape (B, C, H, W)
        *args : [OPTIONAL] list of arguments
        *kwargs : [OPTIONAL] list of keyword arguments

        Return:
        -------
        A noisy version of the images.
        """
        return

class TimestepUniformNoiser(Noiser):
    """
    A noiser instance that uses a noise-tensor to apply noise uniformly 
    to a batch of images for a number of desired timesteps.
    """
    diffusion: GaussianDiffusion

    def __init__(self, diffusion: GaussianDiffusion):
        self.diffusion = diffusion

    def apply_noise(self, 
                    images: torch.TensorType, 
                    num_timesteps: int, 
                    noise_tensor: torch.TensorType=None) -> torch.TensorType:
        """
        Initializes a timesteps tensor, takes in a batch of 
        images and adds noise to them.

        Noise adding process is dependant on the final class implementation.

        Parameters:
        -----------
        images : Tensor
            A batch of images of the shape (B, C, H, W)
        num_timesteps : int
            Number of timesteps to sample noise from
        noise_tensor : Tensor (default=None)
            A custom noise tensor (if is not None) used when applying noise to
            the given batch of images tensor

        Return:
        -------
        A noisy version of the images.
        """
        t_batch = torch.tensor([num_timesteps] * images.shape[0], device=images.device)

        return self.diffusion.q_sample(x_start=images, t=t_batch, noise=noise_tensor)
