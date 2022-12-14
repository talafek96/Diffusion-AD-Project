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
    def apply_noise(images: torch.TensorType, *args, **kwargs) -> torch.TensorType:
        """Takes in a batch of images and adds noise to them.

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

class ModelTimestepUniformNoiser(Noiser):
    """
    A noiser instance that uses a trained 256x256 class unconditional DDM
    in order to apply noise to a batch of images for a number of desired 
    timesteps.
    # TODO: Edit this if needed to add timesteps spacing (probably need to)
    """
    def __init__(self, model: UNetModel):
        self.model = model
        # TODO

    def apply_noise(images: torch.TensorType, num_timesteps: int) -> torch.TensorType:
        # TODO
        pass
