from abc import ABC, abstractmethod
import torch
from torchvision.transforms.functional import gaussian_blur


class ErrorMapGenerator(ABC):
    """
    An abstract class that acts as an interface for error map generators.

    An error map is a pixel-wise error between two corresponding tensors.
    """

    def __call__(self, x: torch.TensorType, y: torch.TensorType, *args, **kwargs):
        return self.generate(*args, **kwargs)
    
    @abstractmethod
    def generate(self, x: torch.TensorType, y: torch.TensorType, *args, **kwargs) -> torch.TensorType:
        """
        Method with which the error map generator creates an error map.

        An instance of this calss should override this method and provide
        an implementation.

        Parameters:
        -----------
        `x` : Tensor
        `y` : Tensor
        `**kwargs` : keyword arguments

        Return:
        -------
        `E` : tensor
            The error map calculated with respect to the difference between x and y.
        """
        return


class BatchFilteredSquaredError(ErrorMapGenerator):
    """
    Generates an squared error based error-map between a single tensor x and a batch of
    tensors y, filtered using an optional filter (gaussian filter by default), according 
    to the method explained in IBM's MAEDAY article.

    Method:
    -------
    Let N be the batch size of y.

    Calculates N different error maps E(i), while for each i:
    
    `E(i) = sum_over_channels(filter((x - y)**2))`

    Then the N error maps are averaged to get a single pixel-level error map:

    `E = (1 / N) * sum([E(i) for i in 1, 2, ..., N])])`
    """

    def __init__(self):
        # Create the kernel types to function mapping
        self._kernel_size = 7
        self._sigma = 1.4
        self.kernels = {
            'gaussian': lambda img: gaussian_blur(img, self._kernel_size, self._sigma),
            'none': lambda img: img,
        }

    def _reset_defaults(self) -> None:
        self._kernel_size = 7
        self._sigma = 1.4

    def generate(self, x: torch.TensorType,
                       y: torch.TensorType,
                       kernel_type: str='gaussian',
                       **kwargs) -> torch.TensorType:
        """
        Calculates the error map.

        Parameters:
        -----------
        `x` : Tensor
            A tensor of the shape [C, H, W].
        
        `y` : Tensor
            A batch tensor of the shape [B, C, H, W] whereas the batch
            dimension is the first dimension, and all the other dimensions
            are from the same shape as x.

        `kernel_type` : str
            One of the supported kernel types -
                `gaussian` - A gaussian kernel based low pass filter used to remove noise.
                             Uses the default parameters of kernel size = 7, sigma = 1.4.
                             In order to use different parameters, pass them using the
                             keyword arguments `kernel_size` and `sigma`.
                `none` - No filter will be used.

        Return:
        -------
        `E` : Tensor
            An error map as described in the class documentation.
        """
        assert isinstance(kernel_type, (type(None), str)), 'Bad "kernel_type" type.'
        kernel_type = 'none' if kernel_type is None else kernel_type.lower().strip()
        if kernel_type not in self.kernels:
            raise RuntimeError(f'Bad argument "kernel_type". Expected one of: {list(self.kernels.keys())}, found: {kernel_type}')
        
        self._reset_defaults()

        if 'kernel_size' in kwargs:
            self._kernel_size = kwargs['kernel_size']
        if 'sigma' in kwargs:
            self._sigma = kwargs['sigma']
        
        x = x.unsqueeze(0).to(float)
        y = y.to(float)
        if x.shape[1:] != y.shape[1:]:
            raise RuntimeError('The shapes of x and y do not match!\n'
                               '\tShapes should be equal in all dimensions but the batch dimension of y.\n'
                               f'\t\tx shape: {x.shape[1:]}, y shape: {y.shape}.')

        return self.kernels[kernel_type]((x - y) ** 2).sum(axis=1).mean(axis=0)


if __name__ == '__main__':
    # Benchmark the BatchFilteredSquaredError class and print out the output
    error_map_gen = BatchFilteredSquaredError()
    x = torch.randint(0, 10, size=(3, 256, 256))
    y = torch.randint(0, 10, size=(2, 3, 256, 256))
    print(error_map_gen.generate(x, y, kernel_type='gaussian'))
