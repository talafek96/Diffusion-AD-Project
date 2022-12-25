from abc import ABC, abstractmethod
import torch

class AnomalyScorer(ABC):
    """
    An abstract class that acts as an interface for anomaly scorer.

    An anomaly score is some aggregation of error maps between 
    two corresponding tensors.
    """

    @abstractmethod
    def score(self, error_map: torch.TensorType, **kwargs) -> float:
        """
        Method with which the anomaly scorer calculates the anomaly score.

        An instance of this class should override this method and provide
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


class MaxValueAnomalyScorer(AnomalyScorer):
    """
    A scorer class calculating anomaly score of a given error-map tensor
    using max function.
    In other words: The anomaly score of a given error-map would be the
                    max value in said error-map
    """

    def __init__(self) -> None:
        pass

    def __call__(self, error_map: torch.TensorType, **kwargs) -> float:
        return self.score(error_map=error_map, **kwargs)

    def score(self, error_map: torch.TensorType, **kwargs) -> float:
        return torch.max(error_map)
