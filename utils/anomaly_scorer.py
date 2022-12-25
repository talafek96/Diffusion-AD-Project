from abc import ABC, abstractmethod
import torch

import error_map

class AnomalyScorer(ABC):
    """
    An abstract class that acts as an interface for anomaly scorer.

    An anomaly score is some aggregation of error maps between 
    two corresponding tensors.
    """

    @abstractmethod
    def score(self, error_map: torch.TensorType, **kwargs) -> float:
        return


class MaxValueAnomalyScorer(AnomalyScorer):
    """
    A scorer class calculating anomaly score of a given error-map tensor
    using max function.
    In other words: The anomaly score of a given error-map would be the
                    max value in said error-map
    """

    def __init__(self) -> None:
        super().__init__()  # should i remove this line since it's abstract?

    def __call__(self, error_map: torch.TensorType, **kwargs) -> float:
        return self.score(error_map=error_map, kwargs=kwargs)  # TODO: is that how kwargs should be passed?

    # TODO: should we scale? normalize? clip? (before returning the max value)
    def score(self, error_map: torch.TensorType, **kwargs) -> float:
        return torch.max(error_map)