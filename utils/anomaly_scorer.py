from abc import ABC, abstractmethod
import torch

class AnomalyScorer(ABC):
    """
    An abstract class that acts as an interface for anomaly scorers.

    An anomaly score is some aggregation of error maps between 
    two corresponding tensors.
    """

    def __call__(self, error_map: torch.Tensor, **kwargs) -> float:
        return self.score(error_map=error_map, **kwargs)

    @abstractmethod
    def score(self, error_map: torch.Tensor, **kwargs) -> float:
        """
        Method with which the anomaly scorer calculates the anomaly score.

        An instance of this class should override this method and provide
        an implementation.

        Parameters:
        -----------
        `error_map` : Tensor
        `**kwargs` : keyword arguments

        Return:
        -------
        `score` : float
            The error score calculated with respect to error_map.
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

    def score(self, error_map: torch.Tensor) -> float:
        """
        Calculates the anomaly score using the max value of the error-map.

        Parameters:
        -----------
        `error_map` : Tensor

        Return:
        -------
        `score` : float
            The error score calculated as the max value in error_map.
        """
        return error_map.max().to(float).item()

if __name__ == '__main__':
    # Benchmark the MaxValueAnomalyScorer class and print out the output
    anomaly_scorer = MaxValueAnomalyScorer()
    error_map_test = torch.randint(0, int(1e10), size=(3, 256, 256))

    print(anomaly_scorer.score(error_map_test))
