from typing import Tuple
import tqdm
import torch
import matplotlib.pyplot as plt
from utils.models import ModelLoader
from utils.noiser import Noiser, TimestepUniformNoiser
from utils.denoiser import Denoiser, ModelTimestepUniformDenoiser
from utils.error_map import ErrorMapGenerator, BatchFilteredSquaredError
from utils.anomaly_scorer import AnomalyScorer, MaxValueAnomalyScorer
from diffusion_ad.base_algo import BaseAlgo


class DiffusionAD(BaseAlgo):
    def __init__(self, hparams):
        super().__init__(hparams)

        # Load the model and save it
        self.model = ModelLoader().get_model('256x256_uncond')

    def get_reconstructed_batch(img: torch.TensorType,
                                noiser: Noiser,
                                denoiser: Denoiser,
                                num_timesteps: int,
                                batch_size: int,
                                interactive_print: bool = False) -> torch.TensorType:
        """
        Using a noiser and a denoiser, adds noise for `num_timesteps` steps to the given img
        `batch_size` times, and reconstructs each noised image.

        Paramters:
        ----------
        `img` : torch.TensorType (C, H, W)
            An image stored as a tensor.
        
        `noiser` : Noiser
        
        `denoiser` : Denoiser
        
        `num_timesteps` : int
            Number of timesteps for both the noiser and the denoiser to add and remove noise in.
        
        `batch_size` : int
            Wanted number of reconstructed images to be generated from `img`.
        
        `interactive_print` : bool
            [OPTIONAL] If true, will display the reconstructed images during the generation process. Default: False.

        Return:
        -------
        `reconstructed_batch`: Tensor (B, C, H, W)
            A batch of batch_size reconstructed images from `img`.
        """
        reconstructed_batch = []

        # Noise and reconstruct `batch_size` times and aggregate into a batch
        for i in tqdm(range(batch_size)):
            curr_timesteps = torch.randint(
                low=int(num_timesteps * 0.9), high=int(num_timesteps * 1.1), size=[1]).item()
            noised_image = noiser.apply_noise(
                img.unsqueeze(0), curr_timesteps).squeeze(0).cuda()
            reconstructed_image = denoiser.denoise(
                noised_image.unsqueeze(0), curr_timesteps, show_progress=True)

            if interactive_print:
                print(f'Reconstructed image No. {i + 1}:')
                reconstructed_image_cpu = (
                    (reconstructed_image.squeeze(0).cpu() / 2) + 0.5).clip(0, 1)
                plt.imshow(reconstructed_image_cpu.permute(1, 2, 0))
                plt.show()

        reconstructed_batch.append(
            ((reconstructed_image.squeeze(0) / 2) + 0.5).clip(0, 1))

        # Aggregate results into a single tensor
        device = reconstructed_batch[0].device
        reconstructed_batch = torch.stack(reconstructed_batch).to(device)

        return reconstructed_batch

    def evaluate_anomaly(img: torch.TensorType,
                         reconstructed_batch: torch.TensorType,
                         error_map_gen: ErrorMapGenerator,
                         anomaly_scorer: AnomalyScorer) -> Tuple[torch.TensorType, float]:
        """
        Given an image, and a batch of image that were reconstructed from noisy versions of it,
        calculates both an anomaly map and an anomaly score.

        Parameters:
        -----------
        `img` : Tensor
            Shape of tensor determined by the requirements of the ErrorMapGenerator object.
        
        `reconstructed_batch` : Tensor
            Shape of tensor determined by the requirements of the ErrorMapGenerator object.
        
        `error_map_gen` : ErrorMapGenerator

        `anomaly_scorer` : AnomalyScorer

        Return:
        -------
        `anomaly_map` : TensorType, `anomaly_score` : float
        """
        # Calculate an anomaly map using all of the results
        anomaly_map = error_map_gen.generate(img, reconstructed_batch)
        anomaly_score = anomaly_scorer.score(anomaly_map)

        return anomaly_map, anomaly_score

    def predict_scores(self, img):
        """
        Computes test time score prediction.
        Returns per pixel scores (B,H,W) and image scores (B,) (numpy arrays).
        This instance assumes that B == 1.

        Parameters:
        -----------
        `img` : Tensor (B, H, W)
            The image to predict the scores for.
        
        Return:
        -------
        `anomaly_map` : ndarray (B, H, W), `image_score` : ndarray (B,)
        """
        pass
