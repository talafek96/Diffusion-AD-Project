from typing import Tuple
import tqdm
import torch
import matplotlib.pyplot as plt
from utils.noiser import Noiser, TimestepUniformNoiser
from utils.denoiser import Denoiser, ModelTimestepUniformDenoiser
from utils.error_map import ErrorMapGenerator, BatchFilteredSquaredError
from utils.anomaly_scorer import AnomalyScorer, MaxValueAnomalyScorer
from diffusion_ad.base_algo import BaseAlgo

DIFFUSION_AD_REQUIRED_HPARAMS = ['reconstruction_batch_size', 'anomaly_map_generator_kwargs', 'anomaly_scorer_kwargs']
CATEGORY_TO_NOISE_TIMESTEPS = dict()


class DiffusionAD(BaseAlgo):
    noiser: Noiser
    denoiser: Denoiser
    anomaly_map_generator: ErrorMapGenerator
    anomaly_scorer: AnomalyScorer

    def __init__(self, noiser, denoiser, anomaly_map_generator, anomaly_scorer, hparams):
        assert all(param in hparams for param in DIFFUSION_AD_REQUIRED_HPARAMS)

        super().__init__(hparams)

        if 'verbosity' not in self.hparams.keys():
            self.hparams['verbosity'] = 0

        # Initiate members
        self.noiser = noiser
        self.denoiser = denoiser
        self.anomaly_map_generator = anomaly_map_generator
        self.anomaly_scorer = anomaly_scorer

    def get_reconstructed_batch(self,
                                img: torch.TensorType,
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

    def evaluate_anomaly(self,
                         img: torch.TensorType,
                         reconstructed_batch: torch.TensorType,
                         error_map_gen: ErrorMapGenerator,
                         anomaly_scorer: AnomalyScorer) -> Tuple[torch.Tensor, float]:
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
        `anomaly_map` : Tensor, `anomaly_score` : float
        """
        # Calculate an anomaly map using all of the results
        anomaly_map = error_map_gen.generate(img, reconstructed_batch, **self.hparams['anomaly_map_generator_kwargs'])
        anomaly_score = anomaly_scorer.score(anomaly_map, **self.hparams['anomaly_scorer_kwargs'])

        return anomaly_map, anomaly_score

    def predict_scores(self, img: torch.Tensor, category: str):
        """
        Computes test time score prediction.
        Returns per pixel scores (B, H, W) and image scores (B,) (numpy arrays).
        This instance assumes that B == 1.

        Parameters:
        -----------
        `img` : Tensor (B, H, W)
            The image to predict the scores for.
        `category` : str
            The category/class of the image.
        
        Return:
        -------
        `anomaly_map` : ndarray (B, H, W), `image_score` : ndarray (B,)
        """
        num_timesteps = CATEGORY_TO_NOISE_TIMESTEPS[category]
        reconstructed_images = self.get_reconstructed_batch(img,
                                                            self.noiser,
                                                            self.denoiser,
                                                            num_timesteps,
                                                            self.hparams['reconstruction_batch_size'],
                                                            interactive_print=self.hparams['verbosity'] >= 1)
        anomaly_map, anomaly_score = self.evaluate_anomaly(reconstructed_images,
                                                           self.anomaly_map_generator,
                                                           self.anomaly_scorer)

        return anomaly_map, anomaly_score
