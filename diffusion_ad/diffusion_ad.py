from typing import Tuple
from tqdm.auto import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.noiser import Noiser
from utils.denoiser import Denoiser
from utils.error_map import ErrorMapGenerator
from utils.anomaly_scorer import AnomalyScorer
from diffusion_ad.base_algo import BaseAlgo
from config.configuration import DIFFUSION_AD_REQUIRED_HPARAMS, CATEGORY_TO_NOISE_TIMESTEPS


class DiffusionAD(BaseAlgo):
    noiser: Noiser
    denoiser: Denoiser
    anomaly_map_generator: ErrorMapGenerator
    anomaly_scorer: AnomalyScorer

    def __init__(self, noiser, denoiser, anomaly_map_generator, anomaly_scorer, hparams):
        assert all(param in hparams for param in DIFFUSION_AD_REQUIRED_HPARAMS)

        super().__init__(hparams)

        if 'verbosity' not in self.args:
            self.args.verbosity = 0
            
        self.args.saved_model_path = None

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
        iterations = tqdm(range(batch_size))
        
        for i in iterations:
            iterations.set_description(f'Reconstructions Done {i}/{self.args.reconstruction_batch_size}', refresh=True)

            curr_timesteps = torch.randint(
                low=int(num_timesteps * 0.9), high=int(num_timesteps * 1.1), size=[1]).item()
            noised_image = noiser.apply_noise(
                img.unsqueeze(0), curr_timesteps).squeeze(0).cuda()
            reconstructed_image = denoiser.denoise(
                noised_image.unsqueeze(0), curr_timesteps, show_progress=True)

            if interactive_print:
                print(f'Reconstructed image No. {i + 1}:')
                reconstructed_image_to_print = (reconstructed_image.squeeze(0).cpu() / 2) + 0.5  # Transform into the dynamic range (0, 1)
                plt.imshow(reconstructed_image_to_print.permute(1, 2, 0))
                plt.show()
            
            iterations.set_description(f'Reconstructions Done {i + 1}/{self.args.reconstruction_batch_size}', refresh=True)

            reconstructed_batch.append(reconstructed_image.squeeze(0))

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
        anomaly_map = error_map_gen.generate(img, reconstructed_batch, **self.args.anomaly_map_generator_kwargs)
        anomaly_score = anomaly_scorer.score(anomaly_map, **self.args.anomaly_scorer_kwargs)

        return anomaly_map, anomaly_score

    def predict_scores(self, img: torch.Tensor):
        """
        Computes test time score prediction.
        Returns per pixel scores (B, H, W) and image scores (B,) (numpy arrays).
        This instance assumes that B == 1.

        Parameters:
        -----------
        `img` : Tensor (B, C, H, W)
            The image to predict the scores for.
            Expects B == 1.
        `category` : str
            The category/class of the image.
        
        Return:
        -------
        `anomaly_map` : ndarray (B, H, W), `image_score` : ndarray (B,)
        """
        num_timesteps = CATEGORY_TO_NOISE_TIMESTEPS[self.args.category]
        img = self.data_transforms(img.squeeze(0).squeeze(0))

        print("img min:", img.min(), "img max:", img.max())

        if self.args.verbosity >= 2:
            # Show the transformed input image
            image_to_print = (img.cpu() / 2) + 0.5
            print('Transformed input image:')
            plt.imshow(image_to_print.permute(1, 2, 0))
            plt.show()

        reconstructed_images = self.get_reconstructed_batch(img,
                                                            self.noiser,
                                                            self.denoiser,
                                                            num_timesteps,
                                                            self.args.reconstruction_batch_size,
                                                            interactive_print=self.args.verbosity >= 2)

        anomaly_map, anomaly_score = self.evaluate_anomaly(img,
                                                           reconstructed_images,
                                                           self.anomaly_map_generator,
                                                           self.anomaly_scorer)

        return anomaly_map.cpu().numpy(), np.array([anomaly_score])
