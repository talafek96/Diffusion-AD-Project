from typing import Tuple
from random import sample
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.models import ModelLoader
from utils.noiser import Noiser, TimestepUniformNoiser
from utils.denoiser import Denoiser, ModelTimestepUniformDenoiser, ModelTimestepDirectDenoiser
from utils.error_map import ErrorMapGenerator, BatchFilteredSquaredError
from utils.anomaly_scorer import AnomalyScorer, MaxValueAnomalyScorer
from extern.guided_diffusion.guided_diffusion import gaussian_diffusion as gd
from extern.spectral_resources.load_and_crop_img import get_sliced_bands_of_HDR_as_256x256_tensor, \
    IMAGE_PATH_SWIR


DEBUG = True


def log(msg):
    if DEBUG:
        print(msg)


def prepare_input_for_model(improper_input): #  TODO: rename param
    permutated_bands = improper_input.permute(2, 0, 1).to('cuda')
    prepared_input_as_batch = permutated_bands.unsqueeze(0)

    return prepared_input_as_batch


def load_model():
    model_loader = ModelLoader()
    model, diffusion = model_loader.get_model('256x256_uncond', path='models/256x256_diffusion_uncond.pt')

    return model, diffusion


def show_heatmap(heat_map_t, with_overlay=False, origin_t=None):
    if not with_overlay:
        display(heat_map_t)  # TODO


def find_anomalies(selected_bands, image_path, center_point, num_noising_timesteps=100, 
                   reconstruct_batch_size=16, should_use_direct_denoiser=False, 
                   should_display_progress=True):
    """
    @param selected_bands: list of 3 integers indicating the selected channels to process
    @param image_path: path to spectral image
    @param center_point: a tuple of 2 integers indicating the center point of the 256x256 slice.
                         e.g: center_point = (128, 128) will get you the leftmost top segment.
    @param num_noising_timesteps: an integer indicating the number of noising steps applied.
    @param reconstruct_batch_size: the number of times a noising-reconstructing process is going 
                                   to happen before averaging the differences.
    @param should_use_direct_denoiser: a boolean flag s.t when True - applies a faster but less 
                                       accurate denoising
    @param should_display_progress: a boolean flag indicating the display of tqdm and 
                                    reconstructed images.

    @returns (Tensor): a heatmap tensor that signifies the difference from the average 
                       reconstruction and the original image.
    """
    if torch.cuda.is_available():
        log('CUDA: On')
    
    log('loading selected bands to tensor...')
    input_t = get_sliced_bands_of_HDR_as_256x256_tensor(selected_bands, image_path, center_point)
    log('SUCCESS.')
    
    log('loading model...')
    unet_model, guided_diffusion = load_model()
    log('SUCCESS.')

    noiser = TimestepUniformNoiser(guided_diffusion)

    if should_use_direct_denoiser:
        log('Direct denoiser is chosen.')
        log('speed: +, accuracy: -')
        denoiser = ModelTimestepDirectDenoiser(unet_model, guided_diffusion)
    else:
        log('Diffusion denoiser is chosen.')
        log('speed: -, accuracy: +')
        denoiser = ModelTimestepUniformDenoiser(unet_model, guided_diffusion)
    
    error_map_gen = BatchFilteredSquaredError()  # can be inherited from
    anomaly_scorer = MaxValueAnomalyScorer()  # can be inherited from
    
    reconstructed_batch = []

    for i in tqdm(range(reconstruct_batch_size)):
        current_timesteps = torch.randint(low=int(num_noising_timesteps * 0.9),
                                          high=int(num_noising_timesteps * 1.1),
                                          size=[1]).item()

        reconstructed_image_cpu = process_image(num_noising_timesteps, reconstruct_batch_size, 
                                                should_display_progress, input_t, noiser, denoiser, 
                                                reconstructed_batch, i, current_timesteps)
        
    # Aggregate results into a single tensor
    device = reconstructed_batch[0].device
    reconstructed_batch = torch.stack(reconstructed_batch).to(device)

    anomaly_map, anomaly_score = evaluate_anomaly(((input_t.permute(2, 0, 1) / 2) + 0.5).cuda(),
                                                  reconstructed_batch,
                                                  error_map_gen,
                                                  anomaly_scorer)

    return anomaly_map, anomaly_score


# generated using "extract method" TODO: reorder parameters so it's more readable
def process_image(num_noising_timesteps, reconstruct_batch_size, should_display_progress, 
                  input_t, noiser, denoiser, reconstructed_batch, i, current_timesteps):
    log('applying noise...')
    noised_image = noiser.apply_noise(input_t.unsqueeze(0), current_timesteps).squeeze(0)
    log(f'SUCCESS. {i}/{reconstruct_batch_size}')

    prepared_input = prepare_input_for_model(noised_image)

    log('applying denoise... (using the model)')
    with torch.no_grad():
        reconstructed_image = denoiser.denoise(prepared_input, current_timesteps, 
                                                   show_progress=should_display_progress)
        reconstructed_batch.append(reconstructed_image)

        log(f'SUCCESS. {i}/{reconstruct_batch_size}')

        if should_display_progress:
            reconstructed_image_cpu = ((reconstructed_image.squeeze(0).cpu() / 2) + 0.5).clip(0, 1)
            plt.imshow(reconstructed_image_cpu.permute(1, 2, 0))
            plt.show()

    return reconstructed_image_cpu


def display_anomaly_map(anomaly_map):
    # TODO: notice how vmin is defined. it's hardcoded for now but should actually be a hyperparameter
    plt.imshow(anomaly_map, cmap='hot', interpolation='nearest', vmin=anomaly_map.min().item() * 50, max=anomaly_map.max().item())


def evaluate_anomaly(img: torch.Tensor,
                     reconstructed_batch: torch.Tensor,
                     error_map_gen: ErrorMapGenerator,
                     anomaly_scorer: AnomalyScorer) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """
    Return:
    -------
    anomaly_map, anomaly_score
    """
    # Calculate an anomaly map using all of the results
    anomaly_map = error_map_gen.generate(img, reconstructed_batch)
    anomaly_score = anomaly_scorer.score(anomaly_map)

    return anomaly_map, anomaly_score


def test():
    model_input = prepare_input_for_model([10, 20, 40], IMAGE_PATH_SWIR, (130, 510))


if __name__ == '__main__':
    test()
