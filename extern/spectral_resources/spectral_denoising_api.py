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
    get_next_center_point_of_spectral_image, read_bands_of_HDR_as_tensor, stitch_256x256_segments, \
    IMAGE_PATH_SWIR


DEBUG = True
VISUAL_CLIP_MIN_FACTOR = 50
VISUAL_CLIP_MAX_FACTOR = 0.9


def log(msg):
    if DEBUG:
        print(msg)


class SpectralAnomalyMarker:
    is_model_loaded: bool

    def __init__(self) -> None:
        self.is_model_loaded = False
        pass  # TODO:

    def prepare_input_for_model(self, improper_input): #  TODO: rename param
        permutated_bands = improper_input.permute(2, 0, 1).to('cuda')
        prepared_input_as_batch = permutated_bands.unsqueeze(0)

        return prepared_input_as_batch

    def load_model(self):
        model_loader = ModelLoader()
        
        if not self.is_model_loaded:
            self.unet_model, self.guided_diffusion = model_loader.get_model('256x256_uncond', path='models/256x256_diffusion_uncond.pt')
            self.is_model_loaded = True

    def find_anomalies(self, input_t, num_noising_timesteps=100, reconstruct_batch_size=16, 
                       should_use_direct_denoiser=False, should_display_progress=True):
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

        log('loading model...')
        self.load_model()
        log('SUCCESS.')

        noiser = TimestepUniformNoiser(self.guided_diffusion)

        if should_use_direct_denoiser:
            log('Direct denoiser is chosen.')
            log('speed: +, accuracy: -')
            denoiser = ModelTimestepDirectDenoiser(self.unet_model, self.guided_diffusion)
        else:
            log('Diffusion denoiser is chosen.')
            log('speed: -, accuracy: +')
            denoiser = ModelTimestepUniformDenoiser(self.unet_model, self.guided_diffusion)

        error_map_gen = BatchFilteredSquaredError()  # can be inherited from
        anomaly_scorer = MaxValueAnomalyScorer()  # can be inherited from

        reconstructed_batch = []

        for i in tqdm(range(reconstruct_batch_size)):
            current_timesteps = torch.randint(low=int(num_noising_timesteps * 0.9),
                                              high=int(num_noising_timesteps * 1.1),
                                              size=[1]).item()
            reconstructed_image = self.process_image(reconstruct_batch_size, i, current_timesteps, 
                                                     input_t, noiser, denoiser, 
                                                     should_display_progress)
            reconstructed_batch.append(reconstructed_image)

        # Aggregate results into a single tensor
        device = reconstructed_batch[0].device
        reconstructed_batch = torch.stack(reconstructed_batch).to(device)

        anomaly_map, anomaly_score = self.evaluate_anomaly(((input_t.permute(2, 0, 1) / 2) + 0.5),
                                                           reconstructed_batch,
                                                           error_map_gen,
                                                           anomaly_scorer)

        return anomaly_map, anomaly_score

    def process_image(self, reconstruct_batch_size, index_in_batch, current_timesteps, input_t, 
                      noiser, denoiser, should_display_progress):
        log('applying noise...')
        noised_image = noiser.apply_noise(input_t.unsqueeze(0), current_timesteps).squeeze(0)
        log(f'SUCCESS. {index_in_batch + 1}/{reconstruct_batch_size}')

        prepared_input = self.prepare_input_for_model(noised_image)

        log('applying denoise... (using the model)')
        with torch.no_grad():
            reconstructed_image = denoiser.denoise(prepared_input, current_timesteps, 
                                                   show_progress=should_display_progress)

            log(f'SUCCESS. {index_in_batch + 1}/{reconstruct_batch_size}')

            reconstructed_image_cpu = \
                ((reconstructed_image.squeeze(0).cpu() / 2) + 0.5).clip(0, 1)

            if should_display_progress:
                plt.imshow(reconstructed_image_cpu.permute(1, 2, 0))
                plt.show()

            return reconstructed_image_cpu

    def show_heatmap(self, heat_map_t, with_side_by_side=False, original=None, origin_t=None):
        if with_side_by_side:
            if original == None:
                print("show_heatmap with_side_by_side requires an original tensor")
            plt.subplot(1, 2, 1)
            plt.imshow(original)
        # visual_clip_min = heat_map_t.min().item() * VISUAL_CLIP_MIN_FACTOR
        visual_clip_min = heat_map_t.max().item() * 0.3
        visual_clip_max = heat_map_t.max().item() * VISUAL_CLIP_MAX_FACTOR
        
        plt.subplot(1, 2, 2)
        plt.imshow(heat_map_t, cmap='hot', interpolation='nearest', 
                   vmin=visual_clip_min, vmax=visual_clip_max)
        plt.show()

    def evaluate_anomaly(self, img: torch.Tensor,
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


# Unit Tests:


IS_QUIET_TESTS = True  # When False: displays every reconstruction in the batch synchrinically
TEST_BANDS = [10, 20, 40]
TEST_CENTER = (460, 220)
TEST_TIMESTEPS = 100
TEST_BATCH_SIZE = 16


def test_diffusion(anomaly_marker: SpectralAnomalyMarker):
    list_of_bands = TEST_BANDS
    spectral_image_path = IMAGE_PATH_SWIR
    selected_test_center = TEST_CENTER
    spectral_im_tensor = read_bands_of_HDR_as_tensor(list_of_bands, spectral_image_path)

    log('loading selected bands to tensor...')
    input_t = get_sliced_bands_of_HDR_as_256x256_tensor(spectral_im_tensor, selected_test_center)
    log('SUCCESS.')
    
    heatmap, anomaly_score = anomaly_marker.find_anomalies(input_t, TEST_TIMESTEPS, 
                                                           should_display_progress=not IS_QUIET_TESTS,
                                                           reconstruct_batch_size=TEST_BATCH_SIZE)
    print(f"anomaly_score: {anomaly_score}")
    anomaly_marker.show_heatmap(heatmap)


def test_direct(anomaly_marker: SpectralAnomalyMarker):
    list_of_bands = TEST_BANDS
    spectral_image_path = IMAGE_PATH_SWIR
    selected_test_center = TEST_CENTER
    spectral_im_tensor = read_bands_of_HDR_as_tensor(list_of_bands, spectral_image_path)

    log('loading selected bands to tensor...')
    input_t = get_sliced_bands_of_HDR_as_256x256_tensor(spectral_im_tensor, selected_test_center)
    log('SUCCESS.')

    heatmap, anomaly_score = anomaly_marker.find_anomalies(input_t, TEST_TIMESTEPS, 
                                                           should_display_progress=not IS_QUIET_TESTS, 
                                                           should_use_direct_denoiser=True,
                                                           reconstruct_batch_size=TEST_BATCH_SIZE)
    print(f"anomaly_score: {anomaly_score}")
    anomaly_marker.show_heatmap(heatmap)


def test_segment_iterator_direct(anomaly_marker: SpectralAnomalyMarker):
    list_of_bands = TEST_BANDS
    spectral_image_path = IMAGE_PATH_SWIR
    spectral_im_tensor = read_bands_of_HDR_as_tensor(list_of_bands, spectral_image_path)

    heatmaps = []
    anomaly_scores = []

    for current_center in get_next_center_point_of_spectral_image(spectral_im_tensor):
        log('loading selected bands to tensor...')
        input_t = get_sliced_bands_of_HDR_as_256x256_tensor(spectral_im_tensor, 
                                                            current_center)
        log('SUCCESS.')

        heatmap, anomaly_score = anomaly_marker.find_anomalies(input_t, TEST_TIMESTEPS,
                                                               should_display_progress=not IS_QUIET_TESTS, 
                                                               should_use_direct_denoiser=True,
                                                               reconstruct_batch_size=TEST_BATCH_SIZE)
        heatmaps.append(heatmap)
        anomaly_scores.append(anomaly_score)
    
    stitched_heatmap = stitch_256x256_segments(heatmaps, 
                                               size_x=spectral_im_tensor.shape[0], 
                                               size_y=spectral_im_tensor.shape[1])
    
    # TODO: something better and quieter like to files
    print(f"summarrized anomaly_score's: {sum(anomaly_scores)}")  
    anomaly_marker.show_heatmap(stitched_heatmap, with_side_by_side=True, 
                                original=spectral_im_tensor) 


def test_segment_iterator_diffuse(anomaly_marker: SpectralAnomalyMarker):
    list_of_bands = TEST_BANDS
    spectral_image_path = IMAGE_PATH_SWIR
    spectral_im_tensor = read_bands_of_HDR_as_tensor(list_of_bands, spectral_image_path)

    heatmaps = []
    anomaly_scores = []

    for current_center in get_next_center_point_of_spectral_image(spectral_im_tensor):
        log('loading selected bands to tensor...')
        input_t = get_sliced_bands_of_HDR_as_256x256_tensor(spectral_im_tensor, 
                                                            current_center)
        log('SUCCESS.')

        heatmap, anomaly_score = anomaly_marker.find_anomalies(input_t, TEST_TIMESTEPS,
                                                               should_display_progress=not IS_QUIET_TESTS,
                                                               reconstruct_batch_size=TEST_BATCH_SIZE)
        heatmaps.append(heatmap)
        anomaly_scores.append(anomaly_score)
    
    stitched_heatmap = stitch_256x256_segments(heatmaps, 
                                               size_x=spectral_im_tensor.shape[0], 
                                               size_y=spectral_im_tensor.shape[1])
    
    # TODO: something better and quieter like to files
    print(f"summarrized anomaly_score's: {sum(anomaly_scores)}")  
    anomaly_marker.show_heatmap(stitched_heatmap, with_side_by_side=True, 
                                original=spectral_im_tensor) 


def run_tests():
    """
    Every test is an example of how to use the SpectralAnomalyMarker
    """
    anomaly_marker = SpectralAnomalyMarker()
    test_segment_iterator_direct(anomaly_marker)
    # test_segment_iterator_diffuse(anomaly_marker)
    # test_direct(anomaly_marker)
    # test_diffusion(anomaly_marker)


if __name__ == '__main__':
    DEBUG = True
    IS_QUIET_TESTS = True
    run_tests()
