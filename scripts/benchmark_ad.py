import os
import argparse
from typing import List
import tempfile
import pytorch_lightning as pl
from guided_diffusion import logger
from guided_diffusion.unet import UNetModel
from guided_diffusion.gaussian_diffusion import GaussianDiffusion
from utils.models import ModelLoader
from utils.noiser import TimestepUniformNoiser
from utils.denoiser import ModelTimestepUniformDenoiser, ModelTimestepDirectDenoiser
from utils.error_map import BatchFilteredSquaredError
from utils.anomaly_scorer import MaxValueAnomalyScorer
from core.diffusion_ad import DiffusionAD
from config.configuration import CATEGORY_TO_TYPE, DIFFUSION_AD_HPARAMS, DEFAULT_ROOT_OUTPUT_DIR


DEFAULT_LOG_BASE = DEFAULT_ROOT_OUTPUT_DIR
os.makedirs(DEFAULT_LOG_BASE, exist_ok=True)
tempfile.tempdir = DEFAULT_LOG_BASE


def parse_arguments():
    parser = argparse.ArgumentParser(description='Anomaly Detection Benchmark Pipeline',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog='For more information, please refer to the documentation.')

    # Add arguments
    parser.add_argument('--model', type=str, default=ModelLoader.MODEL_TO_ARG_SPECIFICS['256x256_uncond']['model_path'],
                        help='Path to the model. Default: %(default)s')
    parser.add_argument('--targets', '--target', type=str, nargs='+', choices=CATEGORY_TO_TYPE.keys(),
                        help='Target categories (space-separated)')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Allow overwriting previous result for a target category')
    parser.add_argument('--reconstruction-batch-size', type=int, default=DIFFUSION_AD_HPARAMS.reconstruction_batch_size,
                        help='Size of the reconstruction batch. Default: %(default)s')
    parser.add_argument('--direct-denoise', '-dd', action='store_true', default=False,
                        help='Denoise the noised images by directly predicting the original image at t = 0.\n'
                             'Uses the denoiser `ModelTimestepDirectDenoiser`.')

    # Summary of the pipeline
    pipeline_summary = '''
    This script implements an anomaly detection benchmark pipeline. It loads a diffusion denoising model 
    and performs automated anomaly detection evaluation. It requires specifying the target category/ies and 
    supports optional arguments for model path, result overwriting, and reconstruction batch size.
    '''

    # Add the pipeline summary to the help screen
    parser.description = parser.description + '\n' + pipeline_summary

    args = parser.parse_args()

    required_args = ['targets']  # List any required arguments here

    missing_args = [arg for arg in required_args if getattr(args, arg) is None]
    if missing_args:
        parser.error(
            f'The following arguments are required: {", ".join(missing_args)}')

    return args


def create_diffusion_ad(model: UNetModel, diffusion: GaussianDiffusion, model_name: str, use_direct_denoise: bool=False) -> DiffusionAD:
    # Create the components
    logger.log("Creating benchmark components and trainer...")
    noiser = TimestepUniformNoiser(diffusion)
    denoiser_class = ModelTimestepDirectDenoiser if use_direct_denoise else ModelTimestepUniformDenoiser
    denoiser = denoiser_class(model, diffusion)
    anomaly_map_generator = BatchFilteredSquaredError()
    anomaly_scorer = MaxValueAnomalyScorer()

    diffusion_ad = DiffusionAD(
        noiser,
        denoiser,
        anomaly_map_generator,
        anomaly_scorer,
        DIFFUSION_AD_HPARAMS,
        model_name=model_name)

    return diffusion_ad


def run_benchmark(model_path: str, target_categories: List[str], should_overwrite: bool=False, use_direct_denoise: bool=False):
    # Load the model
    loader = ModelLoader()
    model_name = loader.get_model_name('256x256_uncond', path=model_path)
    logger.log(f"Loading model {model_name}...")
    model, diffusion = loader.get_model('256x256_uncond', path=model_path)

    diffusion_ad = create_diffusion_ad(model, diffusion, model_name, use_direct_denoise)

    # Create a PyTorch Lightning trainer for each target
    log_dir_base = os.path.join(diffusion_ad.args.root_output_dir, model_name)
    logger.log(
        f"Creating trainers and setting base log path to '{log_dir_base}'")
    trainer = {
        target: pl.Trainer.from_argparse_args(diffusion_ad.args,
                                              default_root_dir=os.path.join(
                                                  log_dir_base, target),
                                              max_epochs=diffusion_ad.args.num_epochs,
                                              accelerator='gpu',
                                              devices=1)
        for target in target_categories
    }

    remaining_categories = diffusion_ad.get_remaining_categories()
    logger.log(f'Target categories: {target_categories}\n'
               f'Remaining categories: {set(target_categories) & remaining_categories}\n'
               f'Overwrite: {should_overwrite}')

    for category in target_categories:
        if should_overwrite or category in remaining_categories:
            logger.log('\033[4m\033[1m' +
                       f'Evaluating category: {category}' + '\033[0m')
            diffusion_ad.args.category = category
            try:
                trainer[category].test(diffusion_ad)

            except FileNotFoundError as e:
                logger.log(
                    e, f'\nFiles missing for category {category}! Please fix your dataset folder.\n')


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Setup logger
    logger.configure()

    # Load the model based on the provided path or default value
    model_path: str = args.model

    # Get the target categories
    target_categories: List[str] = args.targets

    # Update the DIFFUSION_AD_HPARAMS dictionary
    DIFFUSION_AD_HPARAMS.reconstruction_batch_size = args.reconstruction_batch_size

    # Run the benchmark pipeline
    run_benchmark(model_path, target_categories, args.overwrite, args.direct_denoise)


if __name__ == '__main__':
    main()
