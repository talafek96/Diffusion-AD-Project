import os
from argparse import Namespace
import numpy as np

_DEFAULT_TIMESTEPS = 250
_DEFAULT_RECON_BATCH_SIZE = 5
_DEFAULT_OUTPUT_DIR_NAME = 'output'

MAGIC_NORMALIZE_MEAN = np.array([-0.229/0.485, -0.224/0.456, -0.255/0.406])
MAGIC_NORMALIZE_STD = np.array([0.229, 0.224, 0.255])

DIFFUSION_AD_REQUIRED_HPARAMS = [
    'reconstruction_batch_size', 
    'anomaly_map_generator_kwargs', 
    'anomaly_scorer_kwargs',
    'phase',
    'category',
    'dataset_path',
    'num_epochs',  # should be always 1 for our usage
    'batch_size',
    'load_size',
    'input_size',
    'root_output_dir',
    'save_anomaly_map',
]

DIFFUSION_AD_HPARAMS = Namespace(**{
    'reconstruction_batch_size': _DEFAULT_RECON_BATCH_SIZE,
    'anomaly_map_generator_kwargs': {}, 
    'anomaly_scorer_kwargs': {},
    'phase': 'test',
    'category': 'hazelnut',
    'dataset_path': os.path.abspath(os.path.join(__file__, '..', '..', 'extern', 'mvtec')),
    'num_epochs': 1,
    'batch_size': 1,
    'load_size': 256,
    'input_size': 256,
    'root_output_dir': os.path.abspath(os.path.join(__file__, '..', '..', _DEFAULT_OUTPUT_DIR_NAME)),
    'save_anomaly_map': True
})

CATEGORY_TO_NOISE_TIMESTEPS = {
    'bottle': _DEFAULT_TIMESTEPS,
    'cable': _DEFAULT_TIMESTEPS,
    'capsule': _DEFAULT_TIMESTEPS,
    'carpet': _DEFAULT_TIMESTEPS,
    'grid': _DEFAULT_TIMESTEPS,
    'hazelnut': _DEFAULT_TIMESTEPS,
    'leather': _DEFAULT_TIMESTEPS,
    'metal_nut': _DEFAULT_TIMESTEPS,
    'pill': _DEFAULT_TIMESTEPS,
    'screw': _DEFAULT_TIMESTEPS,
    'tile': _DEFAULT_TIMESTEPS,
    'toothbrush': _DEFAULT_TIMESTEPS,
    'transistor': _DEFAULT_TIMESTEPS,
    'wood': _DEFAULT_TIMESTEPS,
    'zipper': _DEFAULT_TIMESTEPS
}
