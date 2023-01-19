import os
import numpy as np
from argparse import Namespace

_DEFAULT_TIMESTEPS = 250
_DEFAULT_V_MIN_MAX = (0.01, 0.1)
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
    'save_anomaly_map'
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
    'capsule': 350,
    'carpet': _DEFAULT_TIMESTEPS,
    'grid': 300,
    'hazelnut': 250,
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

CATEGORY_TO_V_MIN_MAX = {
    'bottle': _DEFAULT_V_MIN_MAX,
    'cable': _DEFAULT_V_MIN_MAX,
    'capsule': (0.04, 0.08),
    'carpet': _DEFAULT_V_MIN_MAX,
    'grid': (0.025, 0.09),
    'hazelnut': (0.02, 0.08),
    'leather': _DEFAULT_V_MIN_MAX,
    'metal_nut': _DEFAULT_V_MIN_MAX,
    'pill': _DEFAULT_V_MIN_MAX,
    'screw': _DEFAULT_V_MIN_MAX,
    'tile': _DEFAULT_V_MIN_MAX,
    'toothbrush': _DEFAULT_V_MIN_MAX,
    'transistor': _DEFAULT_V_MIN_MAX,
    'wood': _DEFAULT_V_MIN_MAX,
    'zipper': _DEFAULT_V_MIN_MAX
}
