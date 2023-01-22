import os
import numpy as np
from argparse import Namespace

# Hidden default constants
_CATEGORY_TYPE_OBJECT = 'object'
_CATEGORY_TYPE_TEXTURE = 'texture'
_DEFAULT_TIMESTEPS = 250
_DEFAULT_V_MIN_MAX = (0.01, 1)
_DEFAULT_RECON_BATCH_SIZE = 3
_DEFAULT_OUTPUT_DIR_NAME = 'output'
_DEFAULT_RESULTS_CSV = 'results.csv'

# Magic constants
MAGIC_NORMALIZE_MEAN = np.array((0.5, 0.5, 0.5))  # np.array([0.485, 0.456, 0.406])
MAGIC_NORMALIZE_STD = np.array((0.5, 0.5, 0.5))  # np.array([0.229, 0.224, 0.255])

# Public default constants
UNLIMITED_MAX_TEST_IMAGES = 0  # 0 is unlimited
DEFAULT_AUGMENT_NAME = ['basic']
DEFAULT_DATASET_PATH = os.path.abspath(os.path.join(__file__, '..', '..', 'extern', 'mvtec'))
DEFAULT_ROOT_OUTPUT_DIR = os.path.abspath(os.path.join(__file__, '..', '..', _DEFAULT_OUTPUT_DIR_NAME))
DEFAULT_CSV_DATA_PATH = os.path.abspath(os.path.join(DEFAULT_ROOT_OUTPUT_DIR, _DEFAULT_RESULTS_CSV))

# Public datastructures

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
    'results_csv_path',
    'save_anomaly_map'
]

DIFFUSION_AD_HPARAMS = Namespace(**{
    'reconstruction_batch_size': _DEFAULT_RECON_BATCH_SIZE,
    'anomaly_map_generator_kwargs': {}, 
    'anomaly_scorer_kwargs': {},
    'phase': 'test',
    'category': 'hazelnut',
    'dataset_path': DEFAULT_DATASET_PATH,
    'num_epochs': 1,
    'batch_size': 1,
    'load_size': 256,
    'input_size': 256,
    'root_output_dir': DEFAULT_ROOT_OUTPUT_DIR,
    'augment': DEFAULT_AUGMENT_NAME,
    'results_csv_path': DEFAULT_CSV_DATA_PATH,
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
    'hazelnut': (0.02, 0.27),
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

CATEGORY_TO_TYPE = {
    'bottle': _CATEGORY_TYPE_OBJECT,
    'cable': _CATEGORY_TYPE_OBJECT,
    'capsule': _CATEGORY_TYPE_OBJECT,
    'carpet': _CATEGORY_TYPE_TEXTURE,
    'grid': _CATEGORY_TYPE_TEXTURE,
    'hazelnut': _CATEGORY_TYPE_OBJECT,
    'leather': _CATEGORY_TYPE_TEXTURE,
    'metal_nut': _CATEGORY_TYPE_OBJECT,
    'pill': _CATEGORY_TYPE_OBJECT,
    'screw': _CATEGORY_TYPE_OBJECT,
    'tile': _CATEGORY_TYPE_TEXTURE,
    'toothbrush': _CATEGORY_TYPE_OBJECT,
    'transistor': _CATEGORY_TYPE_OBJECT,
    'wood': _CATEGORY_TYPE_TEXTURE,
    'zipper': _CATEGORY_TYPE_TEXTURE
}
