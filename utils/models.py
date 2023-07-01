import os
import re
import torch
if __name__ == '__main__':
  import import_fixer
else:
  from . import import_fixer
from guided_diffusion import dist_util
from guided_diffusion.gaussian_diffusion import GaussianDiffusion
from guided_diffusion.unet import UNetModel
from guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion
from typing import Tuple


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ModelLoader:
    MODEL_TO_ARG_SPECIFICS = {
        '64x64': {
            'model_flags': {
                'attention_resolutions': "32,16,8",
                'class_cond': True,
                'diffusion_steps': 1000,
                'dropout': 0.1,
                'image_size': 64,
                'learn_sigma': True,
                'noise_schedule': 'cosine',
                'num_channels': 192,
                'num_head_channels': 64,
                'num_res_blocks': 3,
                'resblock_updown': True,
                'use_new_attention_order': True,
                'use_fp16': True if DEVICE == 'cuda' else False,
                'use_scale_shift_norm': True
            },
            'model_path': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', '64x64_diffusion.pt'))
        },
        '256x256_uncond': {
            'model_flags': {
                'attention_resolutions': "32,16,8",
                'class_cond': False,
                'diffusion_steps': 1000,
                'image_size': 256,
                'learn_sigma': True,
                'noise_schedule': 'linear',
                'num_channels': 256,
                'num_head_channels': 64,
                'num_res_blocks': 2,
                'resblock_updown': True,
                'use_fp16': True if DEVICE == 'cuda' else False,
                'use_scale_shift_norm': True
            },
            'model_path': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', '256x256_diffusion_uncond.pt'))
        },
    }

    def __init__(self):
        self.default_args: dict = dict(
            use_ddim=False,
            model_path="",
        )
        self.default_args.update(model_and_diffusion_defaults())

    def get_model(self, model_name: str, extra_args: dict=None, to_compile: bool=True) -> Tuple[UNetModel, GaussianDiffusion]:
        """
        Creates UNetModel and GaussianDiffusion objects, and loads the trained model from the disk.
        """
        model_name = model_name.strip().lower()  # Normalize string to convention

        if model_name not in type(self).MODEL_TO_ARG_SPECIFICS.keys():
            raise RuntimeError(
                f'Model name "{model_name}" not supported.\nChoose one of: {list(type(self).MODEL_TO_ARG_SPECIFICS.keys())}')
        
        if not os.path.exists(type(self).MODEL_TO_ARG_SPECIFICS[model_name]['model_path']):
            raise RuntimeError(
                f'The trained model .pt file was not found in:\n{type(self).MODEL_TO_ARG_SPECIFICS[model_name]["model_path"]}')

        # Calculate flags:
        model_diff_flags = self.default_args.copy()
        model_diff_flags.update(
            type(self).MODEL_TO_ARG_SPECIFICS[model_name]['model_flags'])
        model_diff_flags.update(
            {'model_path': type(self).MODEL_TO_ARG_SPECIFICS[model_name]['model_path']})
        if extra_args is not None:
            model_diff_flags.update(extra_args)

        # Create model and diffusion objects
        model, diffusion = create_model_and_diffusion(
            **{key: model_diff_flags[key] for key in model_and_diffusion_defaults().keys()})

        # Load and configure the trained model
        model.load_state_dict(
            dist_util.load_state_dict(
                model_diff_flags['model_path'], map_location="cpu")
        )
        model.to(dist_util.dev())
        if model_diff_flags['use_fp16']:
            model.convert_to_fp16()
        model.eval()

        if to_compile:
            # match = re.match(r'([0-9]+\.[0-9]+)', torch.__version__)
            # if match and float(match.group(1)) >= 2.1:  # PyTorch will not support our model before v2.1
            #     model = torch.compile(model)
            pass  # Still unsupported :')

        return model, diffusion


if __name__ == '__main__':
    # Benchmark this module
    loader = ModelLoader()
    model, diffusion = loader.get_model('64x64')
    print(f'Model loaded:\n{model}')
    print(f'Diffusion object loaded:\n{diffusion}')
