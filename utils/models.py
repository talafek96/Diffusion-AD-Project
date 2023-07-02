import os
from pathlib import Path
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

    def get_model(self,
                  model_name: str,
                  extra_args: dict=None,
                  to_compile: bool=True,
                  path: str=None) -> Tuple[UNetModel, GaussianDiffusion]:
        """
        Creates and loads a trained UNetModel and GaussianDiffusion objects based on the specified model name.

        Args:
            model_name (str): The name of the model to load.
            extra_args (dict, optional): Additional arguments to be passed to create the model and diffusion objects. Defaults to None.
            to_compile (bool, optional): Flag indicating whether to compile the model. Defaults to True. (if supported)
            path (str, optional): The path to the trained model .pt file. Defaults to None.

        Returns:
            Tuple[UNetModel, GaussianDiffusion]: A tuple containing the loaded UNetModel and GaussianDiffusion objects.
        """
        model_name = model_name.strip().lower()  # Normalize string to convention

        if model_name not in type(self).MODEL_TO_ARG_SPECIFICS.keys():
            raise RuntimeError(
                f'Model name "{model_name}" not supported.\nChoose one of: {list(type(self).MODEL_TO_ARG_SPECIFICS.keys())}')
        model_path = path if path is not None and path != '' else type(self).MODEL_TO_ARG_SPECIFICS[model_name]['model_path']
        if not os.path.exists(model_path):
            raise RuntimeError(
                f'The trained model .pt file was not found in:\n{model_path}')

        # Calculate flags:
        model_diff_flags = self.default_args.copy()
        model_diff_flags.update(
            type(self).MODEL_TO_ARG_SPECIFICS[model_name]['model_flags'])
        model_diff_flags.update(
            {'model_path': model_path})
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
    
    def get_model_name(self, model_name: str, path: str=None) -> str:
        """
        Returns the model name extracted from a given path if provided, otherwise returns the original model name.

        Args:
            model_name (str): The original model name.
            path (str, optional): The path from which to extract the model name. Defaults to None.

        Returns:
            str: The extracted model name or the original model name if path is not provided.
        """
        if path is None:
            if model_name not in type(self).MODEL_TO_ARG_SPECIFICS:
                raise RuntimeError(f'Input model_name is {model_name} and not found inside the ModelLoader class configuration dictionary!')

            return model_name

        path_obj = Path(path)
        return path_obj.name


if __name__ == '__main__':
    # Benchmark this module
    loader = ModelLoader()
    model, diffusion = loader.get_model('64x64')
    print(f'Model loaded:\n{model}')
    print(f'Diffusion object loaded:\n{diffusion}')
