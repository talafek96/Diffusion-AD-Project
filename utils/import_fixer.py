import sys
import os


EXTERN_GUIDED_DIFFUSION_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'extern', 'guided_diffusion'))
sys.path.append(EXTERN_GUIDED_DIFFUSION_PATH)
