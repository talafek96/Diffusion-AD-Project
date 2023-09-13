"""
Train a diffusion model on images.
"""

import argparse
import os
import tempfile
import torch
from pathlib import Path

from guided_diffusion import dist_util, logger
from guided_diffusion.unet import UNetModel
from guided_diffusion.gaussian_diffusion import GaussianDiffusion
from guided_diffusion.resample import UniformSampler, LossSecondMomentResampler
from guided_diffusion.image_datasets import load_data, get_dataloader
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from utils.models import ModelLoader
from config.configuration import DEFAULT_ROOT_OUTPUT_DIR


DEFAULT_LOG_BASE = DEFAULT_ROOT_OUTPUT_DIR
os.makedirs(DEFAULT_LOG_BASE, exist_ok=True)
tempfile.tempdir = Path(DEFAULT_LOG_BASE)
BASE_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'extern', 'mvtec'))


def dl_wrapper(dl):
    while True:
        yield from dl


def create_argparser():
    defaults = dict(
        data_dir="",
        log_dir=None,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        few_shot_count=10,
        val_size=2,
        target=None,
        save_opt=True,
        save_ema=True,
        steps_limit=0
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def _handle_few_shot_training(args: argparse.Namespace,
                              model: UNetModel,
                              diffusion: GaussianDiffusion,
                              schedule_sampler: UniformSampler | LossSecondMomentResampler):
    logger.log("creating train and validation data loaders...")
    train_dl, val_dl = get_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        few_shot_count=args.few_shot_count,
        validation_size=args.val_size
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=dl_wrapper(train_dl),
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        val_data=val_dl,
        target=args.target,
        save_opt=args.save_opt,
        save_ema=args.save_ema,
        steps_limit=args.steps_limit
    ).run_loop()


def _handle_training(args: argparse.Namespace,
                     model: UNetModel,
                     diffusion: GaussianDiffusion,
                     schedule_sampler: UniformSampler | LossSecondMomentResampler):
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        few_shot_count=args.few_shot_count,
        validation_size=args.val_size
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        save_opt=args.save_opt,
        save_ema=args.save_ema,
        steps_limit=args.steps_limit
    ).run_loop()


def main():
    args = create_argparser().parse_args()
    assert 'few_shot_count' not in args.__dict__ or 'target' in args.__dict__, "ERROR: Please provide a target for few-shot training"
    if 'target' in args.__dict__:
        tempfile.tempdir /= args.target

    if (type(args.few_shot_count) not in [int, None]) or (args.few_shot_count is int and args.few_shot_count <= 0):
        print("few_shot_count argument has to be a positive integer if stated.")
        exit(1)

    torch.manual_seed(42)  # Set manual seed for reproducibility
    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    logger.log("creating model and diffusion...")

    model_loader = ModelLoader()
    model, diffusion = model_loader.get_model(
        '256x256_uncond', extra_args={'use_fp16': args.use_fp16})
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion)

    if args.few_shot_count:
        args.data_dir = os.path.join(
            BASE_DATA_DIR, args.target, "train", "good")
        _handle_few_shot_training(args,
                                  model,
                                  diffusion,
                                  schedule_sampler)
    else:
        _handle_training(args,
                         model,
                         diffusion,
                         schedule_sampler)


if __name__ == "__main__":
    main()
