import copy
import functools
import os
from typing import List, Tuple

import blobfile as bf
import torch as th
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import matplotlib.pyplot as plt

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

from utils.noiser import TimestepUniformNoiser
from utils.denoiser import ModelTimestepUniformDenoiser
from config.configuration import CATEGORY_TO_NOISE_TIMESTEPS

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        val_data=None,
        target: str=None,
        save_opt: bool=True,
        save_ema: bool=True,
        steps_limit: int=0
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.val_data: DataLoader = val_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.target = target
        self.save_opt = save_opt
        self.save_ema = save_ema
        self.steps_limit = steps_limit

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _eval(self):
        """
        Performs evaluation by computing the loss and reconstruction quality for the given target class.

        Args:
            target (str): The target class for evaluation.
            dump_path (str): The path where the evaluation results will be saved.
        """
        all_losses = []
        logger.log(f"Evaluating model at step {self.step + self.resume_step}")

        with th.no_grad():
            for i, (batch, cond) in enumerate(self.val_data):
                for loss in self.calculate_losses(batch, cond, should_log=False):
                    all_losses.append(loss.cpu())
                
        mean_loss = th.tensor(all_losses).mean().item()
        logger.log(f"\tMean loss on validation data: {mean_loss:.6f}")

        with th.no_grad():
            recon_dump_path = os.path.join(logger.get_dir(), "validation_imgs", f"recon_imgs_step_{self.step + self.resume_step}.jpg")
            self._log_batch_recon(batch=th.cat(tensors=[data[0] for data in self.val_data], dim=0).to(dist_util.dev()),
                                  dump_path=recon_dump_path,
                                  target=self.target)

    def _log_batch_recon(self, batch: th.Tensor | List[th.Tensor], dump_path: str, target: str):
        """
        Applies noise to the batch of images, denoises them using the model,
        and logs the original and reconstructed images.

        Args:
            batch (torch.Tensor or List[torch.Tensor]): A batch of images to be reconstructed.
            dump_path (str): The path where the reconstructed images will be saved.
            target (str): The target class for which the images are reconstructed.
        """
        assert target is not None, "`target` evaluation class was None. Expected: str.\nDid you forget to pass the --target argument?"
        noiser = TimestepUniformNoiser(self.diffusion)
        denoiser = ModelTimestepUniformDenoiser(self.model, self.diffusion)
        timesteps = CATEGORY_TO_NOISE_TIMESTEPS[target]

        logger.log(f'evaluating reconstruction for target class {target}')
        processed_imgs = []

        for image in batch:
            noised_image = noiser.apply_noise(image.unsqueeze(0), timesteps)
            reconstructed_image = denoiser.denoise(noised_image, timesteps, show_progress=False).squeeze(0)
            processed_imgs.append((((image.cpu() / 2) + 0.5).clip(0, 1).permute(1, 2, 0),
                                   ((reconstructed_image.cpu() / 2) + 0.5).clip(0, 1).permute(1, 2, 0)))

        logger.log(f'dumping result to {dump_path}')
        self.plot_images(processed_imgs, dump_path)

    @staticmethod
    def plot_images(image_list: List[Tuple], dump_path: str) -> None:
        """
        Plots original and reconstructed images from a list of image tuples.

        Args:
            image_list (List[Tuple]): A list of tuples containing original and reconstructed images.
            dump_path (str): The path where the figure will be saved.

        Returns:
            None
        """
        num_images = len(image_list)
        fig, axs = plt.subplots(num_images, 2, figsize=(10, 10))

        for i, (original_img, reconstructed_img) in enumerate(image_list):
            axs[i, 0].imshow(original_img)
            axs[i, 0].axis('off')
            axs[i, 0].set_title('Original Image')

            axs[i, 1].imshow(reconstructed_img)
            axs[i, 1].axis('off')
            axs[i, 1].set_title('Reconstructed Image')

        plt.tight_layout()
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        plt.savefig(dump_path)

    def run_loop(self):
        while (
            (not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps)
            and
            (self.steps_limit <= 0
             or self.step + self.resume_step < self.steps_limit)
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                if self.val_data is not None:
                    self._eval()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def calculate_losses(self, batch, cond, should_log: bool=True):
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()

            if should_log:
                log_loss_dict(
                    self.diffusion, t, {k: v * weights for k, v in losses.items()}
                )

            yield loss

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for loss in self.calculate_losses(batch, cond):
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)  # Save model

        if self.save_ema:
            for rate, params in zip(self.ema_rate, self.ema_params):
                save_checkpoint(rate, params)  # Save EMA params

        if self.save_opt and dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)  # Save optimizer params

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
