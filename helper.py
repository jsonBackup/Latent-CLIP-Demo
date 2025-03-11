import os
from dataclasses import dataclass, field
from typing import Optional, List
import blobfile as bf
import logging
import torch
import matplotlib.pyplot as plt
from PIL import Image
from huggingface_hub import hf_hub_download
import shutil

import sys

sys.path.insert(0, "./Latent-ReNO/")

from arguments import parse_args
from models import get_model
from rewards import get_reward_losses, get_latent_reward_losses
from training import LatentNoiseTrainer, get_optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16


@dataclass
class Args:
    softpromptdecay = 0.03
    disable_trainable_latents = False
    cache_dir: str = "./cache"
    save_dir: str = "./outputs/"
    model: str = "sdxl-turbo"
    lr: float = 5.0
    n_iters: int = 50
    n_inference_steps: int = 1
    optim: str = "sgd"
    nesterov: bool = True
    grad_clip: float = 0.1
    seed: int = 0
    enable_hps: bool = True
    hps_weighting: float = 10.0
    enable_imagereward: bool = True
    imagereward_weighting: float = 1.0
    enable_only_latents: bool = True
    enable_trainable_prompt: bool = False
    enable_clip_clf: bool = True
    maximize: bool = False
    enable_clip_text: bool = False
    enable_clip_image: bool = False
    enable_clip: bool = False
    clip_model: Optional[str] = None
    clip_weighting: float = 0.01
    latent_guidance_prompt: Optional[str] = None
    enable_pickscore: bool = True
    pickscore_weighting: float = -0.1
    pickscore_weighting: float = -0.1
    enable_aesthetic: bool = False
    aesthetic_weighting: float = -0.1
    enable_md_aesthetic: bool = False
    md_aesthetic_weighting: float = -0.1
    enable_sh_aesthetic: bool = False
    sh_aesthetic_weighting: float = 0.1
    enable_pgen: bool = False
    pgen_weighting: float = 1.0
    enable_nsfw: bool = False
    nsfw_weighting: float = 1.0
    enable_reg: bool = True
    reg_weight: float = 0.01
    task: str = "single"
    prompt: str = "A green elephant and a red mouse"
    negative_prompt: Optional[str] = None
    benchmark_reward: str = "total"
    save_all_images: bool = True
    save_gif: bool = True
    no_optim: bool = False
    imageselect: bool = False
    memsave: bool = False
    device: str = "cuda"
    device_id: Optional[int] = None


def get_sd_model(args):
    return get_model(args, args.model, dtype, device, args.cache_dir, args.memsave)


def get_latent_noise_trainer(args, sd_model):
    bf.makedirs(f"{args.save_dir}/logs/{args.task}")
    settings = (
        f"{args.model}_max-{args.maximize}_{args.latent_guidance_prompt}_{args.optim}"
    )

    logger = logging.getLogger()
    if not logger.hasHandlers():
        file_stream = open(f"{args.save_dir}/logs/{args.task}/{settings}.txt", "w")
        handler = logging.StreamHandler(file_stream)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel("INFO")
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)

    if args.device_id is not None:
        logging.info(f"Using CUDA device {args.device_id}")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICE"] = args.device_id

    if "latent" in args.clip_model.lower():
        reward_losses = get_latent_reward_losses(args, dtype, device, args.cache_dir)
    else:
        reward_losses = get_reward_losses(args, dtype, device, args.cache_dir)

    trainer = LatentNoiseTrainer(
        reward_losses=reward_losses,
        model=sd_model,
        n_iters=args.n_iters,
        n_inference_steps=args.n_inference_steps,
        seed=args.seed,
        save_all_images=args.save_all_images,
        save_gif=args.save_gif,
        device=device,
        no_optim=args.no_optim,
        regularize=args.enable_reg,
        regularization_weight=args.reg_weight,
        grad_clip=args.grad_clip,
        log_metrics=args.task == "single" or not args.no_optim,
        imageselect=args.imageselect,
        optim=args.optim,
    )

    return trainer, settings


def generate_and_optimize(args, trainer, sd_model, settings):
    height = sd_model.unet.config.sample_size * sd_model.vae_scale_factor
    width = sd_model.unet.config.sample_size * sd_model.vae_scale_factor
    shape = (
        1,
        sd_model.unet.in_channels,
        height // sd_model.vae_scale_factor,
        width // sd_model.vae_scale_factor,
    )

    init_latents = torch.randn(shape, device=device, dtype=dtype)
    init_prompt = torch.randn([1, 77, 2048], device=device, dtype=dtype)
    init_add_text = torch.randn([1, 1280], device=device, dtype=dtype)

    latents = torch.nn.Parameter(init_latents, requires_grad=True)

    optimizer = get_optimizer(args.optim, latents, args.lr, args.nesterov)
    save_dir = f"{args.save_dir}/{args.task}/{settings}/{args.prompt[:100]}"
    os.makedirs(f"{save_dir}", exist_ok=True)
    best_image, total_best_rewards, initial_image_pil, total_init_rewards = (
        trainer.train(
            latents,
            args.prompt,
            optimizer,
            save_dir,
            negative_prompt=args.negative_prompt,
        )
    )
    best_image.save(f"{save_dir}/best_image.png")

    return save_dir


def plot_images(save_dir, num_inference_steps=50, only_best=False):
    if only_best:
        best_image_path = os.path.join(save_dir, "best_image.png")
        if os.path.exists(best_image_path):
            image = Image.open(best_image_path)
            plt.figure(figsize=(2, 2))
            plt.imshow(image)
            plt.axis("off")
            plt.title("Best Image")
            plt.show()
        else:
            print("'best_image.png' not found in the directory.")
        return

    # Handling multiple image plotting when only_best is False
    valid_filenames = {f"{i:02d}.png" for i in range(num_inference_steps)}
    image_files = sorted([f for f in os.listdir(save_dir) if f in valid_filenames])

    num_images = len(image_files)
    if num_images == 0:
        print("No valid images found in the directory.")
        return

    cols = min(10, num_images)  # Max 10 columns for better readability
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if num_images > 1 else [axes]

    for ax, img_name in zip(axes, image_files):
        img_path = os.path.join(save_dir, img_name)
        image = Image.open(img_path)

        ax.imshow(image)
        ax.axis("off")
        ax.set_title(img_name.split(".")[0])

    for ax in axes[num_images:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def download_and_rename_model(repo_id, save_as, target_dir, filename="epoch_34.pt"):
    target_path = os.path.join(target_dir, save_as)

    if os.path.exists(target_path):
        print(f"✅ {save_as} already exists. Skipping download.")
        return

    os.makedirs(target_dir, exist_ok=True)

    downloaded_file = hf_hub_download(repo_id=repo_id, filename=filename)

    shutil.copy(downloaded_file, target_path)

    print(f"✅ {filename} downloaded from {repo_id} and saved as {target_path}")
