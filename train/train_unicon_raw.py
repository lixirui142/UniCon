#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""
# /home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/readout_guidance/readout_training/data/raw/PascalVOC/VOC2012/JPEGImages/2012_004104.jpg
# /home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/readout_guidance/readout_training/data/pseudo_labels/PascalVOC/depth/2012_004104.jpg
import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, Image
import PIL
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from pipeline.pipeline_stable_diffusion_inpaint_guidance_modi import StableDiffusionInpaintPipeline
# from  utils.peft_utils import get_peft_model_state_dict
from utils.peft_utils import get_peft_model_state_dict, set_adapters_requires_grad
# from torchvision import transforms
import torchvision.transforms.v2 as transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import load_image, make_image_grid
from patch import patch
from utils.util import load_condition_latents, load_lora_weights, tensor_to_vae_latent, parse_schedule
from utils.dataset import process_frames, TrackDataset, LenIterableDatasetWrapper
from utils.gaussian_2d import get_guassian_2d_rand_mask
import torchvision.transforms as T
from utils.gaussian_2d import get_rand_masks, set_smooth_kernel
import pdb
from omegaconf import OmegaConf
from utils.util import load_image_folder
from safetensors import safe_open
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.28.0.dev0")
from accelerate import DistributedDataParallelKwargs

logger = get_logger(__name__, log_level="INFO")


def save_model_card(
    repo_id: str,
    images: list = None,
    base_model: str = None,
    dataset_name: str = None,
    repo_folder: str = None,
):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "diffusers-training",
        "lora",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", nargs='+', help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        nargs='+',
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--y_caption_column",
        type=str,
        nargs='+',
        default=None,
        help="The column of the dataset containing a caption for y images.",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--y_validation_prompt", type=str, default=None, help="A prompt for y that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=32,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=3,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--y_lora",
        type=str,
        default=None,
        help=(
            "Path to the y_lora weights. If not provided, do not use y_lora."
        ),
    )
    parser.add_argument(
        "--x_column",
        type=str,
        nargs='+',
        default=None,
        help=(
            "The column of the dataset containing an y image."
        ),
    )
    parser.add_argument(
        "--no_timestep_align",
        action="store_true"
    )
    parser.add_argument(
        "--clean_cond",
        action="store_true"
    )
    parser.add_argument(
        "--cond_image_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--cond_depth_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--prompt_dropout_prob",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--mask_dropout_prob",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--joint_dropout_prob",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--trigger_word",
        type=str,
        default=""
    )
    parser.add_argument(
        "--rand_transform",
        action="store_true"
    )
    parser.add_argument(
        "--train_y",
        action="store_true"
    )
    parser.add_argument(
        "--ylora_rank",
        type=int,
        default=None
    )
    parser.add_argument(
        "--skip_encoder",
        action="store_true"
    )
    parser.add_argument(
        "--post_joint",
        type=str,
        default="conv"
    )
    parser.add_argument(
        "--add_joint_norm",
        action="store_true"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="imagefolder"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None
    )
    parser.add_argument(
        "--hw_logratio_max",
        type=float,
        default=1.5
    )
    parser.add_argument(
        "--separate_xy_trans",
        action="store_true"
    )
    parser.add_argument(
        "--symmetric",
        action="store_true"
    )
    parser.add_argument(
        "--only_image_loss",
        action="store_true"
    )
    parser.add_argument(
        "--image_column2", type=str, default=None, help="The column of the dataset containing the second cond image."
    )
    parser.add_argument(
        "--trigger_word2", type=str, default=None
    )
    parser.add_argument(
        "--quantize_cond",
        action="store_true"
    )
    parser.add_argument(
        "--dataset_rescale",
        type=float,
        default=None
    )
    parser.add_argument(
        "--cond_guidance_scale",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--resume_from_model", type=str, default=None
    )
    parser.add_argument(
        "--add_conv_bias",
        action="store_true"
    )
    parser.add_argument(
        "--only_y_offset",
        action="store_true"
    )
    # parser.add_argument(
    #     "--clean_cond",
    #     action="store_true"
    # )
    parser.add_argument(
        "--dataset_len",
        type=int,
        default=None
    )


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None and args.dataset_type != "track":
        raise ValueError("Need either a dataset name or a training folder.")

    return args


DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=args.joint_dropout_prob > 0)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    unet_class = UNet2DConditionModel

    unet = unet_class.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    pipeline_class = StableDiffusionInpaintPipeline

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16



    # Freeze the unet parameters before adding adapters
    for param in unet.parameters():
        param.requires_grad_(False)

    
    if args.skip_encoder:
        name_skip = "down_blocks"
    else:
        name_skip = None
    patch.apply_patch(unet, name_skip = name_skip)
    patch.initialize_joint_layers(unet, post = args.post_joint, add_norm = args.add_joint_norm, add_bias = args.add_conv_bias)

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["attn1n.to_k", "attn1n.to_q", "attn1n.to_v", "attn1n.to_out.0"],
    )
    # unet_lora_config = LoraConfig(
    #     r=args.rank,
    #     lora_alpha=args.rank,
    #     init_lora_weights="gaussian",
    #     target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    # )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    all_loras = []

    # Add adapter and make sure the trainable params are in float32.
    if args.symmetric:
        unet.add_adapter(unet_lora_config, adapter_name = "xy_lora")
        all_loras += ["xy_lora"]
    else:
        unet.add_adapter(unet_lora_config, adapter_name = "xy_lora")
        unet.add_adapter(unet_lora_config, adapter_name = "yx_lora")
        all_loras += ["xy_lora", "yx_lora"]
    # unet.add_adapter(unet_lora_config, adapter_name = "y_lora")

    if args.y_lora is not None:
        y_lora_state_dict, y_lora_network_alphas = StableDiffusionPipeline.lora_state_dict(args.y_lora)
        StableDiffusionPipeline.load_lora_into_unet(y_lora_state_dict, y_lora_network_alphas, unet = unet, adapter_name = "y_lora")
        all_loras += ["y_lora"]
    elif args.train_y:
        y_lora_config = LoraConfig(
            r=args.rank if args.ylora_rank is None else args.ylora_rank,
            lora_alpha=args.rank if args.ylora_rank is None else args.ylora_rank,
            init_lora_weights="gaussian",
            target_modules=["attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0", "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0"],
        )
        unet.add_adapter(y_lora_config, adapter_name = "y_lora")
        all_loras += ["y_lora"]

    patch.hack_lora_forward(unet)


    if "y_lora" in all_loras and "yx_lora" in all_loras:
        patch.initialize_joint_lora(unet, "y_lora", "yx_lora")

    if args.resume_from_model is not None:
        state_dict = {}
        print(f"Resuming from ", {args.resume_from_model})
        with safe_open(args.resume_from_model, framework="pt", device=0) as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
        load_ret = unet.load_state_dict(state_dict, strict = False)
        print(load_ret)

    # unet.set_adapters(["y_lora", "xy_lora", "yx_lora"])
    unet.set_adapters(all_loras)

    for param in unet.parameters():
        param.requires_grad_(False)

    # Whether to train y_lora
    if args.train_y:
        trainable_loras = all_loras
    else:
        trainable_loras = [lora_name for lora_name in all_loras if lora_name != "y_lora"]

    set_adapters_requires_grad(unet, True, trainable_loras)
    patch.set_joint_layer_requires_grad(unet, ["xy_lora", "yx_lora"], True)

    def set_joint_mask(model, lora_names, symmetric = False, do_guidance = False):
        joint_attention_mask = [1,0]
        if symmetric:
            y_mask = [1,1]
            x_mask = [1,1]
        else:
            y_mask = [0,1]
            x_mask = [1,0]
        
        if do_guidance:
            joint_attention_mask *= 2
            y_mask *= 2
            x_mask *= 2
        
        # print("Joint attention mask", joint_attention_mask)
        # print("Y mask", y_mask)
        # print("X mask", x_mask)


        patch.set_joint_attention_mask(model, joint_attention_mask)
        for lora in lora_names:
            if lora[0] == "y":
                patch.set_patch_lora_mask(model, lora, y_mask)
            elif lora[0] == "x":
                patch.set_patch_lora_mask(model, lora, x_mask)
        

    set_joint_mask(unet, all_loras, args.symmetric, False)
    # patch.set_joint_attention_mask(unet, [1,0])
    # patch.set_patch_lora_mask(unet, "y_lora", [0,1])
    # patch.set_patch_lora_mask(unet, "yx_lora", [0,1])
    # patch.set_patch_lora_mask(unet, "xy_lora", [1,0])


    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def load_model_hook(models, input_dir):
    
        #load lora
        # lora_path = os.path.join(input_dir, "pytorch_lora_weights.safetensors")
    # if os.path.exists(lora_path):
        for model in models:
            if isinstance(model, unet_class):
                unet = model
        
        for lora_name in all_loras:

            lora_path = os.path.join(input_dir, lora_name, "pytorch_lora_weights.safetensors")
        # yx_lora_path = os.path.join(input_dir, "yx_lora", "pytorch_lora_weights.safetensors")
            load_lora_weights(unet, lora_path, adapter_name=lora_name)
        # load_lora_weights(unet, yx_lora_path, adapter_name="yx_lora")

        save_path = os.path.join(input_dir, "model.pth")
        state_dict = torch.load(save_path, map_location="cpu")

        unet.load_state_dict(state_dict, strict=False)


    def save_model_hook(models, weights, output_dir):
    
        #load lora
        # lora_path = os.path.join(input_dir, "pytorch_lora_weights.safetensors")
    # if os.path.exists(lora_path):
        for model in models:
            if isinstance(model, unet_class):
                unet = model

        state_dict = dict()
        
        for name, params in unet.named_parameters():
            if params.requires_grad:
                state_dict[name] = params
        
        save_path = os.path.join(output_dir, "model.pth")
        torch.save(state_dict, save_path)



    accelerator.register_load_state_pre_hook(load_model_hook)
    accelerator.register_save_state_pre_hook(save_model_hook)

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_type == "track":
        # Downloading and loading a dataset from the hub.
        # dataset = load_dataset(
        #     args.dataset_name,
        #     args.dataset_config_name,
        #     cache_dir=args.cache_dir,
        #     data_dir=args.train_data_dir,
        # )
        assert args.dataset_config is not None, "Dataset type track requires a dataset config file."
        config_path = args.dataset_config

        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)
        # train_dataset, train_dataloader = get_correspondence_loader(config, config["train_file"], True)

        dataset = TrackDataset(**config)
        image_column = args.image_column
        caption_column = args.caption_column
        x_column = args.x_column
    else:
        dataset = load_image_folder(args.train_data_dir, args.cache_dir, args.dataset_type, args.dataset_rescale)
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
        
        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        # dataset = dataset.with_format("torch")

        dataset["train"] = dataset["train"].remove_columns([cn for cn in dataset["train"].column_names if (cn not in [*args.image_column, *args.x_column] and ".png" in cn)])
        print("Column Names", dataset["train"].column_names)
        # column_names = dataset["train"].column_names
        # for column in column_names:
        #     if "image" in column:
        #         dataset = dataset.cast_column(column, Image())


        # 6. Get the column names for input/target.
        # dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
        # if args.image_column is None:
        #     image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        # else:
        #     image_column = args.image_column
        #     if image_column not in column_names:
        #         raise ValueError(
        #             f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
        #         )
        # if args.caption_column is None:
        #     caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        # else:
        #     caption_column = args.caption_column
        #     if caption_column not in column_names:
        #         raise ValueError(
        #             f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
        #         )
        image_column = args.image_column
        caption_column = args.caption_column
        # image_column2 = args.image_column2
        y_caption_column = args.y_caption_column
        x_column = args.x_column
        
        if y_caption_column is None:
            y_caption_column = caption_column
        # pdb.set_trace()
        yimage_text_columns = [(i_column, t_column) for i_column, t_column in zip(image_column, y_caption_column)]
        ximage_text_columns = [(i_column, t_column) for i_column, t_column in zip(x_column, caption_column)]

    if x_column is not None:
        for clm in x_column:
            dataset = dataset.cast_column(clm, Image())
    if image_column is not None:
        for clm in image_column:
            dataset = dataset.cast_column(clm, Image())



    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, x_text_column, y_text_column, is_train=True, trigger_word = "", symmetric = False): 

        if x_text_column not in examples and "json" in examples:
            tmp_dict = dict()
            tmp_dict[x_text_column] = [exp[x_text_column] for exp in examples["json"]]
            if y_text_column is not None:
                tmp_dict[y_text_column] = [exp[y_text_column] for exp in examples["json"]]
            examples = tmp_dict

        captions = []
        # pdb.set_trace()
        for caption in examples[x_text_column]:
            if isinstance(caption, str):
                captions.append(caption)
            else:
                assert False, "Error when tokenize captions"
        
        if y_text_column is not None:
            y_captions = [caption for caption in examples[y_text_column]]
        else:
            y_captions = captions
   
        y_captions = [trigger_word + cpt for cpt in y_captions]
        if symmetric:
            captions = [trigger_word + cpt for cpt in captions]
        captions += y_captions
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            # transforms.ToTensor(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.5], [0.5]),
            # transforms.Normalize([0.5], [0.5]),
        ]
    )



    def random_train_transforms():


        hw_logratio_range = [-np.log(args.hw_logratio_max), np.log(args.hw_logratio_max)]
        total_pixels = args.resolution * args.resolution
        hw_logratio = random.uniform(*hw_logratio_range)
        # hw_logratio = np.log(1.5)
        hw_ratio = np.e ** (hw_logratio)
        # hw_ratio =  0.5
        width = int(np.sqrt(total_pixels / hw_ratio) / 8 + 0.5) * 8
        height = int(total_pixels // width / 8 + 0.5) * 8
        # random_sizes = [(512, 512), (576, 448), (576, 384), (448, 576), (384, 576), (640, 448), (448, 640)]
        # height, width = random.choice(random_sizes)
        hw_ratio = height / width
        train_transforms = transforms.Compose(
            [   
                transforms.RandomResizedCrop(size = (height, width), scale = (0.75, 1.0), ratio = (1 / hw_ratio, 1 / hw_ratio)),
                # transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                # transforms.ToTensor(),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5], [0.5]),
                # transforms.Normalize([0.5], [0.5]),
            ]
        )
        return train_transforms

    if args.rand_transform:
        train_transforms = random_train_transforms()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def preprocess_train(examples, only_caption = False):
        # pdb.set_trace()
        x_image_column, x_text_column = random.choice(ximage_text_columns)
        y_image_column, y_text_column = random.choice(yimage_text_columns)
        
        if not only_caption:
            x_images = []
            y_images = []
            for x_image, y_image in zip(examples[x_image_column], examples[y_image_column]):
                if x_image.width == y_image.width and x_image.height == y_image.height:
                    x_images.append(x_image.convert("RGB"))
                    y_images.append(y_image.convert("RGB"))
            
            pad_num = len(examples[x_image_column]) - len(x_images)
            x_images = x_images + x_images[-1:] * pad_num
            y_images = y_images + y_images[-1:] * pad_num
            # pdb.set_trace()

            

            if args.separate_xy_trans:
                x_images = [train_transforms(x_image) for x_image in x_images]
                y_images = [train_transforms(y_image) for y_image in y_images]
            else:
                xy_images = [train_transforms(x_image, y_image) for x_image, y_image in zip(x_images, y_images)]
                x_images, y_images = zip(*xy_images)

            # examples["y_pixel_values"] = [train_transforms(y_image) for y_image in y_images]
            # examples["x_pixel_values"] = [train_transforms(x_image) for x_image in x_images]
            examples["x_pixel_values"] = x_images
            examples["y_pixel_values"] = y_images

        input_ids = tokenize_captions(examples, x_text_column = x_text_column, y_text_column = y_text_column, trigger_word = args.trigger_word, symmetric = args.symmetric)
        x_len = len(input_ids) // 2
        examples["x_input_ids"] = input_ids[:x_len]
        examples["y_input_ids"] = input_ids[x_len:]
        return examples

    def filter_incomplete_samples(example):

        # Check if both 'image' and 'json' fields are present in the sample
        has_image = True
        for clm in x_column:
            has_image = has_image and example.get(clm) is not None
        for clm in image_column:
            has_image = has_image and example.get(clm) is not None
        has_json = example.get("json") is not None

        # resolution_pass = max(example[x_column[0]].size) > args.resolution
        
        # Return True if both are present, False otherwise
        return has_image and has_json
    
    

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        
        if args.dataset_type == "webdataset":
            dataset["train"] = dataset["train"].filter(filter_incomplete_samples)


        if args.dataset_type == "track":
            train_dataset = dataset.with_transform(preprocess_train)
        else:

            if hasattr(dataset["train"], "with_transform"):
                train_dataset = dataset["train"].with_transform(preprocess_train)
            else:
                train_dataset = dataset["train"].map(preprocess_train, batched=True, batch_size=args.train_batch_size)
        

    def collate_fn(examples):
        # pdb.set_trace()
        x_pixel_values = torch.stack([example["x_pixel_values"] for example in examples])
        x_pixel_values = x_pixel_values.to(memory_format=torch.contiguous_format).float()
        y_pixel_values = torch.stack([example["y_pixel_values"] for example in examples])
        y_pixel_values = y_pixel_values.to(memory_format=torch.contiguous_format).float()
        x_input_ids = torch.stack([example["x_input_ids"] for example in examples])
        y_input_ids = torch.stack([example["y_input_ids"] for example in examples])
        return {"x_pixel_values": x_pixel_values, "x_input_ids": x_input_ids, "y_input_ids": y_input_ids, "y_pixel_values": y_pixel_values}


    if args.dataset_len is not None and args.dataset_type == "webdataset":
        train_dataset = LenIterableDatasetWrapper(dataset = train_dataset, length = args.dataset_len, accelerator = accelerator)

    dataset_length = args.dataset_len if args.dataset_len is not None else len(train_dataset)
    



    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True if args.dataset_type != "webdataset" else False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    print("Dataset length:", dataset_length)
    print("Dataloader length on each process", len(train_dataloader))

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config={k:v if not isinstance(v, list) else " ".join(v) for k,v in vars(args).items()})

    if accelerator.is_main_process:

        rec_txt1 = open('rec_para_unet.txt', 'w')
        rec_txt2 = open('rec_para_unet_train.txt', 'w')
        for name, para in unet.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
        rec_txt1.close()
        rec_txt2.close()

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {dataset_length}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    print(args.resume_from_checkpoint)

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    noise_patch_size = 1
    latent_resolution = args.resolution // 8
    grid_size_rand_scale = 0.5
    grid_size = latent_resolution // noise_patch_size
    # grid_size_range = [int(grid_size_rand_scale * grid_size), int(grid_size / grid_size_rand_scale)]

    
    initial_step = True
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    logger.info(f"VAE scale factor: {vae_scale_factor}")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        # train_dataset.set_epoch(epoch)
        print(f"Epoch {epoch}, Step {global_step}")
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                if args.rand_transform:
                    train_transforms = random_train_transforms()
                # pdb.set_trace()

                xy_pixel_values = torch.cat([batch["x_pixel_values"], batch["y_pixel_values"]], dim=0)
                xy_input_ids = torch.cat([batch["x_input_ids"], batch["y_input_ids"]], dim=0)

                bsz = xy_pixel_values.shape[0]

                if args.no_timestep_align:
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
                else:
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz // 2,))
                    timesteps = torch.cat([timesteps, timesteps], dim=0)
                # timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)

                if args.clean_cond:
                    timesteps[bsz // 2:] = 0

                if args.quantize_cond:
                    # quanitization levels: 2n. n in [0.5, 127.5], t in [0, T). n = 127.5 - (127 / T) t. 
                    y_timesteps = timesteps[bsz // 2:].to(xy_pixel_values)
                    y_ns = 127.5 - y_timesteps * (127. / noise_scheduler.config.num_train_timesteps)
                    y_ns = y_ns[:,None,None,None]

                    xy_pixel_values[bsz // 2:] = (xy_pixel_values[bsz // 2:] * y_ns).round() / y_ns.round()
                
                timesteps = timesteps.long()

                # pixel_values = batch["y_pixel_values"]
                # input_ids = batch["input_ids"]


                # Convert images to latent space
                latents = vae.encode(xy_pixel_values.to(dtype=weight_dtype)).latent_dist.sample()
                # latents = vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                timesteps = timesteps.to(latents.device)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    if args.only_y_offset:
                        noise[bsz // 2:] += args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                        )[bsz // 2:]
                    else:
                        noise += args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                        )



                height, width = xy_pixel_values.shape[-2], xy_pixel_values.shape[-1]

                # Sample a random timestep for each image


                # print(timesteps)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if args.quantize_cond:
                    noisy_latents[bsz // 2:] = latents[bsz // 2:]

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(xy_input_ids, return_dict=False)[0]


                if args.prompt_dropout_prob is not None:
                    random_p = torch.rand(
                        bsz // 2, device=latents.device)
                    # Sample masks for the edit prompts.
                    x_prompt_mask = random_p < 2 * args.prompt_dropout_prob
                    y_prompt_mask = torch.logical_and(random_p > args.prompt_dropout_prob, random_p < 3 * args.prompt_dropout_prob)
                    prompt_mask = torch.cat([x_prompt_mask, y_prompt_mask], dim = 0)
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = torch.zeros_like(encoder_hidden_states)
                    encoder_hidden_states = torch.where(
                        prompt_mask, null_conditioning, encoder_hidden_states)
                
                if args.joint_dropout_prob is not None:
                    if random.random() < args.joint_dropout_prob:
                        patch.set_joint_attention(unwrap_model(unet), enable = False)
                    else:
                        patch.set_joint_attention(unwrap_model(unet), enable = True)


                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")


                input_data = noisy_latents

                # Predict the noise residual and compute loss
                model_pred = unet(input_data, timesteps, encoder_hidden_states, return_dict=False)[0]
                
                if args.only_image_loss or args.clean_cond:
                    model_pred = model_pred[:bsz // 2]
                    target = target[:bsz // 2]
                    timesteps = timesteps[:bsz // 2]



                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                if args.validation_prompt is not None and (global_step % args.validation_steps == 0 or initial_step):
                    initial_step = False
                    logger.info(
                        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                        f" {args.validation_prompt}."
                    )

                    raw_unet = unwrap_model(unet)
                    set_joint_mask(raw_unet, all_loras, args.symmetric, True)
                    # patch.set_joint_attention_mask(raw_unet, [1,0,1,0])
                    # patch.set_patch_lora_mask(raw_unet, "y_lora", [0,1,0,1])
                    # patch.set_patch_lora_mask(raw_unet, "yx_lora", [0,1,0,1])
                    # patch.set_patch_lora_mask(raw_unet, "xy_lora", [1,0,1,0])

                    # create pipeline
                    pipeline = pipeline_class.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=unwrap_model(unet),
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                    )
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                    # run inference
                    generator = torch.Generator(device=accelerator.device)
                    if args.seed is not None:
                        generator = generator.manual_seed(args.seed)
                    images = []
                    if torch.backends.mps.is_available():
                        autocast_ctx = nullcontext()
                    else:
                        autocast_ctx = torch.autocast(accelerator.device.type)

                    batch_size = 4
                    init_image = PIL.Image.open(args.cond_image_path).convert("RGB")
                    init_depth = PIL.Image.open(args.cond_depth_path).convert("RGB")
                    init_images = [init_image, init_depth]
                    init_images = process_frames(init_images, args.resolution, args.resolution)
                    init_images = [init_images[0] for i in range(batch_size // 2)] + [init_images[1] for i in range(batch_size // 2)]


                    height, width = args.resolution, args.resolution
                    grid_size = max(height, width)
                    rand_masks = get_rand_masks(batch_size, grid_size, noise_patch_size = noise_patch_size, thresh = 0)
                    rand_masks = rand_masks[...,:height, :width]

                    # rand_mask[batch_size // 2:] = 1 - rand_mask[:batch_size // 2]
                    rand_masks[:batch_size // 2] = 1
                    rand_masks[batch_size // 2:] = 0
                    mask_images = [T.ToPILImage()(mask) for mask in rand_masks]
                    # latent_height, latent_width = height, width
                    grid_size = max(height, width)

                    prompt = args.validation_prompt
                    y_prompt = args.y_validation_prompt if args.y_validation_prompt is not None else prompt

                    if args.symmetric:
                        prompts = [args.trigger_word + prompt if i < (batch_size // 2) else args.trigger_word + y_prompt for i in range(batch_size)]
                    else:
                        prompts = [prompt if i < (batch_size // 2) else args.trigger_word + y_prompt for i in range(batch_size)]

                    cond_guidance_scale = args.cond_guidance_scale
                    sample_schedule = parse_schedule("[0-49,49-49,1,0]")

                    with autocast_ctx:
                        for _ in range(args.num_validation_images):
                            images += pipeline(
                                    prompts,
                                    image = init_images,
                                    mask_image = mask_images,
                                    height = args.resolution,
                                    width = args.resolution,
                                    # num_inference_steps=30, 
                                    generator=generator,
                                    y_advance = 1.0 if args.no_timestep_align else None,
                                    cond_guidance_scale = cond_guidance_scale,
                                    sample_schedule = sample_schedule
                                ).images

                    
                    # mask_images_upscale = process_frames(mask_images, height, width)
                    image_grid = make_image_grid([*images], rows=1, cols=batch_size)
                    set_joint_mask(raw_unet, all_loras, args.symmetric, False)
                    patch.set_joint_attention(raw_unet, enable = True)
                    # patch.set_joint_attention_mask(raw_unet, [1,0])
                    # patch.set_patch_lora_mask(raw_unet, "y_lora", [0,1])
                    # patch.set_patch_lora_mask(raw_unet, "yx_lora", [0,1])
                    # patch.set_patch_lora_mask(raw_unet, "xy_lora", [1,0])

                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = np.stack([image_grid])
                            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                        if tracker.name == "wandb":
                            tracker.log(
                                {
                                    "validation": [
                                        wandb.Image(image_grid, caption=f"{prompt}, {args.trigger_word + y_prompt}")
                                    ]
                                }
                            )

                    del pipeline
                    torch.cuda.empty_cache()


                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        
                        unwrapped_unet = unwrap_model(unet)
                        for lora_name in trainable_loras:
                            cur_save_path = os.path.join(save_path, f"{lora_name}")
                            unet_lora_state_dict = convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(unwrapped_unet, adapter_name=lora_name)
                            )


                            StableDiffusionPipeline.save_lora_weights(
                                save_directory=cur_save_path,
                                unet_lora_layers=unet_lora_state_dict,
                                safe_serialization=True,
                            )

                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        # if accelerator.is_main_process:
            

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        unwrapped_unet = unwrap_model(unet)
        for lora_name in trainable_loras:
            save_dir = os.path.join(args.output_dir, f"{lora_name}")
            unet_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(unwrapped_unet, adapter_name=lora_name)
            )

            StableDiffusionPipeline.save_lora_weights(
                save_directory=save_dir,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )
        
        state_dict = dict()
        for name, params in unwrapped_unet.named_parameters():
            if params.requires_grad:
                state_dict[name] = params
        
        save_path = os.path.join(args.output_dir, "model.pth")
        torch.save(state_dict, save_path)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                dataset_name=args.dataset_name,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        # Final inference
        # Load previous pipeline
        # if args.validation_prompt is not None:
        #     pipeline = pipeline_class.from_pretrained(
        #         args.pretrained_model_name_or_path,
        #         revision=args.revision,
        #         variant=args.variant,
        #         torch_dtype=weight_dtype,
        #     )
        #     pipeline = pipeline.to(accelerator.device)






        #     # patch.apply_patch(pipeline.unet)
        #     # patch.initialize_joint_layers(pipeline.unet)


        #     # if args.y_lora is not None:
        #     #     # y_lora_state_dict, y_lora_network_alphas = StableDiffusionPipeline.lora_state_dict(args.y_lora)
        #     #     # StableDiffusionPipeline.load_lora_into_unet(y_lora_state_dict, y_lora_network_alphas, unet = pipeline.unet, adapter_name = "y_lora")
        #     #     pipeline.load_lora_weights(args.y_lora, adapter_name="y_lora")


        #     # load attention processors
        #     # for lora_name in ["xy_lora", "yx_lora"]:
        #     #     save_dir = os.path.join(args.output_dir, f"{lora_name}")
        #     #     # lora_state_dict, lora_network_alphas = StableDiffusionPipeline.lora_state_dict(save_dir)
        #     #     # StableDiffusionPipeline.load_lora_into_unet(lora_state_dict, lora_network_alphas, unet = pipeline.unet, adapter_name = lora_name)
        #     #     pipeline.load_lora_weights(save_dir, adapter_name=lora_name)
            
            
        #     # pipeline.set_adapters(["y_lora", "xy_lora", "yx_lora"])
        #     # patch.hack_lora_forward(pipeline.unet)
        #     # patch.set_patch_lora_mask(pipeline.unet, "y_lora", [0,1,0,1])
        #     # patch.set_patch_lora_mask(pipeline.unet, "yx_lora", [0,1,0,1])
        #     # patch.set_patch_lora_mask(pipeline.unet, "xy_lora", [1,0,1,0])

        #     # save_path = os.path.join(args.output_path, "model.pth")
        #     # state_dict = torch.load(save_path, map_location="cpu")

        #     # pipeline.unet.load_state_dict(state_dict, strict=False)

        #     pipeline = pipeline.to(accelerator.device)


        #     # run inference
        #     generator = torch.Generator(device=accelerator.device)
        #     if args.seed is not None:
        #         generator = generator.manual_seed(args.seed)
        #     images = []
        #     if torch.backends.mps.is_available():
        #         autocast_ctx = nullcontext()
        #     else:
        #         autocast_ctx = torch.autocast(accelerator.device.type)

        #     with autocast_ctx:
        #         for _ in range(args.num_validation_images):
        #             images += pipeline(args.validation_prompt, num_inference_steps=30, num_images_per_prompt = 2, generator=generator).images

        #     for tracker in accelerator.trackers:
        #         if len(images) != 0:
        #             if tracker.name == "tensorboard":
        #                 np_images = np.stack([np.asarray(img) for img in images])
        #                 tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
        #             if tracker.name == "wandb":
        #                 tracker.log(
        #                     {
        #                         "test": [
        #                             wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
        #                             for i, image in enumerate(images)
        #                         ]
        #                     }
        #                 )

    accelerator.end_training()


if __name__ == "__main__":
    main()