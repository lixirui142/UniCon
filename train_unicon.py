#!/usr/bin/env python
# coding=utf-8
# Adapted from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py

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

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
import pdb

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import Image
from packaging import version
from peft import LoraConfig
import torchvision.transforms.v2 as transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from patch import patch
from utils.utils import set_unicon_config_train
from utils.load_utils import load_image_folder
from utils.peft_utils import get_peft_model_state_dict, set_adapters_requires_grad
from safetensors import safe_open

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.28.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="UniCon training script.")
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
        "--x_image_column",
        type=str,
        nargs='+',
        default=None,
        help=(
            "The column of the dataset containing x image."
        ),
    )
    parser.add_argument(
        "--y_image_column",
        type=str,
        default="image",
        nargs='+',
        help="The column of the dataset containing y image."
    )
    parser.add_argument(
        "--x_caption_column",
        type=str,
        default="text",
        nargs='+',
        help="The column of the dataset containing x caption.",
    )
    parser.add_argument(
        "--y_caption_column",
        type=str,
        nargs='+',
        default=None,
        help="The column of the dataset containing y caption. If not set, use the same as x_caption_column.",
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
        default="unicon",
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
        default=5.0,
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
    # UniCon specific arguments
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="json"
    )
    parser.add_argument(
        "--prompt_dropout_prob",
        type=float,
        default=0.1,
        help=("The prob of dropping prompt when training."),
    )
    parser.add_argument(
        "--joint_dropout_prob",
        type=float,
        default=0.0,
        help=("The prob of dropping joint modules when training."),
    )
    parser.add_argument(
        "--trigger_word",
        type=str,
        default="",
        help=("Trigger word to prepend to the y captions."),
    )
    parser.add_argument(
        "--train_y_lora",
        action="store_true",
        help=("Whether to add and train y lora."),
    )
    parser.add_argument(
        "--ylora_rank",
        type=int,
        default=None,
        help=("The dimension of the y LoRA update matrices. If not set, use the same as args.rank."),
    )
    parser.add_argument(
        "--skip_encoder",
        action="store_true",
        help=("Whether to skip the encoder when adding joint modules."),
    )
    parser.add_argument(
        "--post_joint",
        type=str,
        default="conv",
        help=("The post joint module type. Choose between 'conv' or 'conv_fuse'."),
    )
    parser.add_argument(
        "--rand_transform",
        action="store_true",
        help=("Whether to use random hw ratio for training."),
    )
    parser.add_argument(
        "--separate_xy_trans",
        action="store_true",
        help=("Whether to use separate transforms for x and y images."),
    )
    parser.add_argument(
        "--hw_logratio_max",
        type=float,
        default=1.5,
        help=(
            "The max hw logratio for random hw ratio. The ratio is sampled from [1/hw_logratio_max, hw_logratio_max]."
        ),
    )
    parser.add_argument(
        "--resume_from_model", type=str, default=None,
        help=(
            "The path to the model to resume from. The model should be a state dict of the unet in safetensor. "
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank


    return args


def main():
    args = parse_args()

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

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # Freeze the unet parameters before adding adapters
    for param in unet.parameters():
        param.requires_grad_(False)
    
    # Insert and initialize UniCon modules    
    patch.apply_patch(unet, name_skip = "down_blocks" if args.skip_encoder else None, train = True)

    patch.initialize_joint_layers(unet, post = args.post_joint)

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["attn1n.to_k", "attn1n.to_q", "attn1n.to_v", "attn1n.to_out.0"],
    )

    all_loras = []

    # Add adapter and make sure the trainable params are in float32.

    unet.add_adapter(unet_lora_config, adapter_name = "xy_lora")
    unet.add_adapter(unet_lora_config, adapter_name = "yx_lora")
    all_loras += ["xy_lora", "yx_lora"]

    if args.train_y_lora:
        y_lora_config = LoraConfig(
            r=args.rank if args.ylora_rank is None else args.ylora_rank,
            lora_alpha=args.rank if args.ylora_rank is None else args.ylora_rank,
            init_lora_weights="gaussian",
            target_modules=["attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0", "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0"],
        )
        unet.add_adapter(y_lora_config, adapter_name = "y_lora")
        all_loras += ["y_lora"]

    patch.hack_lora_forward(unet)

    if args.resume_from_model is not None:
        state_dict = {}
        logger.info(f"Resuming from ", {args.resume_from_model})
        with safe_open(args.resume_from_model, framework="pt", device=0) as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
        unet.load_state_dict(state_dict, strict = False)

    unet.set_adapters(all_loras)

    for param in unet.parameters():
        param.requires_grad_(False)

    trainable_loras = all_loras
    set_adapters_requires_grad(unet, True, trainable_loras)
    patch.set_joint_layer_requires_grad(unet, ["xy_lora", "yx_lora"], True) 


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
    
        for model in models:
            if isinstance(model, unet_class):
                unet = model        

        save_path = os.path.join(input_dir, "model.pth")
        state_dict = torch.load(save_path, map_location="cpu")

        unet.load_state_dict(state_dict, strict=False)


    def save_model_hook(models, weights, output_dir):
    
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

    trainable_params = filter(lambda p: p.requires_grad, unet.parameters())

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
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Load training dataset    
    dataset = load_image_folder(args.train_data_dir, args.cache_dir, args.dataset_type)


    y_image_column = args.y_image_column
    x_caption_column = args.x_caption_column
    y_caption_column = args.y_caption_column
    x_image_column = args.x_image_column
        
    if y_caption_column is None:
       y_caption_column = x_caption_column
    
    train_columns = [*y_image_column, *y_caption_column, *x_image_column, *x_caption_column]

    dataset["train"] = dataset["train"].remove_columns([cn for cn in dataset["train"].column_names if (cn not in train_columns)])

    yimage_text_columns = [(i_column, t_column) for i_column, t_column in zip(y_image_column, y_caption_column)]
    ximage_text_columns = [(i_column, t_column) for i_column, t_column in zip(x_image_column, x_caption_column)]

    if x_image_column is not None:
        for clm in x_image_column:
            dataset = dataset.cast_column(clm, Image())
    if y_image_column is not None:
        for clm in y_image_column:
            dataset = dataset.cast_column(clm, Image())

    # Preprocessing the datasets. (fixed or random transforms)
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.5], [0.5]),
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
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        return train_transforms

    if args.rand_transform:
        train_transforms = random_train_transforms()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, x_text_column, y_text_column, trigger_word = ""): 

        if x_text_column not in examples and "json" in examples:
            tmp_dict = dict()
            tmp_dict[x_text_column] = [exp[x_text_column] for exp in examples["json"]]
            if y_text_column is not None:
                tmp_dict[y_text_column] = [exp[y_text_column] for exp in examples["json"]]
            examples = tmp_dict
        
        captions = [caption for caption in examples[x_text_column]]
        
        if y_text_column is not None:
            y_captions = [caption for caption in examples[y_text_column]]
        else:
            y_captions = captions
   
        y_captions = [trigger_word + cpt for cpt in y_captions]

        captions += y_captions
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def preprocess_train(examples):
        # For possible multiple captions, we randomly select one.
        x_image_column, x_text_column = random.choice(ximage_text_columns)
        y_image_column, y_text_column = random.choice(yimage_text_columns)
        
        x_images = []
        y_images = []
        for x_image, y_image in zip(examples[x_image_column], examples[y_image_column]):
            # check x y sizes match
            assert args.separate_xy_trans or (x_image.width == y_image.width and x_image.height == y_image.height)
            
            x_images.append(x_image.convert("RGB"))
            y_images.append(y_image.convert("RGB"))
        
        # pad_num = len(examples[x_image_column]) - len(x_images)
        # x_images = x_images + x_images[-1:] * pad_num
        # y_images = y_images + y_images[-1:] * pad_num

        if args.separate_xy_trans:
            x_images = [train_transforms(x_image) for x_image in x_images]
            y_images = [train_transforms(y_image) for y_image in y_images]
        else:
            xy_images = [train_transforms(x_image, y_image) for x_image, y_image in zip(x_images, y_images)]
            x_images, y_images = zip(*xy_images)

        examples["x_pixel_values"] = x_images
        examples["y_pixel_values"] = y_images

        input_ids = tokenize_captions(examples, x_text_column = x_text_column, y_text_column = y_text_column, trigger_word = args.trigger_word)
        x_len = len(input_ids) // 2
        examples["x_input_ids"] = input_ids[:x_len]
        examples["y_input_ids"] = input_ids[x_len:]
        return examples

    def filter_incomplete_samples(example):

        # Check if both 'image' and 'json' fields are present in the sample
        has_image = True
        for clm in x_image_column:
            has_image = has_image and example.get(clm) is not None
        for clm in y_image_column:
            has_image = has_image and example.get(clm) is not None
        has_json = example.get("json") is not None

        # resolution_pass = max(example[x_image_column[0]].size) > args.resolution
        
        # Return True if both are present, False otherwise
        return has_image and has_json
    
    

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        
        if args.dataset_type == "webdataset":
            dataset["train"] = dataset["train"].filter(filter_incomplete_samples)

        if hasattr(dataset["train"], "with_transform"):
            train_dataset = dataset["train"].with_transform(preprocess_train)
        else:
            train_dataset = dataset["train"].map(preprocess_train, batched=True, batch_size=args.train_batch_size)
        

    def collate_fn(examples):
        x_pixel_values = torch.stack([example["x_pixel_values"] for example in examples])
        x_pixel_values = x_pixel_values.to(memory_format=torch.contiguous_format).float()
        y_pixel_values = torch.stack([example["y_pixel_values"] for example in examples])
        y_pixel_values = y_pixel_values.to(memory_format=torch.contiguous_format).float()
        x_input_ids = torch.stack([example["x_input_ids"] for example in examples])
        y_input_ids = torch.stack([example["y_input_ids"] for example in examples])
        return {"x_pixel_values": x_pixel_values, "x_input_ids": x_input_ids, "y_input_ids": y_input_ids, "y_pixel_values": y_pixel_values}

    dataset_length = len(train_dataset)
    
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True if args.dataset_type != "webdataset" else False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    logger.info(f"Dataset length: {dataset_length}")
    logger.info(f"Dataloader length on each process {len(train_dataloader)}")

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
        accelerator.init_trackers("unicon", config={k:v if not isinstance(v, list) else " ".join(v) for k,v in vars(args).items()})

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
    logger.info(f"Resume training from {args.resume_from_checkpoint}")

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
    
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    logger.info(f"VAE scale factor: {vae_scale_factor}")

    bsz = 0

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        logger.info(f"Epoch {epoch}, Step {global_step}")
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                if args.rand_transform:
                    train_transforms = random_train_transforms()

                xy_pixel_values = torch.cat([batch["x_pixel_values"], batch["y_pixel_values"]], dim=0)
                xy_input_ids = torch.cat([batch["x_input_ids"], batch["y_input_ids"]], dim=0)

                if bsz != xy_pixel_values.shape[0]:
                    bsz = xy_pixel_values.shape[0]
                    set_unicon_config_train(unwrap_model(unet), bsz)

                # Sample random timestep for each input
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))

                
                timesteps = timesteps.long()

                # Convert images to latent space
                latents = vae.encode(xy_pixel_values.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                timesteps = timesteps.to(latents.device)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(xy_input_ids, return_dict=False)[0]

                if args.prompt_dropout_prob > 0:
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
                
                if args.joint_dropout_prob > 0:
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
                    params_to_clip = trainable_params
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
                        
                        # Save the lora layers
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

                        logger.info(f"Saved checkpoint to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break


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

        logger.info(f"Saved final weights to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
