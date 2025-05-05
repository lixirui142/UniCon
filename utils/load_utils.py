import os
import json
import sys
from glob import glob
import pdb

import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import DDIMScheduler, DDPMScheduler, PNDMScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from datasets import load_dataset

from patch import patch
from annotator.hed import HEDdetector
from annotator.depth_anything_v2 import DepthAnythingV2Detector
from annotator.openpose import OpenposeDetector
from utils.utils import get_active_adapters

model_config_root = "config/model_config"

annotator_dict = {
    "depth": DepthAnythingV2Detector,
    "edge": HEDdetector,
    "pose": OpenposeDetector
}

def load_model_configs():
    model_config_paths = glob(os.path.join(model_config_root, "*.json"))
    model_configs = dict()
    for model_config_path in model_config_paths:
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
        model_configs[model_config["model_name"]] = model_config
    return model_configs

# Load lora adapter and post projection weights of a UniCon model to UNet
def load_unicon_weights(unet, checkpoint_path, post_joint, model_name = None, adapter_names = ["xy_lora", "yx_lora", "y_lora"]):

    active_adapters = []

    for lora_name in adapter_names:
        save_dir = os.path.join(checkpoint_path, lora_name)
        if os.path.exists(save_dir):
            lora_state_dict, lora_network_alphas = StableDiffusionPipeline.lora_state_dict(save_dir)
            if model_name is not None:
                adapter_name = f"{model_name}_{lora_name}"
            else:
                adapter_name = lora_name
            StableDiffusionPipeline.load_lora_into_unet(lora_state_dict, lora_network_alphas, unet = unet, adapter_name = adapter_name)
            active_adapters.append(adapter_name)

    model_path = os.path.join(checkpoint_path, "model.pth")
    assert os.path.exists(model_path), f"{model_path} is not found."
    
    patch.add_post_joint(unet, model_name, post = post_joint)

    state_dict = torch.load(model_path, map_location="cpu")
    state_dict = {key: value for key, value in state_dict.items() if 'lora' not in key}
    # Deal with training weight name
    state_dict = {key.replace("conv1n", f"post1n.{model_name}"): value for key, value in state_dict.items()}
    missing_keys, unexpected_keys = unet.load_state_dict(state_dict, strict=False)
    

    if hasattr(unet, "unicon_adapters"):
        unet.unicon_adapters[model_name] = active_adapters
    else:
        unet.unicon_adapters = {model_name:active_adapters}

    print(model_name, "model weights loaded")
    
    return active_adapters


# Load UniCon model from checkpoint path to UNet
def load_unicon_to_unet(unet, checkpoint_path, model_name = None, post_joint = "conv"):

    patch.apply_patch(unet)
    patch.initialize_joint_layers(unet, post = post_joint)
    print("UniCon layers initialized.")

    active_adapters = load_unicon_weights(unet, checkpoint_path, post_joint, model_name = model_name)

    unet.set_adapters(active_adapters)
    patch.hack_lora_forward(unet)

    return unet

def load_unicon_pipeline(pipeline_class, base_model_id, vae_id = None):
    if vae_id is not None:
        print(f"VAE: {vae_id}")
        vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16)
        pipeline = pipeline_class.from_pretrained(base_model_id, vae = vae, safety_checker=None, torch_dtype=torch.float16)
    else:
        pipeline = pipeline_class.from_pretrained(base_model_id, safety_checker=None, torch_dtype=torch.float16)
        
    print(f"UniCon pipeline loaded. Base model: {base_model_id}")    
    return pipeline

# 1 Load pipeline. 2 Add UniCon to UNet and load weights. 3 Load more UniCon models if any.
def load_unicon(pipeline_class, model_configs):
    if not isinstance(model_configs, list):
        model_configs = [model_configs]
    
    model_config =  model_configs[0]
    pipeline = load_unicon_pipeline(pipeline_class, model_configs[0]["base_model_id"])
    pipeline.unet.base_model_id = model_configs[0]["base_model_id"]

    checkpoint_path = model_config["checkpoint_path"]
    model_name = model_config["model_name"]
    post_joint = model_config["post_joint"]
    pipeline.unet = load_unicon_to_unet(pipeline.unet, checkpoint_path, model_name = model_name, post_joint = post_joint)


    active_adapters = get_active_adapters(pipeline.unet)
    for model_config in model_configs[1:]:
        checkpoint_path = model_config["checkpoint_path"]
        model_name = model_config["model_name"]
        post_joint = model_config["post_joint"]
        active_adapters += load_unicon_weights(pipeline.unet, checkpoint_path, post_joint, model_name = model_name)
    
    pipeline.unet.set_adapters(active_adapters)

    return pipeline

def load_blip_processor():
    # blip_processor = BlipProcessor.from_pretrained(
    #         "Salesforce/blip-image-captioning-large")
    # blip_model = BlipForConditionalGeneration.from_pretrained(
    #         "Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
    blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", 
        revision="51572668da0eb669e01a189dc22abe6088589a24")
    blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16, revision="51572668da0eb669e01a189dc22abe6088589a24").to("cuda")

    return blip_processor, blip_model

def load_annotator(annotator_name, **annotator_kwargs):
    return annotator_dict[annotator_name](**annotator_kwargs)

def load_scheduler(pipeline, mode="ddim"):
    if mode == "ddim":
        scheduler_cls = DDIMScheduler
    elif mode == "ddpm":
        scheduler_cls = DDPMScheduler
    elif mode == "pndm":
        scheduler_cls = PNDMScheduler
    elif mode == "ead":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif mode == "ed":
        scheduler_cls = EulerDiscreteScheduler
    elif mode == "dpm":
        scheduler_cls = DPMSolverMultistepScheduler
    pipeline.scheduler = scheduler_cls.from_config(pipeline.scheduler.config)

def load_image_folder(train_data_dir = None, cache_dir = None, dataset_type = None):
    data_files = {}
    if train_data_dir is not None:
        if dataset_type == "json":
            data_files["train"] = train_data_dir
        elif dataset_type == "imagefolder": 
            data_files["train"] = os.path.join(train_data_dir, "**")
        elif dataset_type == "parquet":
            data_files["train"] = glob.glob(os.path.join(train_data_dir, "*.parquet"))
        elif dataset_type == "webdataset":
            data_files["train"] = glob.glob(os.path.join(train_data_dir, "*.tar"))
        else:
            assert False, "Dataset type not defined"

    dataset = load_dataset(
        dataset_type,
        data_files=data_files,
        cache_dir=cache_dir,
        streaming=True if dataset_type == "webdataset" else False
    )
    
    return dataset