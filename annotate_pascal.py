import argparse
from tqdm import tqdm
import glob
from PIL import Image
import numpy as np
import os
import json

import torch
from accelerate import Accelerator
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from accelerate.utils import gather_object

from annotator.openpose import OpenposeDetector
from annotator.hed import HEDdetector
from annotator.depth_anything_v2 import DepthAnythingV2Detector
from annotator.util import HWC3



parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    default="data/PascalVOC/VOC2012/",
    help="path to the data root directory",
)

args = parser.parse_args()

data_root = args.data_root

image_dir = os.path.join(data_root, "JPEGImages")

depth_dir, hed_dir, pose_dir = [
    os.path.join(data_root, "depth"),
    os.path.join(data_root, "hed"),
    os.path.join(data_root, "pose"),
]

save_dirs = [depth_dir, hed_dir, pose_dir]

for save_dir in save_dirs:
    os.makedirs(save_dir, exist_ok=True)

dataset = glob.glob(os.path.join(image_dir, "*.jpg"))

accelerator = Accelerator()

depth_model, hed_model, pose_model = [
    DepthAnythingV2Detector(device=accelerator.device),
    HEDdetector(device=accelerator.device),
    OpenposeDetector(),
]
annotators = [depth_model, hed_model, pose_model]

device = accelerator.device
blip2_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", revision="51572668da0eb669e01a189dc22abe6088589a24")
blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", revision="51572668da0eb669e01a189dc22abe6088589a24", torch_dtype=torch.float16).to(device, torch.float16)

@torch.no_grad()
def generate_caption(images):

    inputs = blip2_processor(images, return_tensors="pt").to(device, torch.float16)
    generated_ids = blip2_model.generate(**inputs, max_new_tokens=77)
    generated_text = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_text

metadata_path = os.path.join(data_root, "metadata.json")
if not os.path.exists(metadata_path):
    metadata = dict()

    with accelerator.split_between_processes(dataset) as dataset_split:
        for image_path in tqdm(dataset_split):
            img_raw = Image.open(image_path).convert("RGB")
            img = np.array(img_raw)
            img = HWC3(img)

            name = os.path.basename(image_path)

            for model, save_dir in zip(annotators, save_dirs):
                output_path = os.path.join(save_dir, name)
                # if not os.path.exists(output_path):
                control = model(img)
                control = Image.fromarray(control)
                control.save(output_path)

            metadata[name] = {
                "image": image_path,
                "depth": os.path.join(depth_dir, name),
                "hed": os.path.join(hed_dir, name),
                "pose": os.path.join(pose_dir, name),
            }

    names, images = [], []
    caption_batch_size = 128

    with accelerator.split_between_processes(dataset) as dataset_split:
        dataset_len = len(dataset_split)
        for i, image_path in enumerate(tqdm(dataset_split)):
            img_raw = Image.open(image_path).convert("RGB")

            name = os.path.basename(image_path)

            names.append(name)
            images.append(img_raw)

            if len(images) > 0 and ((i + 1) % caption_batch_size == 0 or i == dataset_len - 1):
            
                captions = generate_caption(images)
                for name, caption in zip(names, captions):
                    metadata[name]["caption"] = caption
                names, images = [], []

    metadata = gather_object(metadata)
else:
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

if accelerator.is_main_process:
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    train_split, val_split, human_split = "data/PascalVOC_train.json", "data/PascalVOC_val.json", "data/PascalVOC_people.json"

    with open(train_split, "r") as f:
        train_split_ls = json.load(f)

    with open(val_split, "r") as f:
        val_split_ls = json.load(f)

    with open(human_split, "r") as f:
        human_split_ls = json.load(f)

    train_dataset, val_dataset, human_dataset = [], [], []

    for name, data in metadata.items():
        if name in train_split_ls:
            train_dataset.append(data)
        if name in val_split_ls:
            val_dataset.append(data)
        if name in human_split_ls:
            human_dataset.append(data)
    
    with open(os.path.join(data_root, "train_dataset.json"), "w") as f:
        json.dump(train_dataset, f)
    
    with open(os.path.join(data_root, "val_dataset.json"), "w") as f:
        json.dump(val_dataset, f)
    
    with open(os.path.join(data_root, "human_dataset.json"), "w") as f:
        json.dump(human_dataset, f)
        
