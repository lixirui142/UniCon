import random
import json
from datetime import datetime
import numpy as np
from PIL import Image, PngImagePlugin

import torch
import torchvision.transforms.v2 as transforms
from peft.tuners.lora.layer import BaseTunerLayer
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from patch import patch
from utils.load_utils import load_scheduler

import pdb

def normalize_image(image):
    return (image - 0.5) / 0.5

def denormalize_image(image):
    return (image + 1) / 2

@torch.no_grad()
def blip2_cap(images, processor = None, model = None, device = "cuda"):

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b") if processor is None else processor
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to("cuda") if model is None else model

    if not isinstance(images[0], Image.Image):
        raw_images = [transforms.ToPILImage()(image) for image in images]
    else:
        raw_images = images

    inputs = processor(raw_images, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=77)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return [text.strip() for text in generated_text]

def save_png_with_comment(image, metadata, output_path):
    # Convert metadata (input variables) to a JSON string
    metadata_str = json.dumps(metadata)
    
    if not isinstance(image, Image.Image):
    # Convert the image array to a PIL image
        image = Image.fromarray(image.astype('uint8'))
    
    # Create a PngInfo object to hold the metadata
    png_info = PngImagePlugin.PngInfo()
    
    # Add the metadata as a comment (tEXt chunk)
    png_info.add_text("Comment", metadata_str)
    
    # Save the image with the metadata comment
    image.save(output_path, "PNG", pnginfo=png_info)

    return output_path

import uuid

def get_combined_filename(name="image", extension="png"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_id = uuid.uuid4().hex[:6]  # Short UUID for brevity
    return f"{name}_{timestamp}_{unique_id}.{extension}"

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_adapter_names(unet, models = None):
    unicon_adapters = unet.unicon_config["unicon_adapters"]
    models = unicon_adapters.keys() if models is None else models
    adapter_names = []
    for model_name in models:
        adapter_names += unicon_adapters[model_name]
    return adapter_names

def set_unicon_infer_adapters(unet, adapters):
    
    if "active_adapters" not in unet.unicon_config or set(adapters) != set(unet.unicon_config["active_adapters"]):
        unet.unicon_config["active_adapters"] = adapters
        unet.set_adapters(adapters)
        for name, param in unet.named_parameters():
            param.requires_grad = False
    

def set_unicon_config_inference(unet, input_pairs, use_cfg = True, input_len = None, device = "cuda", debug=False):
    """ Set config for unicon inference.
        It does two things:
        1. Tell the model how to pair the inputs for joint cross attention.
            The input_pairs shoule be:
            [
                (x_0, y_0, wx_0, wy_0, model_name_0),
                (x_1, y_1, wx_1, wy_1, model_name_1),
                ...
            ]
            A simple example is [(0,1,1.0,1.0,"depth")], which means the model will pair your 1st and 2nd input (count in batch dimension) for the joint cross attention of depth model. And the attention output will be scaled by 1.0 for both inputs.
        2. Set active LoRA adapters and their masks.
            According to input pairs, we can determine which LoRA adapters to use. Then set their masks so that the adapters can selectively apply to the inputs.
            In above simple example [(0,1,1.0,1.0,"depth")], we need masks for depth_y_lora, depth_xy_lora, depth_yx_lora.
            As x has index 0 and y has index 1, depth_y_lora mask is [False, True].
            In the joint cross attention, the input is:
                [x,y] -> Q
                [y,x] -> K,V
            So depth_xy_lora mask is (Q: [True, False], K,V: [False, True]) and depth_yx_lora mask is (Q: [False, True], K,V: [True, False]), so that depth_xy_lora apply to x and depth_yx_lora apply to y.
    """

    attn_config = list(zip(*input_pairs)) 
    for i in range(len(attn_config) - 1):
        attn_config[i] = torch.tensor(attn_config[i])
    x_ids, y_ids, x_weights, y_weights, model_names = attn_config
    x_weights = x_weights.view(-1, 1, 1)
    y_weights = y_weights.view(-1, 1, 1)

    if use_cfg:
        x_ids = torch.cat([x_ids, x_ids + input_len])
        y_ids = torch.cat([y_ids, y_ids + input_len])
        x_weights, y_weights = torch.cat([x_weights] * 2), torch.cat([y_weights] * 2)
        model_names = model_names * 2
        # batch_size *= 2
    
    cond_masks = dict()
    false_mask = [False] * len(model_names)

    input_len = input_len * 2 if use_cfg else input_len
    
    active_adapters = get_adapter_names(unet, set(model_names))
    set_unicon_infer_adapters(unet, active_adapters)

    
    for cur_cond in set(model_names):
        cond_masks[cur_cond] = [model_name == cur_cond for model_name in model_names]
        xy_lora = f"{cur_cond}_xy_lora"
        yx_lora = f"{cur_cond}_yx_lora"
        xy_lora_qo_mask = yx_lora_kv_mask = cond_masks[cur_cond] + false_mask
        xy_lora_kv_mask = yx_lora_qo_mask = false_mask + cond_masks[cur_cond]
        patch.set_patch_lora_mask(unet, xy_lora, xy_lora_qo_mask, kv_lora_mask = xy_lora_kv_mask)
        patch.set_patch_lora_mask(unet, yx_lora, yx_lora_qo_mask, kv_lora_mask = yx_lora_kv_mask)
        
        if debug:
            print("Set", xy_lora, xy_lora_qo_mask, xy_lora_kv_mask)
            print("Set", yx_lora, yx_lora_qo_mask, yx_lora_kv_mask)
        y_lora = f"{cur_cond}_y_lora"
        if y_lora in active_adapters:
            cur_y_ids = y_ids[cond_masks[cur_cond]]
            y_lora_mask = [True if i in cur_y_ids else False for i in range(input_len)]
            # y_lora_mask = torch.zeros(input_len).to(torch.bool)
            # for y_id in y_ids[cond_masks[cur_cond]]:
            #     y_lora_mask[y_id] = True
            patch.set_patch_lora_mask(unet, y_lora, y_lora_mask)
            if debug:
                print("Set", y_lora, y_lora_mask)
            

    attn_config = x_ids.to(device), y_ids.to(device), x_weights.to(device), y_weights.to(device)
    unet.unicon_config["attn_config"] = attn_config
    unet.unicon_config["cond_masks"] = cond_masks
    if debug:
        print("Set attn_config", attn_config)
        print("Set cond masks", cond_masks)
    
    return unet

def set_unicon_config_train(unet, input_len, device = "cuda", dtype = torch.float16, debug = False):

    xy_lora = "xy_lora"
    yx_lora = "yx_lora"
    xlen = ylen = input_len // 2
    true_mask = [True] * xlen
    false_mask = [False] * xlen
    xy_lora_qo_mask = yx_lora_kv_mask = true_mask + false_mask
    xy_lora_kv_mask = yx_lora_qo_mask = false_mask + true_mask
    patch.set_patch_lora_mask(unet, xy_lora, xy_lora_qo_mask, kv_lora_mask = xy_lora_kv_mask)
    patch.set_patch_lora_mask(unet, yx_lora, yx_lora_qo_mask, kv_lora_mask = yx_lora_kv_mask)

    if debug:
        print("Set", xy_lora, xy_lora_qo_mask, xy_lora_kv_mask)
        print("Set", yx_lora, yx_lora_qo_mask, yx_lora_kv_mask)

    y_lora = "y_lora"
    y_lora_mask = false_mask + true_mask
    patch.set_patch_lora_mask(unet, y_lora, y_lora_mask)
    if debug:
        print("Set", y_lora, y_lora_mask)

    x_ids = torch.arange(xlen).to(device)
    y_ids = torch.arange(xlen, input_len).to(device)
    x_weights = torch.ones([1,1,1]).to(device).to(dtype)
    y_weights = torch.ones([1,1,1]).to(device).to(dtype)
    attn_config = x_ids, y_ids, x_weights, y_weights
    unet.unicon_config["attn_config"] = attn_config
    if debug:
        print("Set attn_config", attn_config)
    
    return unet

def parse_schedule(sample_schedule, num_inference_step):
    """ Translate user-friendly schedule to actual sampling schedule.
        Original sample schedule looks like: [seg0,seg1,...] and each seg is [(l_0,r_0),(l_1,r_1),...,(l_n,r_n),with_guidance].
        with_guidance: whether to add guidance for inpainting
        l_i,r_i: an interval in (0,1), will be scaled to (0, N) where N = num_inference_step. It indicates that the i_th input will be denoised from l_i* step to r_i* step in current schedule segment (*: after scaling). 
        A simple example is [[(0,1),(1,1),False]]. Suppose our input is [x,y]. It means sampling x from step 0 to step N (i.e. from pure noise to clean latent) while keeping y at step N (i.e. clean latent), without guidance.
        And the translated schedule will be:
        [
            (0,N-1,0),
            (1,N-1,0),
            ...,
            (N-1,N-1,0)
        ]
    """
    num_inference_step -= 1
    full_schedule = []
    for seg_id, schedule_seg in enumerate(sample_schedule):
        with_guidance = schedule_seg[-1]
        step_segs = []
        step_ranges = []
        for sep_schedule in schedule_seg[:-1]:
            le, ri = int(sep_schedule[0] * num_inference_step), int(sep_schedule[1] * num_inference_step)
            # le, ri = min(num_inference_step - 1, le), min(num_inference_step - 1, ri)
            step_segs.append((le,ri))
            step_ranges.append(ri - le + 1)

        step_num = max(step_ranges)
        for i in range(step_num):
            cur_steps = []
            for step_seg, step_range in zip(step_segs, step_ranges):
                step = step_seg[0] + int(step_range * i / step_num)
                cur_steps.append(step)
            
            if i == 0 and seg_id > 0:
                prev_steps = full_schedule[-1][:-2]
                assert False not in [cur_step == prev_step for cur_step, prev_step in zip(cur_steps, prev_steps)], "Schedule segments should be continuous."
                continue

            full_schedule.append([*cur_steps, with_guidance])
    return full_schedule

def process_images(images, h = None, w = None, verbose = False, div = None, rand_crop = False):
    # Resize and crop image to (h, w) while keeping the aspect ratio

    if isinstance(images[0], Image.Image):
        fh, fw = images[0].height, images[0].width
    else:
        fh, fw = images.shape[-2:]
        assert len(images.shape) >= 3
        if len(images.shape) == 3:
            images = [images]
    
    if h is None and w is None:
        ratio = 1
        h, w = fh, fw
    elif h is None:
        ratio = w / fw
        h = int(fh * ratio)
    elif w is None:
        ratio = h / fh
        w = int(fw * ratio)
    else:
        h_ratio = h / fh
        w_ratio = w / fw
        ratio = max(h_ratio, w_ratio)

    
    if div is not None:
        h = h // div * div
        w = w // div * div
    

    size = (int(fh * ratio + 0.5), int(fw * ratio + 0.5))
    # print(ratio, size)
        # if nw >= w:
        #     size = (h, nw)
        # else:
        #     size = (int(fh / fw * w), w)

    if verbose:
        print(
            f"[INFO] image size {(fh, fw)} resize to {size} and centercrop to {(h, w)}")

    image_ls = []
    for image in images:
        if ratio <= 1 and rand_crop:
            resized_frame = image
        else:
            resized_frame = transforms.Resize(size, antialias=True)(image)
        if rand_crop:
            cropped_frame = transforms.RandomCrop([h, w])(resized_frame)
        else:
            cropped_frame = transforms.CenterCrop([h, w])(resized_frame)
        
        image_ls.append(cropped_frame)
    if isinstance(images[0], Image.Image):
        return image_ls
    else:
        return torch.stack(image_ls)


def repeat_interleave(ls, num_repeats):

    return sum([[ls[i]] * num_repeats for i in range(len(ls))], start = [])

def unicon_infer(pipeline, input_config, prompt = "", negative_prompt = "", height = 512, width = 512, batch_size = 1, num_inference_step = 50, seed = None, scheduler_selection = "ead", guidance_scale = 7.5, joint_scale = 1.0, cond_guidance_scale = 0.0, debug = False):

    init_images = input_config["input_images"]

    if "image_masks" in input_config:
        masks = input_config["image_masks"]
    else:
        mask = Image.new('RGB', (width, height), (255, 255, 255))
        masks = [mask] * len(init_images)

    num_input = len(init_images)
    prompts = prompt if isinstance(prompt, list) else [prompt] * num_input
    negative_prompts = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * num_input

    init_images, masks = repeat_interleave(init_images, batch_size), repeat_interleave(masks, batch_size)
    prompts, negative_prompts = repeat_interleave(prompts, batch_size), repeat_interleave(negative_prompts, batch_size)

    input_pairs = []
    for input_pair in input_config["input_pairs"]:
        x_id, y_id, x_wt, y_wt, mn = input_pair
        input_pairs += [(x_id * batch_size + i, y_id * batch_size + i, x_wt, y_wt, mn) for i in range(batch_size)]

    set_unicon_config_inference(pipeline.unet, input_pairs, input_len = len(init_images), debug=debug)
    load_scheduler(pipeline, mode = scheduler_selection)

    generator = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None

    sample_schedule = parse_schedule(input_config["sample_schedule"], num_inference_step)

    pipeline.to("cuda")

    inference_kwargs = {
        "prompt" : prompts,
        "image" : init_images,
        "mask_image" : masks,
        "height" : height,
        "width" : width,
        "num_inference_steps" : num_inference_step,
        "guidance_scale" : guidance_scale,
        "negative_prompt" : negative_prompts,
        "eta" : 1.0,
        "sample_schedule" : sample_schedule,
        "cond_guidance_scale" : cond_guidance_scale,
        "xlen" : batch_size,
        "generator":generator,
    }

    # pdb.set_trace()
    with torch.no_grad():
        with torch.autocast(device_type = "cuda", dtype = torch.float16):
            images = pipeline(**inference_kwargs).images
        
    return images

# ToTensor = transforms.Compose(
#                 [
#                     transforms.ToImage(),
#                     transforms.ToDtype(torch.float32, scale=True),
#                 ]
#             )

# ToPILImage = transforms.ToPILImage()