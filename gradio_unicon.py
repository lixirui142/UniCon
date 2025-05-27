import os
import json
import numpy as np
from PIL import Image
from copy import copy

import torch
from diffusers.utils import make_image_grid
import gradio as gr

from patch import patch
from pipeline.pipeline_unicon import StableDiffusionUniConPipeline
from utils.utils import blip2_cap, get_combined_filename, save_png_with_comment, set_unicon_config_inference, parse_schedule, process_images, unicon_infer
from utils.load_utils import load_model_configs, load_unicon, load_blip_processor, load_annotator, annotator_dict, load_unicon_weights, load_scheduler
from annotator.util import HWC3

import pdb

model_configs = load_model_configs()
cur_model = "depth"
pipeline_class = StableDiffusionUniConPipeline
pipe = load_unicon(pipeline_class, model_configs[cur_model])

save_dir = "output/gallery"
os.makedirs(save_dir, exist_ok=True)
annotator_cache = dict()
blip_cache = dict()
base_model_dict = {
    "sd1.5": "runwayml/stable-diffusion-v1-5",
    "realistic-v4": "digiplay/majicMIX_realistic_v4",
    "anything-v5": "stablediffusionapi/anything-v5",
}
default_base_model_ids = {name: cfg["base_model_id"] for name, cfg in model_configs.items()}

example_path = "data/example_inputs/example_config.json"
example_settings = ["image_input", "cond_input", "prompt", "seed", "model_selection", "scheduler_selection", "num_inference_step", "height", "width", "image_noise_strength", "cond_noise_strength", "joint_scale", "cond_guidance_scale", "y_prompt"]

def load_examples(example_path):
    with open(example_path, "r") as f:
        example_configs = json.load(f)
    examples = []
    for config in example_configs.values():
        examples.append([config[key] for key in example_settings])
    return examples

examples = load_examples(example_path)

debug = False

def annotate(image, annotator_selection):
    if annotator_selection not in annotator_cache:
        annotator_cache[annotator_selection] = load_annotator(annotator_selection)
    img = Image.open(image).convert("RGB")
    img = np.array(img)
    img = HWC3(img)
    cond_img = annotator_cache[annotator_selection](img)
    save_path = os.path.join(save_dir, get_combined_filename())
    
    cond_img = Image.fromarray(cond_img)
    cond_img.save(save_path)
    return save_path

def generate_caption(image_input):
    if len(blip_cache) == 0:
        blip_processor, blip_model = load_blip_processor()
        blip_cache["processor"] = blip_processor
        blip_cache["model"] = blip_model
    else:
        blip_processor, blip_model = blip_cache["processor"], blip_cache["model"]
    image = Image.open(image_input).convert("RGB")
    prompt = blip2_cap([image], processor = blip_processor, model = blip_model)[0]
    
    return prompt

def process(image_input, cond_input, prompt, model_selection, additional_prompt, negative_prompt, y_prompt, enable_joint_attn, batch_size, height, width, num_inference_step, seed, scheduler_selection, base_model_selection, guidance_scale, joint_scale, image_noise_strength, cond_noise_strength, cond_guidance_scale):
    
    metadata = copy(locals())

    global cur_model, pipe
    if base_model_selection == "default":
        base_model_id = default_base_model_ids[cur_model]
    else:
        base_model_id = base_model_dict[base_model_selection]

    if base_model_id != pipe.unet.base_model_id:
        pipe.unet.base_model_id = model_configs[cur_model]["base_model_id"] = base_model_id
        pipe = load_unicon(pipeline_class, model_configs[cur_model])
    elif cur_model != model_selection:
        cur_model = model_selection
        unicon_adapters = pipe.unet.unicon_config["unicon_adapters"]
        if cur_model not in unicon_adapters:
            model_config = model_configs[cur_model]
            checkpoint_path = model_config["checkpoint_path"]
            model_name = model_config["model_name"]
            post_joint = model_config["post_joint"]
            adapter_names = model_config["adapter_names"]
            active_adapters = load_unicon_weights(pipe.unet, checkpoint_path, post_joint, model_name = model_name, adapter_names = adapter_names)
        else:
            active_adapters = unicon_adapters[cur_model]
    
    model_config = model_configs[cur_model]
    
    if image_input is None:
        image_input = Image.new('RGB', (width, height), (255, 255, 255))
    else:
        image_input = Image.open(image_input).convert("RGB")
    if cond_input is None:
        cond_input = Image.new('RGB', (width, height), (255, 255, 255))
    else:
        cond_input = Image.open(cond_input).convert("RGB")

    init_images = [image_input,  cond_input]
    init_images = [process_images([init_image], height, width,  verbose = debug, div = 8)[0] for init_image in init_images]
    
    sample_schedule = [[(1 - image_noise_strength, 1), (1 - cond_noise_strength, 1), 0]]
    input_pairs = [(0, 1, joint_scale, joint_scale, cur_model)]

    input_config = {
        "input_images" : init_images,
        "sample_schedule" : sample_schedule,
        "input_pairs": input_pairs
    }

    prompt = additional_prompt + prompt
    y_prompt = prompt if y_prompt == "None" else additional_prompt + y_prompt

    trigger_word = model_config["trigger_word"]
    prompts = [prompt, trigger_word + y_prompt]

    if negative_prompt is not None:
        negative_prompts = [negative_prompt] * 2
    
    patch.set_joint_attention(pipe.unet, enable = enable_joint_attn)

    images = unicon_infer(pipe, input_config, prompt = prompts, negative_prompt = negative_prompts, height = height, width = width, batch_size = batch_size, num_inference_step = num_inference_step, seed = seed, scheduler_selection = scheduler_selection, guidance_scale = guidance_scale, joint_scale = joint_scale, cond_guidance_scale = cond_guidance_scale, debug = False)

    input_images = [init_images[0]] * batch_size + [init_images[1]] * batch_size
    image_grid = make_image_grid([*input_images, *images], rows=2, cols=batch_size * 2)
    save_path = save_png_with_comment(image_grid, metadata, os.path.join(save_dir, get_combined_filename()))

    # return image_grid, images[:batch_size], images[batch_size:]
    return [image_grid, *images]


demo = gr.Blocks()
with demo:
    with gr.Row():
        gr.Markdown("## UniCon models demo")
    gr.Markdown("""
1. Upload an image and/or a condition input
2. Provide a prompt (or click "Generate Caption") and select the UniCon model corresponding to your image adnd condition
3. Set image and condition noise strengths. They indicate the noise levels added to the image input and the condition input to initialize the latents. Setting to 0 means no noise will be added (clean input). Setting to 1 equals to using random noise as initial latents.

There are examples of different UniCon models and inference tasks below for you to get familiar with all the options.
""")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                image_input = gr.Image(type='filepath', label='image input (x)')
                cond_input = gr.Image(type='filepath', label='condition input (y)')
            prompt = gr.Textbox(label="prompt")
            model_selection = gr.Dropdown(choices =list(model_configs.keys()), label="model selection", value=cur_model)
            image_noise_strength = gr.Slider(label="image noise strength", minimum=0.0, maximum=1.0, value=1.0, step=0.05)
            cond_noise_strength = gr.Slider(label="condition noise strength", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
            with gr.Accordion("Advanced options", open=False):
                additional_prompt = gr.Textbox(label="additional prompt", value = "best quality, extremely detailed, ")
                negative_prompt = gr.Textbox(label="negative prompt", value = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality")
                y_prompt = gr.Textbox(label="y prompt", value="None")
                enable_joint_attn = gr.Checkbox(label='enable joint cross attention', value=True)
                batch_size = gr.Number(label="batch size", value=1, precision=0)
                height = gr.Slider(label="height", minimum=128, maximum=1024, value=512, step=64)
                width = gr.Slider(label="width ", minimum=128, maximum=1024, value=512, step=64)
                num_inference_step = gr.Slider(label="inference step number", minimum=1, maximum=100, value=50, step=1)
                seed = gr.Slider(label="seed", minimum=-1, maximum=2147483647, step=1, randomize=True) #8621329
                # seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=8621329)
                scheduler_selection = gr.Dropdown(choices= ["ddim", "ddpm", "ead", "dpm"], label="Scheduler Selection", value="ead")
                base_model_selection = gr.Dropdown(choices =list(base_model_dict.keys()) + ["default"], label="base model selection", value="default")
                guidance_scale = gr.Number(label="CFG guidance sacle", value=7.5)
                joint_scale = gr.Slider(label="joint attention scale", minimum=0, maximum = 2, value = 1, step = 0.1)
                cond_guidance_scale = gr.Number(label="condition guidance scale", value=0.0)
            
            annotator_selection = gr.Dropdown(choices = annotator_dict.keys(), label="annotator selection", value="depth")
            
            with gr.Row():
                annotate_button = gr.Button(value = "Annotate Image")
                caption_button = gr.Button(value = "Generate Caption")
            
            with gr.Row():
                run_button = gr.Button()
                
        with gr.Column():
            result_image = gr.Gallery(label="Results", elem_id="gallery",object_fit="contain", height="auto")
            # , format="png")
            gr.Markdown("""Results:
            1. a image grid of input x,y (Row 1) and output x,y (Row 2)
            2. N image (x) outputs 
            3. N condition (y) outputs
            """)
            # ixy_oxy = gr.Image(type='filepath', label='input x,y (Row 1) and output x,y (Row 2)')
            # with gr.Row():
            #     image_output = gr.Gallery(label="image output (x)", elem_id="gallery",object_fit="contain", height="auto", format="png")
            #     cond_output = gr.Gallery(label="condition output (x)", elem_id="gallery",object_fit="contain", height="auto", format="png")
                
    with gr.Row():
        examples = gr.Examples(examples, [image_input, cond_input, prompt, seed, model_selection, scheduler_selection, num_inference_step, height, width, image_noise_strength, cond_noise_strength, joint_scale, cond_guidance_scale, y_prompt], label = "Image Examples")

    ips = [image_input, cond_input, prompt, model_selection, additional_prompt, negative_prompt, y_prompt, enable_joint_attn, batch_size, height, width, num_inference_step, seed, scheduler_selection, base_model_selection, guidance_scale, joint_scale, image_noise_strength, cond_noise_strength, cond_guidance_scale]
    run_button.click(fn=process, inputs=ips, outputs=[result_image])
    caption_button.click(fn=generate_caption, inputs=[image_input], outputs = [prompt])
    annotate_button.click(fn=annotate, inputs=[image_input, annotator_selection], outputs = [cond_input])


demo.launch(allowed_paths=["."], server_name='0.0.0.0', share=True)