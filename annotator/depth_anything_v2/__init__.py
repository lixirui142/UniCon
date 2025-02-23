import cv2
import numpy as np
import torch
import os

from huggingface_hub import hf_hub_download

from .depth_anything_v2.dpt import DepthAnythingV2
from .metric_depth.depth_anything_v2.dpt import DepthAnythingV2 as DepthAnythingV2Metric
from annotator.util import annotator_ckpts_path

repo_ids_metric = {
   'vits': "depth-anything/Depth-Anything-V2-Metric-Hypersim-Small",
   'vitb': "depth-anything/Depth-Anything-V2-Metric-Hypersim-Base",
   'vitl': "depth-anything/Depth-Anything-V2-Metric-Hypersim-Large"
}

repo_ids = {
   'vits': "depth-anything/Depth-Anything-V2-Small",
   'vitb': "depth-anything/Depth-Anything-V2-Base",
   'vitl': "depth-anything/Depth-Anything-V2-Large"
}

class DepthAnythingV2Detector:
     # encoder in ['vitl 'vits', 'vitb', 'vitg']
    def __init__(self, encoder = 'vitl', device="cuda", metric = False):
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        if metric:
            dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
            max_depth = 20 # 20 for indoor model, 80 for outdoor model
            model = DepthAnythingV2Metric(**{**model_configs[encoder], 'max_depth': max_depth})
            filename = f'depth_anything_v2_metric_{dataset}_{encoder}.pth'
            modelpath = os.path.join(annotator_ckpts_path, filename)
            if not os.path.exists(modelpath):
                hf_hub_download(repo_id=repo_ids_metric[encoder], filename=filename, local_dir = annotator_ckpts_path)
            model.load_state_dict(torch.load(modelpath, map_location='cpu'))
        else:
            model = DepthAnythingV2(**model_configs[encoder])
            filename = f'depth_anything_v2_{encoder}.pth'
            modelpath = os.path.join(annotator_ckpts_path, filename)
            if not os.path.exists(modelpath):
                hf_hub_download(repo_id=repo_ids[encoder], filename=filename, local_dir = annotator_ckpts_path)
            model.load_state_dict(torch.load(modelpath, map_location='cpu'))
        self.model = model.to(device).eval()

    def __call__(self, input_image, return_depth = False):

        assert input_image.ndim == 3
        image_depth = input_image
        image_depth = cv2.cvtColor(image_depth, cv2.COLOR_RGB2BGR)
        with torch.no_grad():
            
            depth = self.model.infer_image(image_depth)
            depth -= np.min(depth)
            depth /= np.max(depth)
            depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)

            if return_depth:
                return depth_image, depth
            else:
                return depth_image
