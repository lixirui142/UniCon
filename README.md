## **UniCon: A Simple Approach to Unifying Diffusion-based Conditional Generation (ICLR 2025)**

[Xirui Li](https://lixirui142.github.io/), [Charles Herrmann](https://scholar.google.com/citations?user=LQvi5XAAAAAJ&hl=en&oi=ao), [Kelvin C.K. Chan](https://ckkelvinchan.github.io/), [Yinxiao Li](https://research.google/people/yinxiaoli/?&type=google), [Deqing Sun](https://deqings.github.io/), [Chao Ma](https://vision.sjtu.edu.cn/), [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/)<br>

<a href="https://arxiv.org/abs/2410.11439"><img src='https://img.shields.io/badge/arXiv-UniCon-red' alt='Paper PDF'></a>
<a href='https://lixirui142.github.io/unicon-diffusion'><img src='https://img.shields.io/badge/Project_Page-UniCon-green' alt='Project Page'></a>

TL;DR: The proposed UniCon enables diverse generation behavior in one model for a target image-condition pair.

https://github.com/user-attachments/assets/3e832c64-35c5-45fa-8af8-fbe54bdfbc07

<details><summary> Abstract </summary>

> *Recent progress in image generation has sparked research into controlling these models through condition signals, with various methods addressing specific challenges in conditional generation. Instead of proposing another specialized technique, we introduce a simple, unified framework to handle diverse conditional generation tasks involving a specific image-condition correlation. By learning a joint distribution over a correlated image pair (e.g. image and depth) with a diffusion model, our approach enables versatile capabilities via different inference-time sampling schemes, including controllable image generation (e.g. depth to image), estimation (e.g. image to depth), signal guidance, joint generation (image & depth), and coarse control. Previous attempts at unification often introduce significant complexity through multi-stage training, architectural modification, or increased parameter counts. In contrast, our simple formulation requires a single, computationally efficient training stage, maintains the standard model input, and adds minimal learned parameters (15% of the base model). Moreover, our model supports additional capabilities like non-spatially aligned and coarse conditioning. Extensive results show that our single model can produce comparable results with specialized methods and better results than prior unified methods. We also demonstrate that multiple models can be effectively combined for multi-signal conditional generation.*
</details>

## Setup
1. Clone the repository and install requirements.
```shell
git clone https://github.com/lixirui142/UniCon
cd UniCon
pip install -r requirements.txt
```
2. Download pretrained UniCon model weights from [here](https://huggingface.co/lixirui142/unicon) to "weights" dir by running:
```shell
python download_pretrained_weights.py
```
Now we have four unicon models (depth, edge, pose, id) based on [SDv1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).

## Usage

### Gradio Demo
We provide a [gradio demo](gradio_unicon.py) to showcase the usage of UniCon models. There are some examples to get you familiar with inference options for different tasks. To run the demo: 
```shell
python gradio_unicon.py
```

### Train

To train UniCon Depth, Edge and Pose model on PascalVOC, first download and annotate the PascalVOC dataset:
```shell
bash train/download_pascal.sh
python annotate_pascal.py
```
It will download the dataset to [data dir](data) and generate condition maps and captions. Some json files are created to save the dataset information. Then run the training script:
```shell
bash train/train_unicon_depth.sh
bash train/train_unicon_hed.sh
bash train/train_unicon_pose.sh
```
It costs about 13 hours to train one model on single NVIDIA A100 80G. You can run `python train_unicon.py --help` to check available training parameters.

## TODO
- [ ] Provide notebooks and python scripts for more inference cases.
- [x] Clean and release training code.

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{li2024unicon,
    title={A Simple Approach to Unifying Diffusion-based Conditional Generation},
    author={Li, Xirui and Herrmann, Charles and Chan, Kelvin CK and Li, Yinxiao and Sun, Deqing and Yang, Ming-Hsuan},
    booktitle={arXiv preprint arxiv:2410.11439},
    year={2024}
    }
```
