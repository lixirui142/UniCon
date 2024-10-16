## UniCon: A Simple Approach to Unifying Diffusion-based Conditional Generation

[Xirui Li](https://github.com/lixirui142), [Charles Herrmann](https://scholar.google.com/citations?user=LQvi5XAAAAAJ&hl=en&oi=ao), [Kelvin C.K. Chan](https://ckkelvinchan.github.io/), [Yinxiao Li](https://research.google/people/yinxiaoli/?&type=google), [Deqing Sun](https://deqings.github.io/), [Chao Ma](https://vision.sjtu.edu.cn/), [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/)<br>

[**Project Page**](https://lixirui142.github.io/unicon-diffusion/) | [**Paper**](http://arxiv.org/abs/2410.11439)


https://github.com/user-attachments/assets/3e832c64-35c5-45fa-8af8-fbe54bdfbc07


TL;DR: The proposed UniCon enables diverse generation behavior in one model for a target image-condition pair.

<details><summary> Abstract </summary>

> *Recent progress in image generation has sparked research into controlling these models through condition signals, with various methods addressing specific challenges in conditional generation. Instead of proposing another specialized technique, we introduce a simple, unified framework to handle diverse conditional generation tasks involving a specific image-condition correlation. By learning a joint distribution over a correlated image pair (e.g. image and depth) with a diffusion model, our approach enables versatile capabilities via different inference-time sampling schemes, including controllable image generation (e.g. depth to image), estimation (e.g. image to depth), signal guidance, joint generation (image & depth), and coarse control. Previous attempts at unification often introduce significant complexity through multi-stage training, architectural modification, or increased parameter counts. In contrast, our simple formulation requires a single, computationally efficient training stage, maintains the standard model input, and adds minimal learned parameters (15% of the base model). Moreover, our model supports additional capabilities like non-spatially aligned and coarse conditioning. Extensive results show that our single model can produce comparable results with specialized methods and better results than prior unified methods. We also demonstrate that multiple models can be effectively combined for multi-signal conditional generation.*
</details>

