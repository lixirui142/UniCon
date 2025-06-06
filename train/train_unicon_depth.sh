#!/bin/bash

output_dir="output/unicon_depth"
train_data_dir="data/PascalVOC/VOC2012/train_dataset.json"
echo "$(realpath $0)" "$output_dir"
mkdir $output_dir
cp $(realpath $0) $output_dir

# Run the second program
accelerate launch train_unicon.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --train_data_dir=$train_data_dir \
 --dataset_type="json" \
 --x_image_column "image" \
 --y_image_column="depth" \
 --x_caption_column="caption" \
 --output_dir=$output_dir \
 --random_flip \
 --train_batch_size 16 \
 --num_train_epochs 20 \
 --mixed_precision fp16 \
 --rank 64 \
 --snr_gamma 5.0 \
 --checkpointing_steps 5000 \
 --prompt_dropout_prob 0.1 \
 --seed 142857 \
 --trigger_word "depth_map, " \
 --rand_transform \
 --train_y_lora \
 --report_to wandb \
 --post_joint conv_fuse
