#!/bin/bash

# # Define the name of the first program
first_program="/opt/conda/envs/jointdiff/bin/accelerate"

# Function to check if the first program is running
is_running() {
    pgrep -f $first_program > /dev/null 2>&1
    return $?
}

# Wait until the first program is no longer running
while is_running; do
    sleep 5  # Check every 5 seconds
done

output_dir="output_final/output_depth_lora_joint_zoe_rank64_nta_cf_40e"
echo "$(realpath $0)" "$output_dir"
mkdir $output_dir
cp $(realpath $0) $output_dir

# Run the second program
accelerate launch train_models/train_depth_lora_joint.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --train_data_dir="/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/data/PascalVOC/VOC2012/zoe/train_dataset.json" \
 --dataset_type="json" \
 --image_column="image" \
 --caption_column="text" \
 --validation_prompt="there is a stork that is standing on a nest in the middle of a tree" \
 --output_dir=$output_dir \
 --random_flip \
 --train_batch_size 16 \
 --num_train_epochs 40 \
 --mixed_precision fp16 \
 --rank 64 \
 --x_column "original_image" \
 --snr_gamma 5.0 \
 --cond_image_path "/home/bml/.storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/readout_guidance/readout_training/data/raw/PascalVOC/VOC2012/JPEGImages/2010_004657.jpg" \
 --cond_depth_path "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/data/PascalVOC/VOC2012/depth/2010_004657.jpg" \
 --checkpointing_steps 5000 \
 --mask_dropout_prob 0.1 \
 --prompt_dropout_prob 0.1 \
 --seed 142857 \
 --trigger_word "depth_map, " \
 --rand_transform \
 --train_y \
 --report_to wandb \
 --post_joint conv_fuse \
 --no_timestep_align
#  --resume_from_checkpoint "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_depth_lora_joint_gray_rank64_nta_cf_40e_resume_metric/halfstone-20000"
#  --joint_dropout_prob 0.1
#  --dataloader_num_workers 0
#  --num_train_epochs 2 \
#  --resume_from_checkpoint "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_depth_lora_joint_gray_rank64_nta_cf_ds001/checkpoint-5000"
#  --quantize_cond \
#  --only_image_loss
#  --resume_from_checkpoint "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_normal_lora_joint_rank64/halfstone-10000"
#  --post_joint conv_fuse \
#  --skip_encoder
#  --y_lora "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_depthanythingv2_color_rank64_randtrans_trigger/y_lora" \

#  --noise_offset 0.1 \
#  --resume_from_checkpoint "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint/checkpoint-13000"
#  --resume_from_checkpoint latest
#  --train_data_dir="/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/data/open-images/train-200k/train_dataset_200k.json" \
#  --dataset_type="json" \