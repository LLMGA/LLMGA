export http_proxy=http://sys-proxy-rd-relay.byted.org:8118  https_proxy=http://sys-proxy-rd-relay.byted.org:8118  no_proxy=code.byted.org


sudo mkdir ./checkpoints
sudo chmod -R 777 ./checkpoints


deepspeed --master_port=7001 llmga/llava/train/pretrain_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./base_models/gemma-2b-it \
    --version gemma \
    --data_path ./data/llava_pretrain/images/blip_laion_cc_sbu_558k.json \
    --image_folder ./data/llava_pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llmga-gemma-pretrain/mm_projector.bin \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --image_aspect_ratio "resizesquare" \
    # --report_to wandb



# export http_proxy=http://sys-proxy-rd-relay.byted.org:8118  https_proxy=http://sys-proxy-rd-relay.byted.org:8118  no_proxy=code.byted.org

# sudo chmod -R 777 /mnt/bn/xiabinpaintv2/CVPR2024/res
# git config --global --add safe.directory /mnt/bn/xiabinpaintv2/CVPR2024/LLMGA-v1.5-v83
# MODEL_VERSION="gemma-2b-it"

# deepspeed --master_port=7001 llmga/llava/train/train_mem.py \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path /mnt/bn/xiabinpaintv2/gemma-2b-it \
#     --version gemma \
#     --data_path /mnt/bn/xiabinpaint/CVPR2024v5/LLaVA/playground/data/llava_v1_5_mix665k.json \
#     --data_path2 /mnt/bn/wyt-large-dataset/xiabin-dataset/llava16/coco \
#     --data_path3 /mnt/bn/wyt-large-dataset/xiabin-dataset/llava16/LLM-infov3 \
#     --data_path4 /mnt/bn/wyt-large-dataset/xiabin-dataset/text-data \
#     --data_path5 /mnt/bn/wyt-large-dataset/xiabin-dataset/llava16/instruction_data \
#     --image_folder /mnt/bn/inpainting-bytenas-lq/xiabin/ShareGPT4V \
#     --image_folder2 /mnt/bn/inpainting-bytenas-lq/data/inpainting/improved_aesthetics_6.25plus \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --pretrain_mm_mlp_adapter ./checkpoints/llmga-gemma-pretrain \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_output_start_end False \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir /mnt/bn/xiabinpaintv2/CVPR2024/res/LLMGA1.5-v83/llmga-$MODEL_VERSION-full-finetune \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --image_aspect_ratio "resizesquare" \