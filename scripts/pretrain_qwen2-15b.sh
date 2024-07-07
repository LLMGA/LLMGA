
sudo mkdir ./checkpoints
sudo chmod -R 777 ./checkpoints

deepspeed --master_port=7001 llmga/llava/train/pretrain_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./base_models/Qwen2-1.5B-Instruct \
    --version qwen_2 \
    --data_path ./data/llava_pretrain/images/blip_laion_cc_sbu_558k.json \
    --image_folder ./data/llava_pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llmga-Qwen2-1.5B-pretrain/mm_projector.bin \
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




