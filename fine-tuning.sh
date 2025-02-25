export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

# User-defined parameters
lr=0.0002
lora_rank=4
lora_alpha=32
lora_trainable="gate_proj,down_proj,up_proj"
lora_dropout=0.1                 
pretrained_model="Qwen/Qwen2.5-0.5B"
tokenizer_path="Qwen/Qwen2.5-0.5B"
dataset_dir="data/processed_data"
per_device_train_batch_size=6
per_device_eval_batch_size=16
gradient_accumulation_steps=16
max_seq_length=512
seed=0
output_dir="output"
lora_b_nums=5

torch_dtype="bfloat16"
lora_A="unit"
lora_B="unit"
init_scale="stable"
stable_gamma=64
init_bs=5
sample_size=8000
exp_name="${lora_A}__seed${seed}_head${lora_b_nums}"

CUDA_VISIBLE_DEVICES=0 \
python   ft_mt.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --processed_dir ${processed_dir} \
    --tokenized_dir ${tokenized_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --reinit True \
    --init_scale ${init_scale} \
    --rank_stablization True \
    --sample True \
    --use_rlora True \
    --sample_size ${sample_size} \
    --lora_A_init ${lora_A} \
    --lora_B_init ${lora_B} \
    --stable_gamma ${stable_gamma} \
    --init_bs ${init_bs} \
    --do_train \
    --do_eval \
    --seed ${seed} \
    --bf16 True\
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_total_limit 5 \
    --save_steps 1500 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir}/${exp_name} \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_nums ${lora_b_nums} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype ${torch_dtype} \
    --validation_file ${validation_file} \
    --load_in_kbits 16 \
    --overwrite_output_dir