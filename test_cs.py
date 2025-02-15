import torch
import random
import sys
import os
from datasets import Dataset, load_dataset, get_dataset_config_names, concatenate_datasets, interleave_datasets
from peft import LoraConfig, TaskType, PeftModel, get_peft_model
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from data_load import *
from test_fc import *
from collections import Counter
from torch.utils.data import DataLoader

torch_dtype = torch.float32
model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/qwen/Qwen2.5-0.5B',
                                              device_map="auto",
                                              torch_dtype=torch_dtype,
                                              output_hidden_states=True
                                              )

# for name, param in model.named_parameters():
#     print(f"{name}: {param.dtype}")

tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/qwen/Qwen2.5-0.5B', 
                                          use_fast=False, trust_remote_code=True,
                                          padding_side='left')
lora_path = '/root/autodl-tmp/HydraLoRA/HydraLoRA/output/Q-B-0.5-mt2_unit_nostable_seed7_5/checkpoint-416'

experiment_params = os.path.basename(os.path.dirname(lora_path))
filename = f"result/{experiment_params}.txt"
# filename = "result/try.txt"

model = PeftModel.from_pretrained(model, model_id=lora_path)

for name, param in model.named_parameters():
    if param.dtype != torch.bfloat16:
        param.data = param.data.to(torch_dtype)
        # print(f"Converted {name} to {torch.bfloat16}")
        
# invalid_params = []
# for name, parameters in model.named_parameters():
#     # logger.info(f"{name}, : {parameters.dtype}")
#     if parameters.dtype != torch_dtype:
#         invalid_params.append((name, parameters.dtype))

# if invalid_params:
#     for name, dtype in invalid_params:
#         print(f"Parameter {name} is not of type {torch_dtype}, but {dtype}.")
    # sys.exit(1)
# task = 'gsm8k'
# save_dir = "/root/autodl-tmp/data/gsm8k_processed"
# dataset = load_from_disk(f"{save_dir}")
# gsm_test_ds = {task:dataset["test"]}
save_dir = "/root/autodl-tmp/data/glue_processed"
DATASET = ["sst2","qqp", "qnli"]
_, glue_test_ds = load_ds_from_disk(save_dir, DATASET)
save_dir = "/root/autodl-tmp/data/common_sense_processed"
DATASET = ["piqa", "siqa"]
# DATASET = ["siqa"]
_, cs_test_ds = load_ds_from_disk(save_dir, DATASET)
test_ds = {**glue_test_ds, **cs_test_ds}
# test_ds = cs_test_ds

set_seed(7)

sample_size = 1000
for key in test_ds.keys():
    if len(test_ds[key]) > sample_size:
        indices = random.sample(range(len(test_ds[key])), sample_size)
        test_ds[key] = test_ds[key].select(indices)

# # 检查所有数据集的标签分布
for name, dataset in test_ds.items():
    print(name)
    # print(dataset[0])
    print(dataset)
#     check_label_distribution(dataset, f"{name} 测试集")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = model.device
# model.to(device)
model.eval()
# print(model)
print('Model loaded')


batch_size = 16
batch_test_mt(model, test_ds, tokenizer, device, batch_size, filename)
