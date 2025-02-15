import os
from transformers import AutoModelForCausalLM
from peft import PeftModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_path="/root/autodl-tmp/qwen/Qwen2.5-0.5B-Instruct"
lora_path="/root/autodl-tmp/HydraLoRA/HydraLoRA/output/Qwen-0.5-mt_reinit_bfloat16_drop0.1_seed0/checkpoint-16000"
experiment_params = os.path.basename(os.path.dirname(lora_path))
# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(model_path)

# 加载 HydraLoRA 适配器
model = PeftModel.from_pretrained(base_model, lora_path)

# print(model)
# 提取 HydraLoRA 参数
hydralora_params = model.state_dict()

def extract_lora_A_matrix(hydralora_params):
    for name, param in hydralora_params.items():
        if "lora_A" in name and "weight" in name:
            return param
    raise KeyError("lora_A.weight not found in hydralora_params")

def extract_lora_B_matrices(hydralora_params):
    B_matrices = {}
    for name, param in hydralora_params.items():
        if "lora_B" in name and "weight" in name:
            B_matrices[name] = param
    if not B_matrices:
        raise KeyError("No lora_B matrices found in hydralora_params")
    return B_matrices

def extract_matrices(hydralora_params, key):
    matrices = {}
    for name, param in hydralora_params.items():
        if key in name :
            matrices[name] = param
    if not matrices:
        raise KeyError("No lora_B matrices found in hydralora_params")
    return matrices

keys = ["lora_A", "lora_B"]

# 打印所有参数名称以检查
# for name in hydralora_params.keys():
#     print(name)

# 提取 A 矩阵和 B 矩阵
try:
    A_matrix = extract_matrices(hydralora_params, key=keys[0])
    print("A matrix extracted successfully.")
except KeyError as e:
    print(e)
# for key in A_matrix.keys():
#     print(key)

try:
    B_matrices = extract_matrices(hydralora_params, key=keys[1])
    print("B matrices extracted successfully.")
except KeyError as e:
    print(e)
# for key in B_matrices.keys():
#     print(key, B_matrices[key].shape)

module = ["up_proj", "down_proj", "gate_proj"]
lora = ["lora_A", "lora_B0", "lora_B1", "lora_B2"]

# 将 B 矩阵按 module 分类
B_matrices_by_module = {mod: [] for mod in module}
for key, matrix in B_matrices.items():
    for mod in module:
        if mod in key:
            B_matrices_by_module[mod].append(matrix)

# 确保每个模块有 24 层的三个 B 矩阵
for mod in module:
    assert len(B_matrices_by_module[mod]) == 36 * 3, f"{mod} does not have 24 layers of 3 B matrices"


# 计算每层的余弦相似度并记录均值
cosine_similarity_means = {mod: [] for mod in module}

for mod in module:
    for i in range(24):
        layer_matrices = B_matrices_by_module[mod][i*3:(i+1)*3]
        layer_matrices_flattened = [matrix.flatten() for matrix in layer_matrices]
        similarity_matrix = cosine_similarity(layer_matrices_flattened)
        
        # 计算相似度矩阵的非0元素均值
        non_zero_elements = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        mean_similarity = np.mean(non_zero_elements)
        cosine_similarity_means[mod].append(mean_similarity)
colors = {"up_proj": "b", "down_proj": "g", "gate_proj": "r"}
# 绘制折线图
plt.figure(figsize=(12, 8))
for mod in module:
    plt.plot(range(1, 25), cosine_similarity_means[mod], color=colors[mod], marker='o', label=f'{mod} Cosine Similarity Mean')
    # 计算并绘制平均值的横虚线
    overall_mean = np.mean(cosine_similarity_means[mod])
    plt.axhline(y=overall_mean, color=colors[mod], linestyle='--', label=f'{mod} Overall Mean')

plt.title('Cosine Similarity Mean of B Matrices by Layer')
plt.xlabel('Layer')
plt.ylabel('Cosine Similarity Mean')
plt.legend()
plt.grid(True)
plt.savefig(f'analysis/b_cs_{experiment_params}.png')
# plt.show() 
