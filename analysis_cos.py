import os
from transformers import AutoModelForCausalLM
from peft import PeftModel
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_path="path_to_model"
lora_path="path_to_lora"

experiment_params = os.path.basename(os.path.dirname(lora_path))

base_model = AutoModelForCausalLM.from_pretrained(model_path)

model = PeftModel.from_pretrained(base_model, lora_path)

# print(model)
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


# for name in hydralora_params.keys():
#     print(name)

# extract matrix
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

# 
B_matrices_by_module = {mod: [] for mod in module}
for key, matrix in B_matrices.items():
    for mod in module:
        if mod in key:
            B_matrices_by_module[mod].append(matrix)


layer_num = 36
head_num = 3

cosine_similarity_means = {mod: [] for mod in module}

for mod in module:
    for i in range(layer_num):
        layer_matrices = B_matrices_by_module[mod][i*head_num:(i+1)*head_num]
        layer_matrices_flattened = [matrix.flatten() for matrix in layer_matrices]
        similarity_matrix = cosine_similarity(layer_matrices_flattened)
        
        # 计算相似度矩阵的非0元素均值
        non_zero_elements = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        mean_similarity = np.mean(non_zero_elements)
        cosine_similarity_means[mod].append(mean_similarity)
colors = {"up_proj": "b", "down_proj": "g", "gate_proj": "r"}


plt.figure(figsize=(12, 8))
for mod in module:
    plt.plot(range(0, layer_num), cosine_similarity_means[mod], color=colors[mod], marker='o', label=f'{mod} Cosine Similarity Mean')
    overall_mean = np.mean(cosine_similarity_means[mod])
    plt.axhline(y=overall_mean, color=colors[mod], linestyle='--', label=f'{mod} Overall Mean')

plt.title('Cosine Similarity Mean of B Matrices by Layer')
plt.xlabel('Layer')
plt.ylabel('Cosine Similarity Mean')
plt.legend()
plt.grid(True)
plt.savefig(f'analysis/b_cs_{experiment_params}.png')
# plt.show() 
