import os
from transformers import AutoModelForCausalLM
from peft import PeftModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

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

B_matrices_by_module = {mod: [] for mod in module}
for key, matrix in B_matrices.items():
    for mod in module:
        if mod in key:
            B_matrices_by_module[mod].append(matrix)

layer_num = 36
head_num = 3


# 对每个模块的 B 矩阵进行 tsne 降维和可视化
for mod in module:
    B_matrices_flattened = [matrix.flatten() for matrix in B_matrices_by_module[mod]]
    B_matrices_flattened = np.array(B_matrices_flattened)
    
    tsne = TSNE(n_components=2, random_state=42)
    B_matrices_tsne = tsne.fit_transform(B_matrices_flattened)
    
    plt.figure(figsize=(10, 8))
    colors = ['r', 'g', 'b']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd', '|', '_', '.', ',', '1', '2', '3', '4']
    
    for i in range(head_num):  # 假设有 24 层
        # 计算当前组的点的索引范围
        start_idx = i * head_num
        end_idx = start_idx + head_num
        
        # 绘制当前组的点
        for j in range(head_num):
            idx = start_idx + j
            plt.scatter(B_matrices_tsne[idx, 0], B_matrices_tsne[idx, 1], 
                        color=colors[j], marker='o', 
                        label=f'Layer {i+1} B{j}' if i == 0 and j == 0 else "")
        
        # 计算当前组的中心位置
        group_center_x = np.mean(B_matrices_tsne[start_idx:end_idx, 0])
        group_center_y = np.mean(B_matrices_tsne[start_idx:end_idx, 1])
        
        # 在中心位置添加组编号
        plt.text(group_center_x, group_center_y, f'{i+1}', 
                 fontsize=12, ha='center', va='center', 
                 )
        
    plt.title(f'TSNE Visualization of {mod} B Matrices')
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    # plt.legend()
    plt.savefig(f'b_tsne_{experiment_params}_{mod}.png')
    # plt.show()