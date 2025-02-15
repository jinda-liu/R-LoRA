import os
from transformers import AutoModelForCausalLM
from peft import PeftModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

model_path="/root/autodl-tmp/qwen/Qwen2.5-0.5B-Instruct"
lora_path="/root/autodl-tmp/HydraLoRA/HydraLoRA/output/Qwen-0.5-mt_trydata_bfloat16_drop0.1_seed41/checkpoint-16000"
experiment_params = os.path.basename(os.path.dirname(lora_path))
# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(model_path)

# 加载 HydraLoRA 适配器
model = PeftModel.from_pretrained(base_model, lora_path)

print(model)
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

# # 提取 A 矩阵和 B 矩阵
# A_matrix = hydralora_params["lora_A.weight"]  # 假设 A 矩阵的名称为 "lora_A.weight"
# B_matrices = {k: v for k, v in hydralora_params.items() if "lora_B" in k}  # 提取所有 B 矩阵

# B_matrix_list = [B_matrix.cpu().numpy().flatten() for B_matrix in B_matrices.values()]

# similarity_matrix = cosine_similarity(B_matrix_list)
# plt.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
# plt.colorbar()
# plt.title("Cosine Similarity Between B Matrices")
# plt.show()

# 使用 t-SNE 对 B 矩阵进行降维可视化
# B_matrix_stack = np.vstack(B_matrix_list)
# tsne = TSNE(n_components=2, random_state=42)
# B_matrix_tsne = tsne.fit_transform(B_matrix_stack)

# plt.scatter(B_matrix_tsne[:, 0], B_matrix_tsne[:, 1], c=range(len(B_matrix_list)), cmap='viridis')
# plt.colorbar(label="B Matrix Index")
# plt.title("t-SNE Visualization of B Matrices")
# plt.show()

# 将 B 矩阵按 module 分类
B_matrices_by_module = {mod: [] for mod in module}
for key, matrix in B_matrices.items():
    for mod in module:
        if mod in key:
            B_matrices_by_module[mod].append(matrix)

# 确保每个模块有 24 层的三个 B 矩阵
for mod in module:
    assert len(B_matrices_by_module[mod]) == 36 * 3, f"{mod} does not have 24 layers of 3 B matrices"

# 对每个模块的 B 矩阵进行 tsne 降维和可视化
for mod in module:
    B_matrices_flattened = [matrix.flatten() for matrix in B_matrices_by_module[mod]]
    B_matrices_flattened = np.array(B_matrices_flattened)
    
    tsne = TSNE(n_components=2, random_state=42)
    B_matrices_tsne = tsne.fit_transform(B_matrices_flattened)
    
    plt.figure(figsize=(10, 8))
    colors = ['r', 'g', 'b']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd', '|', '_', '.', ',', '1', '2', '3', '4']
    
    # for i in range(24):
    #     for j in range(3):
    #         idx = i * 3 + j
    #         plt.scatter(B_matrices_tsne[idx, 0], B_matrices_tsne[idx, 1], 
    #                     color=colors[j], marker=markers[i % len(markers)], 
    #                     label=f'Layer {i+1} B{j}')
    for i in range(24):  # 假设有 24 层
        # 计算当前组的点的索引范围
        start_idx = i * 3
        end_idx = start_idx + 3
        
        # 绘制当前组的点
        for j in range(3):
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