import torch
import math
from tqdm import tqdm
from typing import List, Dict  # 类型提示，用于在函数定义中指定参数和返回值的类型
from peft.tuners.lora import Linear as LoraLinear


@torch.no_grad()
def reinit_lora_modules(name, module, init_config, **kwargs):
    r"""
    Reinitialize the lora model with the given configuration.
    """
    m_B, n_B = getattr(module, f"lora_B{0}").weight.shape
    b_dim = max(m_B, n_B)
    m_A, n_A = module.lora_A.weight.shape
    lora_r = min(m_A, n_A)
    if init_config.rank_stablization == True:
        module.scaling = module.lora_alpha / math.sqrt(lora_r)

    if init_config.init_mode == "simple":
        match init_config.lora_A_init:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_A.weight, mean=0.0, std=init_config.lora_A_std
                )
            case "kaiming":
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                torch.nn.init.kaiming_uniform_(module.lora_A.weight, a=math.sqrt(5))
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_A.weight, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_A.weight)
            case "zeros":
                torch.nn.init.zeros_(module.lora_A.weight)
            case "unit":
                a_dim = max(m_A, n_A)
                torch.nn.init.normal_(
                    module.lora_A.weight, mean=0.0, std=1.0 / (a_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_A.weight)
            case _:
                raise ValueError(f"Unknown lora_A initialization: {init_config.lora_A}")
                    
        for i in range(module.lora_num):
            lora_B = getattr(module, f"lora_B{i}")
            match init_config.lora_B_init:
                case "gaussian":
                    torch.nn.init.normal_(
                        lora_B.weight, mean=0.0, std=init_config.lora_B_std
                    )
                case "kaiming":
                    torch.nn.init.kaiming_normal_(lora_B.weight)
                case "fan_out_kaiming":
                    torch.nn.init.kaiming_normal_(
                        lora_B.weight, mode="fan_out"
                    )
                case "xavier":
                    torch.nn.init.xavier_normal_(lora_B.weight)
                case "zeros":
                    torch.nn.init.zeros_(lora_B.weight)
                case "unit":
                    torch.nn.init.normal_(
                        lora_B.weight, mean=0.0, std=1.0 / (b_dim**0.5)
                    )
                case "orthogonal":
                    torch.nn.init.orthogonal_(lora_B.weight)
                case _:
                    raise ValueError(f"Unknown lora_B initialization: {init_config.lora_B}")
            
            if init_config.init_scale == "stable":
                gamma = init_config.stable_gamma
                module.lora_A.weight.data *= (m_A**0.25) / gamma**0.5
                for i in range(module.lora_num):
                    lora_B = getattr(module, f"lora_B{i}")
                    lora_B.weight.data *= (n_B**0.25) / gamma**0.5

    elif init_config.init_mode == "gradient":
        # named_grad is a dict where keys are dataset names and values are gradients
        named_grads_list = kwargs["named_grads_list"]
        
        # Initialize lora_A using the average gradient across all datasets
        avg_grad = torch.stack(named_grads_list).mean(dim=0)  # Average gradients
        U_avg, S_avg, V_avg = torch.svd_lowrank(avg_grad.cuda().float(), q=4 * lora_r, niter=4)
        V_avg = V_avg.T

        # Initialize lora_A using the average gradient
        if init_config.direction == "ArBr":
            A = V_avg[1 : 2 * lora_r : 2, :]
        elif init_config.direction == "A2rBr":
            A = V_avg[lora_r : 2 * lora_r, :]
        elif init_config.direction == "ArB2r":
            A = V_avg[:lora_r, :]

        # Apply scaling to lora_A
        if init_config.init_scale == "gd":
            A = A / module.scaling
        elif init_config.init_scale == "stable":
            gamma = init_config.stable_gamma
            A = A * (m**0.25) / gamma**0.5

        # Set lora_A
        module.lora_A.weight = torch.nn.Parameter(A.contiguous().cuda())

        # Initialize each lora_B using the corresponding dataset's gradient
        for i, grads in enumerate(named_grads_list):
            if i >= module.lora_num:
                break  # Ensure we don't exceed the number of lora_B modules
            U, S, V = torch.svd_lowrank(grads.cuda().float(), q=4 * lora_r, niter=4)
            V = V.T

            # Get the corresponding lora_B module
            lora_B = getattr(module, f"lora_B{i}")

            # Set direction for lora_B
            if init_config.direction == "ArBr":
                B = U[:, 0 : 2 * lora_r : 2]
            elif init_config.direction == "A2rBr":
                B = U[:, :lora_r]
            elif init_config.direction == "ArB2r":
                B = U[:, lora_r : 2 * lora_r]

            # Apply scaling to lora_B
            if init_config.init_scale == "gd":
                B = B / module.scaling
            elif init_config.init_scale == "stable":
                gamma = init_config.stable_gamma
                B = B * (m**0.25) / gamma**0.5

            # Set lora_B
            lora_B.weight = torch.nn.Parameter(B.contiguous().cuda())

    elif init_config.init_mode == "svd":
        U, S, V = torch.svd_lowrank(module.weight.float(), q=4 * lora_r, niter=4)
        V = V.T
        m, n = module.weight.shape
        if init_config.init_scale == "default":
            S = S / module.scaling
            module.lora_B.weight = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])).contiguous()
            )
            module.lora_A.weight = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])).T.contiguous()
            )
        elif init_config.init_scale == "stable":
            gamma = init_config.stable_gamma
            module.lora_B.weight = torch.nn.Parameter(
                (U[:, :lora_r] * (m**0.25) / gamma**0.5).contiguous()
            )
            module.lora_A.weight = torch.nn.Parameter(
                (V[:lora_r, :] * (n**0.25) / gamma**0.5).contiguous()
            )
        elif init_config.init_scale == "unit":
            module.lora_B.weight = torch.nn.Parameter(
                (U[:, :lora_r]).contiguous()
            )
            module.lora_A.weight = torch.nn.Parameter(
                (V[:lora_r, :]).contiguous()
            )
        elif init_config.init_scale == "normalized":
            S_sum = S[:lora_r].sum()
            module.lora_B.weight = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])/torch.sqrt(S_sum)*lora_r**0.5).contiguous()
            )
            module.lora_A.weight = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])/torch.sqrt(S_sum)*lora_r**0.5).T.contiguous()
            )
    
    with torch.no_grad():
        # if "torch_dtype" in init_config:
        if hasattr(init_config, 'torch_dtype'):
            
            if init_config.torch_dtype == "bfloat16":
                module.lora_route.weight.data = module.lora_route.weight.data.to(torch.bfloat16)
                module.lora_A.weight.data = module.lora_A.weight.data.to(torch.bfloat16)
                for i in range(module.lora_num):
                    lora_B = getattr(module, f"lora_B{i}")
                    lora_B.weight.data = lora_B.weight.data.to(torch.bfloat16)
           
            elif init_config.torch_dtype == "float32":
                module.lora_route.weight.data = module.lora_route.weight.data.to(torch.float32)
                module.lora_A.weight.data = module.lora_A.weight.data.to(torch.float32)
                for i in range(module.lora_num):
                    lora_B = getattr(module, f"lora_B{i}")
                    lora_B.weight.data = lora_B.weight.data.to(torch.float32)
        
        #If lora_A@lora_B is not zero, 
        #then we need to subtract lora_A@lora_B from the original weight matrix
        
        if init_config.lora_B_init != "zero":
            # Initialize offset as zero
            offset = torch.zeros_like(module.weight.data)

            # Calculate offset as the average of (A @ B) for all B matrices
            for i in range(module.lora_num):
                lora_B = getattr(module, f"lora_B{i}")
                offset += (lora_B.weight @ module.lora_A.weight).to(module.weight.data.device)

            # Average the offset
            offset /= module.lora_num

            # Apply scaling factor
            offset *= module.scaling
            hasattr(init_config, 'norm_clip')
            # Handle norm_clip for numerical stability
            if hasattr(init_config, 'norm_clip') and init_config.norm_clip:
                ratio = torch.max(torch.abs(module.weight.data)) / torch.max(torch.abs(offset))
                if ratio < 1:
                    offset *= ratio
                    module.lora_A.weight.data *= ratio**0.5
                    for i in range(module.lora_num):
                        lora_B = getattr(module, f"lora_B{i}")
                        lora_B.weight.data *= ratio**0.5

            # Subtract the offset from the original weight matrix
            try:
                module.weight.data -= offset
            except:
                breakpoint()


def reinit_lora(model, init_config, **kwargs):
    r"""
    Reinitialize the lora model with the given configuration.
    """
    init_params = [
         'rank_stablization', 'lora_A_init', 'lora_B_init',
        'init_mode', 'init_scale', 'stable_gamma'
    ]
    for param in init_params:
        if hasattr(init_config, param):
            print(f"{param}: {getattr(init_config, param)}")
    for name, module in tqdm(
        model.named_modules(),
        desc="Reinitializing Lora",
        total=len(list(model.named_modules())),
    ):
        if isinstance(module, LoraLinear):
            reinit_lora_modules(name, module, init_config, **kwargs)

    return model


# PyTorch的register_hook 允许在梯度计算完成后、但在梯度实际应用于更新之前，插入自定义处理函数
# 对每个参数计算完梯度后，将梯度记录到record_dict并转移到cpu，然后清空梯度，释放显存
def get_record_gradient_hook(model, record_dict):
    def record_gradient_hook(grad):
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if n not in record_dict:
                    record_dict[n] = p.grad.cpu()
                else:
                    record_dict[n] += p.grad.cpu()
                p.grad = None
        return grad

    return record_gradient_hook


def estimate_gradient(
    model, dataset, batch_size: int = 4
) -> Dict[str, List[torch.Tensor]]:
    # 函数返回一个字典，字典的键是字符串类型，值是 torch.Tensor 类型的列表
    r"""
    Estimate the gradient of the model on the given dataset
    """

    model.train()
    named_grads = {}
    hooks = []

    # 为所有参数注册梯度hook
    for name, param in model.named_parameters():
        hook = param.register_hook(get_record_gradient_hook(model, named_grads))
        hooks.append(hook)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    num = 0
    for batch in tqdm(dataloader, desc="Estimating gradient"):
        num += 1
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        outputs.loss.backward()
        get_record_gradient_hook(model, named_grads)(None)  # get gradient of last layer
        # make sure the gradient is cleared
        for n, p in model.named_parameters():
            if p.grad is not None:
                p.grad = None
    # 计算平均梯度
    for n, g in named_grads.items():
        named_grads[n] /= num
    # 移除所有hook 避免内存泄漏
    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()
    return named_grads