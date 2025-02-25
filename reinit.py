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