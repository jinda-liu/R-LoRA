# R-LoRA: Random Initialization of Multi-Head LoRA for Multi-Task Learning

This repository contains the source code and related resources for [R-LoRA](https://arxiv.org/abs/2502.15455). The project aims to enhance multi-task adaptation for Multi-Head LoRA via Multi-Head randomization. The code is provided to facilitate reproducibility and to allow others to build upon this work.

### Abstract

Fine-tuning large language models (LLMs) is prohibitively expensive in terms of computational and memory costs. Low-rank Adaptation (LoRA), as one of the most popular parameter-efficient fine-tuning (PEFT) methods, offers a cost-effective alternative by approximating the model changes $\Delta W \in \mathbb{R}^{m \times n}$ through the product of down-projection matrix $A \in \mathbb{R}^{m \times r}$ and head matrix $B \in \mathbb{R}^{r \times n}$, where $r \ll \min(m, n)$. In real-world scenarios, LLMs are fine-tuned on data from multiple domains to perform tasks across various fields, embodying multi-task learning (MTL). LoRA often underperforms in such complex scenarios. To enhance LoRA's capability in multi-task learning, we propose R-LoRA, which incorporates Multi-Head Randomization. Multi-Head Randomization diversifies the head matrices through Multi-Head Random Initialization and Multi-Head Dropout, enabling more efficient learning of task-specific features while maintaining shared knowledge representation. Extensive experiments demonstrate that R-LoRA is better at capturing task-specific knowledge, thereby improving performance in multi-task scenarios.

### Motivation

In the Multi-Head architecture, the parameter similarity among head matrices remains high, hindering task-specific knowledge learning and slowing convergence speed. This is due to zero initialization of head matrices B, leading to similar update directions. To address this, we use multi-head randomization in R-LoRA, combining random initialization and multi-head dropout to diversify starting points and inputs, thereby improving task-specific learning.

<img src="figure\framework.png" alt="framework" style="zoom:50%;" />

<img src="D:\A_code&research\picture\cs_hyd.png" alt="cs_hyd" style="zoom:50%;" />

To address this, we use multi-head randomization in R-LoRA, combining random initialization and multi-head dropout to diversify starting points and inputs, thereby improving task-specific learning. This approach enables LLMs to better learn task-specific knowledge by breaking the symmetry of initial parameters and diversifying optimization trajectories.

### Install

```sh
https://github.com/jinda-liu/R-LoRA.git
cd R-LoRA
conda create -n rlora python=3.10
conda activate rlora
pip install -r requirements.txt
```

### Project Structure

```sh
peft
transformers
analysis_cos.py # analyzing the the parameter similarity among head matrices
analysis_tsne.py
configuration.py # get the configuration of training
data_load.py # download the data
eval_bbh.py 
ft_rlora.py # train with R-LoRA
test_fc # function for the evaluation
fine-tuning.sh
```

### Train

```shell
bash fine-tuning.sh
```

### Test

```shel
python eval_bbh.py
```

### Overview

## <img src="figure\architecture.png" alt="architecture" style="zoom: 25%;" />

## <img src="figure\initialization.png" alt="initialization" style="zoom: 25%;" />



### Experiments

<img src="figure\experiment.png" alt="experiment" style="zoom: 33%;" />

### Ablation Study

<img src="figure\ablation.png" alt="ablation" style="zoom: 33%;" />

### Citation

If you find this project useful in your research or work, please consider citing it:

```sh
@misc{liu2025rlorarandominitializationmultihead,
      title={R-LoRA: Random Initialization of Multi-Head LoRA for Multi-Task Learning}, 
      author={Jinda Liu and Yi Chang and Yuan Wu},
      year={2025},
      eprint={2502.15455},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.15455}, 
}
```

### Reference

The code refers to the [HydraLoRA](https://github.com/Clin0212/HydraLoRA), [LoRA-GA](https://github.com/Outsider565/LoRA-GA)

