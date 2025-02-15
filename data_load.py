import torch
import random
import os
import numpy as np
from collections import Counter
from datasets import Dataset, DatasetDict, load_dataset, get_dataset_config_names, concatenate_datasets, load_from_disk
from tqdm import tqdm  # 用于显示进度条

def set_seed_self(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def check_random_seeds():
    # 检查 PyTorch 的随机种子
    print("PyTorch random seed:", torch.initial_seed())
    if torch.cuda.is_available():
        print("CUDA random seed:", torch.cuda.initial_seed())

    # 检查 Python 的随机种子
    print("Python random seed:", random.getstate()[1][0])

    # 检查 NumPy 的随机种子
    print("NumPy random seed:", np.random.get_state()[1][0])
    
def load_sst2():
    dataset = load_dataset("glue", "sst2", cache_dir='/root/autodl-tmp/data')
    instruction = "classify the sentiment of the text: "
    label_map = {0: "negative", 1: "positive", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}Is the following sentence "{e["sentence"]}"sentimently positive? Answer: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set


def load_cola():
    dataset = load_dataset("glue", "cola", cache_dir='/root/autodl-tmp/data')
    # print(type(dataset))
    instruction = "classify the grammaticality of the text: "
    label_map = {0: "unacceptable", 1: "acceptable", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}Is the following sentence "{e["sentence"]}" grammatically acceptable? Answer: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set


def load_qqp():
    dataset = load_dataset("glue", "qqp", cache_dir='/root/autodl-tmp/data')
    instruction = "classify the semantic similarity of the text: "
    label_map = {0: "different", 1: "duplicate", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}Is the following question "{e["question1"]}" essentially asking the same thing as "{e["question2"]}"? Answer: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set


def load_mrpc():
    dataset = load_dataset("glue", "mrpc", cache_dir='/root/autodl-tmp/data')
    instruction = "classify the semantic similarity of the text: "
    label_map = {0: "different", 1: "equivalent", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["sentence1"]}\n{e["sentence2"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set


def load_mnli():
    dataset = load_dataset("glue", "mnli", cache_dir='/root/autodl-tmp/data')
    instruction = "classify the semantic similarity of the text: "
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}Dose the statement "{e["premise"]}" imply that "{e["hypothesis"]}" ? Answer: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation_matched"]
    return train_set, validation_set, validation_set


def load_qnli():
    dataset = load_dataset("glue", "qnli", cache_dir='/root/autodl-tmp/data')
    instruction = "Please classify the semantic similarity of the question and the sentence: "
    label_map = {0: "entailment", 1: "not_entailment", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}Based on the statement: "{e["question"]}" dose the following sentence "{e["sentence"]}" have a definitive answer? Answer: ',
            "y": label_map[e["label"]],
        }
    )
    'Based on the statement: "{e["question"]}" dose the following sentence "{e["sentence"]}" have a definitive answer? Answer:'
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    test_set = dataset["test"]
    
    random.seed(42)
    
    combined_dataset = concatenate_datasets([train_set, test_set])
    # print(type(combined_dataset))
    # 重新采样
    combined_dataset = combined_dataset.shuffle(seed=42)
    # print(type(combined_dataset))
    # 重新划分训练集和测试集
    train_size = int(0.95 * len(combined_dataset))
    # indices = random.sample(range(len(combined_dataset)), train_size)
    train_set = combined_dataset.select(range(train_size))
    test_set = combined_dataset.select(range(train_size, len(combined_dataset)))
    return train_set, validation_set, test_set

def load_boolq():
    dataset = load_dataset("boolq", cache_dir='/root/autodl-tmp/data')
    instruction = "Please read the passage and answer the following question with true or false. "
    label_map = {True: "true", False: "false"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}\nPassage:{e["passage"]}\nQuestion:{e["question"]}\nAnswer format: true/false. Answer: ',
            "y": label_map[e["answer"]],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set

def load_piqa():
    dataset = load_dataset("piqa", cache_dir='/root/autodl-tmp/data', trust_remote_code=True)
    instruction = "Please choose the correct solution to the question: "
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["goal"]}\nSolution1: {e["sol1"]}\nSolution2: {e["sol2"]}\nAnswer format: solution1/solution2. Answer: ',
            "y": f'solution{e["label"] + 1}',
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set

def load_siqa():
    dataset = load_dataset("social_i_qa", cache_dir='/root/autodl-tmp/data', trust_remote_code=True)
    instruction = "Please choose the correct answer to the question: "
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["question"]}\nAnswer1: {e["answerA"]}\nAnswer2: {e["answerB"]}\nAnswer3: {e["answerC"]}\nAnswer format: answer1/answer2/answer3. Answer: ',
            "y": f'answer{e["label"]}',
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set

def load_hellaswag():
    dataset = load_dataset("hellaswag", cache_dir='/root/autodl-tmp/data', trust_remote_code=True)
    instruction = "Please choose the correct ending to complete the given sentence: "
    dataset = dataset.filter(lambda e: e["label"].strip() != "")

    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["activity_label"]}: {e["ctx"]}\nEnding1: {e["endings"][0]}\nEnding2: {e["endings"][1]}\nEnding3: {e["endings"][2]}\nEnding4: {e["endings"][3]}\nAnswer format: ending1/ending2/ending3/ending4. Answer: ',
            "y": f'ending{str(int(e["label"]) + 1)}',
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set

def load_winogrande():
    subsets = ["winogrande_xs", "winogrande_s","winogrande_m", "winogrande_l", "winogrande_debiased", "winogrande_xl"]
    train_set = []
    instruction = "Please choose the correct answer to fill in the blank to complete the given sentence: "
    for subset in subsets:
        dataset = load_dataset("winogrande", subset, cache_dir='/root/autodl-tmp/data', trust_remote_code=True)
        dataset = dataset.map(
            lambda e: {
                "x": f'{instruction}{e["sentence"]}\nOption1: {e["option1"]}\nOption2: {e["option2"]}\nAnswer format: option1/option2. Answer: ',
                "y": f'option{str(int(e["label"]) + 1)}',
            }
        )
        train_set.append(dataset["train"])
    train_set = concatenate_datasets(train_set)
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set

def load_metamathqa():
    # 加载 MetaMathQA 数据集
    dataset = load_dataset("meta-math/MetaMathQA", cache_dir='/root/autodl-tmp/data', trust_remote_code=True)
    
    # 定义模板
    instruction_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    # 格式化数据集
    formatted_dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}\n### Instruction:{e["query"]}\n### Response:',
            "y": e["response"]
        }
    )
    formatted_dataset = dataset.map(
        lambda e: {
            "x": instruction_template.format(instruction=e["query"]),
            "y": e["response"]
        }
    )
    
    # 分割数据集为训练集和验证集
    train_set = formatted_dataset["train"]
    # validation_set = formatted_dataset["validation"]
    
    return train_set#, validation_set, validation_set

def load_commongen():
    # 加载 common_gen 数据集
    dataset = load_dataset("allenai/common_gen", cache_dir='/root/autodl-tmp/data', trust_remote_code=True)
    
    # 定义指令模板
    instruction_template = "Given several concepts (i.e., nouns or verbs), write a short and simple sentence that contains *all* the required words. The sentence should describe a common scene in daily life, and the concepts should be used in a natural way.\n\n### Concepts:\n{concepts}\n\n### Sentence:"
    
    # 格式化数据集
    formatted_dataset = dataset.map(
        lambda e: {
            "x": instruction_template.format(concepts=", ".join(e["concepts"])),
            "y": e["target"]
        }
    )
    
    # 分割数据集为训练集和验证集
    train_set = formatted_dataset["train"]
    validation_set = formatted_dataset["validation"]
    
    return train_set, validation_set, validation_set

def load_dart():
    dataset = load_dataset("gem/dart", cache_dir='/root/autodl-tmp/data', trust_remote_code=True)
    # 定义指令模板
    instruction_template = "Given the following RDF triplets, write a detailed description that incorporates the information naturally and comprehensively.\n### RDF Triplets:\n{triplets}\n### Description:"
    
    # 格式化数据集
    formatted_dataset = dataset.map(
        lambda e: {
            "x": instruction_template.format(
                triplets="\n".join(
                    [f"{s} - {p} - {o}" for (s, p, o) in e["tripleset"]]
                )
            ),
            "y": e["target"]
        }
    )
    
    # 分割数据集为训练集和验证集
    train_set = formatted_dataset["train"]
    validation_set = formatted_dataset["validation"]
    
    return train_set, validation_set, validation_set

def load_cosmosqa():
    # 加载 cosmosqa 数据集
    dataset = load_dataset("cosmos_qa", cache_dir='/root/autodl-tmp/data', trust_remote_code=True)
    
    # 定义指令模板
    instruction = "Please read the context carefully and answer the question according to the context: "
    
    # 格式化数据集
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["context"]}\nQuestion: {e["question"]}\nOption1: {e["answer0"]}\nOption2: {e["answer1"]}\nOption3: {e["answer2"]}\nOption4: {e["answer3"]}\nAnswer format: option1/option2/option3/option4.\n Answer: ',
            "y": f'option{str(int(e["label"]) + 1)}',
        }
    )
    
    # 分割数据集为训练集和验证集
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    
    return train_set, validation_set, validation_set

def load_obqa():
    # 加载 openbookqa 数据集
    dataset = load_dataset("allenai/openbookqa", "main", cache_dir='/root/autodl-tmp/data', trust_remote_code=True)

    # 定义更合适的指令模板
    instruction = "Complete the sentence in the most reasonable way by choosing the best option: "

    # 格式化数据集（移除换行符，使选项紧凑）
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["question_stem"]} '
                 f'A: {e["choices"]["text"][0]} B: {e["choices"]["text"][1]} '
                 f'C: {e["choices"]["text"][2]} D: {e["choices"]["text"][3]} '
                 f'Answer format: A/B/C/D. Answer: ',
            "y": e["answerKey"],  # 直接使用字母标签
        }
    )

    # 训练集 = train + test，验证集 = validation
    train_set = dataset["train"] 
    validation_set = dataset["validation"]
    test_set = dataset["test"]

    return train_set, validation_set, test_set

def load_arc():
    subsets = ["ARC-Challenge", "ARC-Easy"]
    splits = ["train", "validation", "test"]
    datasets = {split: [] for split in splits}  # 存储所有数据划分
    
    def process_dataset_split(dataset, split_name, instruction):
        """ 处理数据集的指定划分（train/validation/test），统一格式 """
        return dataset[split_name].map(
            lambda e: {
            "x": f'{instruction}\nQuestion: {e["question"]}\n'
                 f'Choices: {", ".join([f"{label}) {text}" for label, text in zip(e["choices"]["label"], e["choices"]["text"])])}\n'
                 f'Answer format: A/B/C/D. Answer: ',
                "y": e["answerKey"]
            }
        )

    instruction = "Please Answer the following multiple-choice question based on given choices.\nQuestion："

    for subset in subsets:
        dataset = load_dataset("ai2_arc", subset, cache_dir='/root/autodl-tmp/data', trust_remote_code=True)
        
        for split in splits:
            datasets[split].append(process_dataset_split(dataset, split, instruction))

    # 分别返回训练集、验证集和测试集
    train_set = concatenate_datasets(datasets["train"])
    validation_set = concatenate_datasets(datasets["validation"])
    test_set = concatenate_datasets(datasets["test"])

    return train_set, validation_set, test_set

def load_nq():
    # 加载 natural-questions 数据集
    dataset = load_dataset("Sentence-transformers/natural-questions", split="train[:13000]", cache_dir='/root/autodl-tmp/data', trust_remote_code=True)
    
    # 定义指令模板
    instruction = "Please answer the following question based on the provided information: "
    
    # 格式化数据集
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}\nQuestion: {e["query"]}\nAnswer: ',
            "y": e["answer"],
        }
    )
    
    # 返回格式化后的数据集
    return dataset

def load_gsm8k():
    dataset = load_dataset("gsm8k", "main", split="test", cache_dir='/root/autodl-tmp/data')
    instruction = "Please solve the following math problem and answer a number. Question: "
    
    # The dataset appears to have 'question' and 'answer' fields based on the image.
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["question"]}\nAnswer format: number. Answer: ',  # Adding instruction before the question
            "y": e["answer"],  # Directly using the answer from the dataset
        }
    )
    
    # Split the dataset into train, validation, and test sets
    
    return test_set
def save_datasets(train_set, validation_set, test_set, task_name, save_dir):
    # 将数据集保存到指定目录
    dataset_dict = DatasetDict({
        "train": train_set,
        "validation": validation_set,
        "test": test_set
    })
    dataset_dict.save_to_disk(f"{save_dir}/{task_name}")


def save_ds_to_disk(save_dir, tasks):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载并保存每个任务的数据集
    # tasks = {
    #     "sst2": load_sst2,
    #     "cola": load_cola,
    #     "qqp": load_qqp,
    #     "mrpc": load_mrpc,
    #     "mnli": load_mnli,
    #     "qnli": load_qnli
    # }
    # tasks = {
    #     "boolq": load_boolq,
    #     "piqa": load_piqa,
    #     "siqa": load_siqa,
    #     "hellaswag": load_hellaswag,
    #     "winogrande": load_winogrande
    # }

    for task_name, load_func in tasks.items():
        print(task_name)
        train_set, validation_set, test_set = load_func()
        save_datasets(train_set, validation_set, test_set, task_name, save_dir)
        print(f"Saved {task_name} dataset to {save_dir}/{task_name}")

def check_label_distribution(dataset, dataset_name):
    if 'label' in dataset.column_names:
        label_counts = Counter(dataset['label'])
    elif 'y' in dataset.column_names:
        label_counts = Counter(dataset['y'])
    print(f"{dataset_name} 标签分布: {label_counts}")


def load_ds_from_disk(save_dir, DATASET):

    train_ds = {task: None for task in DATASET}
    test_ds = {task: None for task in DATASET}

    print('start')
    # sample_size = 8500
    # random.seed(0)
    for task_name in DATASET:
        print(task_name)
        dataset = load_from_disk(f"{save_dir}/{task_name}")
        train_set = dataset["train"]
        validation_set = dataset["validation"]
        test_set = dataset["test"]
        print(len(train_set),len(test_set))
        # indices = random.sample(range(len(train_set)), sample_size)
        # train_set = train_set.select(indices)
        # print(type(train_set), type(train_sample),len(train_sample))
        train_ds[task_name] = train_set
        test_ds[task_name] = test_set
        # print(len(train_set))
        print()
    # 检查所有数据集的标签分布
    # for name, dataset in train_ds.items():
    #     check_label_distribution(dataset, f"{name} 训练集")
    #     check_label_distribution(test_ds[name], f"{name} 测试集")
    
    return train_ds, test_ds

def preprocess_fc(examples, tokenizer):
    MAX_LENGTH = 512
    input_ids = []
    attention_mask = []
    labels = []
    
    # 批量 tokenize instruction 和 response
    inst_enc = tokenizer(examples['x'], add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
    resp_enc = tokenizer(examples['y'], add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
    
    for inst_ids, inst_mask, resp_ids in zip(inst_enc["input_ids"], inst_enc["attention_mask"], resp_enc["input_ids"]):
        ids = inst_ids + resp_ids + [tokenizer.pad_token_id]
        mask = inst_mask + [1] * (len(resp_ids) + 1)
        label = ([-100] * len(inst_ids)) + resp_ids + [tokenizer.pad_token_id]
        
        if len(ids) > MAX_LENGTH:
            ids = ids[:MAX_LENGTH]
            mask = mask[:MAX_LENGTH]
            label = label[:MAX_LENGTH]
        else:
            padding_length = MAX_LENGTH - len(ids)
            ids += [tokenizer.pad_token_id] * padding_length
            mask += [0] * padding_length
            label += [-100] * padding_length
        
        input_ids.append(ids)
        attention_mask.append(mask)
        labels.append(label)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    

def save_tokenized_datasets(train_ds, test_ds, save_dir):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    # 保存每个任务的处理后的训练集和测试集
    for task_name in train_ds.keys():
        # 保存训练集
        train_ds[task_name].save_to_disk(f"{save_dir}/{task_name}_train")
        print(f"Saved tokenized {task_name} train dataset to {save_dir}/{task_name}_train")

        # 保存测试集
        test_ds[task_name].save_to_disk(f"{save_dir}/{task_name}_test")
        print(f"Saved tokenized {task_name} test dataset to {save_dir}/{task_name}_test")

def load_tokenized_datasets(save_dir, DATASET):
    # DATASET = ["sst2", "cola", "qqp", "mnli", "qnli"]
    tokenized_train_ds = {}
    tokenized_test_ds = {}

    for task_name in DATASET:
        # 加载处理后的训练集
        tokenized_train_ds[task_name] = load_from_disk(f"{save_dir}/{task_name}_train")
        print(f"Loaded tokenized {task_name} train dataset from {save_dir}/{task_name}_train")

        # # 加载处理后的测试集
        # tokenized_test_ds[task_name] = load_from_disk(f"{save_dir}/{task_name}_test")
        # print(f"Loaded tokenized {task_name} test dataset from {save_dir}/{task_name}_test")

    return tokenized_train_ds#, tokenized_test_ds


def tokenized_dataset(train_ds, test_ds, tokenizer):
    for name, dataset in train_ds.items():
        train_ds[name] = dataset.map(
            lambda examples: preprocess_fc(examples, tokenizer),
            batched=True,
            batch_size=1000,
            remove_columns=dataset.column_names
        )
        print(f"Tokenized {name} train dataset")
        
        test_ds[name] = test_ds[name].map(
            lambda examples: preprocess_fc(examples, tokenizer),
            batched=True,
            batch_size=1000,
            remove_columns=dataset.column_names
        )
        print(f"Tokenized {name} test dataset")

    print()
    
    return train_ds, test_ds

def load_data():
    SEED = 42
    ran = random.Random(SEED)
    SAMPLE_SIZE = {"sst2": 20000, "qqp": 20000, "mnli": 10000, "qnli": 10000,
                "boolq": 10000, "obqa": 5000, "piqa": 10000, "siqa": 10000, "winogrande": 20000,
                "metamathqa":20000, "cosmosqa":30000, "commongen":10000,
                    "dart":10000, "arc": 10000, "nq": 13000}
    glue_dir = "/root/autodl-tmp/data/glue_tokenized"
    DATASET = ["sst2","qqp", "mnli", "qnli",]
    glue_ds ={task: None for task in DATASET}
    for task_name in DATASET:
        # 加载处理后的训练集
        train_set = load_from_disk(f"{glue_dir}/{task_name}_train")
        if len(train_set) > SAMPLE_SIZE[task_name]:        
            indices = ran.sample(range(len(train_set)), SAMPLE_SIZE[task_name])
            train_set = train_set.select(indices)
        glue_ds[task_name] = train_set  
        print(f"Loaded tokenized {task_name} train dataset from {glue_dir}/{task_name}_train")

    cs_dir = "/root/autodl-tmp/data/commonsense_tokenized"
    DATASET = ["boolq","piqa", "siqa", "winogrande"]
    cs_ds = {task: None for task in DATASET}
    for task_name in DATASET:
        # 加载处理后的训练集
        train_set = load_from_disk(f"{cs_dir}/{task_name}_train")
        if len(train_set) > SAMPLE_SIZE[task_name]:
            indices = ran.sample(range(len(train_set)), SAMPLE_SIZE[task_name])
            train_set = train_set.select(indices)
        cs_ds[task_name] = train_set  
        print(f"Loaded tokenized {task_name} train dataset from {cs_dir}/{task_name}_train")

    obqa_train_ds = load_from_disk("/root/autodl-tmp/data/obqa_tokenized/obqa_train")
    obqa_test_ds = load_from_disk("/root/autodl-tmp/data/obqa_tokenized/obqa_test")
    obqa_ds = concatenate_datasets([obqa_train_ds, obqa_test_ds])
    obqa_ds = {"obqa": obqa_ds}
    print(f"Loaded tokenized obqa train dataset from /root/autodl-tmp/data/obqa_tokenized/obqa_train")

    cos_ds = load_from_disk("/root/autodl-tmp/data/cosmosqa_tokenized/cosmosqa_train")
    if len(cos_ds) > SAMPLE_SIZE["cosmosqa"]:
        indices = ran.sample(range(len(cos_ds)), SAMPLE_SIZE["cosmosqa"])
        cos_ds = cos_ds.select(indices)
    cos_ds = {"cosmosqa": cos_ds}
    print(f"Loaded tokenized cosmosqa train dataset from /root/autodl-tmp/data/cosmosqa_tokenized/cosmosqa_train")

    commongen_ds = load_from_disk("/root/autodl-tmp/data/commongen_tokenized/commongen_train")
    if len(commongen_ds) > SAMPLE_SIZE["commongen"]:
        indices = ran.sample(range(len(commongen_ds)), SAMPLE_SIZE["commongen"])
        commongen_ds = commongen_ds.select(indices)
    commongen_ds = {"commongen": commongen_ds}
    print(f"Loaded tokenized commongen train dataset from /root/autodl-tmp/data/commongen_tokenized/commongen_train")

    dart_ds = load_from_disk("/root/autodl-tmp/data/dart_tokenized/dart_train")
    if len(dart_ds) > SAMPLE_SIZE["dart"]:
        indices = ran.sample(range(len(dart_ds)), SAMPLE_SIZE["dart"])
        dart_ds = dart_ds.select(indices)
    dart_ds = {"dart": dart_ds}
    print(f"Loaded tokenized dart train dataset from /root/autodl-tmp/data/dart_tokenized/dart_train")

    arc_train_ds = load_from_disk("/root/autodl-tmp/data/arc_tokenized/arc_train")
    arc_test_ds = load_from_disk("/root/autodl-tmp/data/arc_tokenized/arc_test")
    arc_ds = concatenate_datasets([arc_train_ds, arc_test_ds])
    arc_ds = {"arc": arc_ds}
    print(f"Loaded tokenized arc train dataset from /root/autodl-tmp/data/arc_tokenized/arc_train")

    metamathqa_ds = load_from_disk("/root/autodl-tmp/data/metamathqa_tokenized/metamathqa_train")
    if len(metamathqa_ds) > SAMPLE_SIZE["metamathqa"]:
        indices = ran.sample(range(len(metamathqa_ds)), SAMPLE_SIZE["metamathqa"])
        metamathqa_ds = metamathqa_ds.select(indices)
    metamathqa_ds = {"metamathqa": metamathqa_ds}
    print(f"Loaded tokenized metamathqa train dataset from /root/autodl-tmp/data/metamathqa_tokenized/metamathqa_train")

    nq_ds = load_from_disk("/root/autodl-tmp/data/nq_tokenized/nq_train")
    if len(nq_ds) > SAMPLE_SIZE["nq"]:
        indices = ran.sample(range(len(nq_ds)), SAMPLE_SIZE["nq"])
        nq_ds = nq_ds.select(indices)
    nq_ds = {"nq": nq_ds}
    print(f"Loaded tokenized nq train dataset from /root/autodl-tmp/data/nq_tokenized/nq_train")

    train_ds = {**glue_ds, **cs_ds, **obqa_ds, **cos_ds, **commongen_ds, **dart_ds,**metamathqa_ds,  **arc_ds, **nq_ds}

    for name, dataset in train_ds.items():
        print(name)
        print(dataset)


    train_datasets = concatenate_datasets([dataset for dataset in train_ds.values()])
    # train_datasets = train_datasets.shuffle(seed=SEED)
    return train_datasets

def save_glue_disk(save_dir):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载并保存每个任务的数据集
    tasks = {
        "sst2": load_sst2,
        "cola": load_cola,
        "qqp": load_qqp,
        "mrpc": load_mrpc,
        "mnli": load_mnli,
        "qnli": load_qnli
    }

    for task_name, load_func in tasks.items():
        print(task_name)
        train_set, validation_set, test_set = load_func()
        save_datasets(train_set, validation_set, test_set, task_name, save_dir)
        print(f"Saved {task_name} dataset to {save_dir}/{task_name}")


def save_common_sense_datasets(save_dir):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载并保存每个常识推理任务的数据集
    tasks = {
        "boolq": load_boolq,
        "piqa": load_piqa,
        "siqa": load_siqa,
        "hellaswag": load_hellaswag,
        "winogrande": load_winogrande
    }

    for task_name, load_func in tasks.items():
        print(task_name)
        train_set, validation_set, test_set = load_func()
        save_datasets(train_set, validation_set, test_set, task_name, save_dir)
        print(f"Saved {task_name} dataset to {save_dir}/{task_name}")

def load_glue_from_disk(save_dir, DATASET):
    
    DATASET = ["sst2", "cola", "qqp", "mnli", "qnli",]
    train_ds = {task: None for task in DATASET}
    test_ds = {task: None for task in DATASET}

    print('start')
    # sample_size = 8500
    # random.seed(0)
    for task_name in DATASET:
        print(task_name)
        dataset = load_from_disk(f"{save_dir}/{task_name}")
        train_set = dataset["train"]
        validation_set = dataset["validation"]
        test_set = dataset["test"]
        print(len(train_set),len(test_set))
        # indices = random.sample(range(len(train_set)), sample_size)
        # train_set = train_set.select(indices)
        # print(type(train_set), type(train_sample),len(train_sample))
        train_ds[task_name] = train_set
        test_ds[task_name] = test_set
        # print(len(train_set))
        print()

    # 检查所有数据集的标签分布
    for name, dataset in train_ds.items():
        # print(f"检查 {name} 数据集的标签分布")
        check_label_distribution(dataset, f"{name} 训练集")
    for name, dataset in test_ds.items():
        # print(f"检查 {name} 数据集的标签分布")
        check_label_distribution(dataset, f"{name} 训练集")

    return train_ds, test_ds


def load_common_sense_from_disk(save_dir, DATASET):
    
    train_ds = {task: None for task in DATASET}
    test_ds = {task: None for task in DATASET}

    # print('start')
    # sample_size = 8500
    # random.seed(0)
    for task_name in DATASET:
        print(task_name)
        dataset = load_from_disk(f"{save_dir}/{task_name}")
        train_set = dataset["train"]
        validation_set = dataset["validation"]
        test_set = dataset["test"]
        print(len(train_set), len(test_set))
        # indices = random.sample(range(len(train_set)), sample_size)
        # train_set = train_set.select(indices)
        train_ds[task_name] = train_set
        test_ds[task_name] = validation_set
        print()

    # 检查所有数据集的标签分布
    for name, dataset in train_ds.items():
        check_label_distribution(dataset, f"{name} 训练集")
        check_label_distribution(test_ds[name], f"{name} 测试集")

    return train_ds, test_ds


# def preprocess_fc(examples, tokenizer):
#     MAX_LENGTH = 512
#     input_ids = []
#     attention_mask = []
#     labels = []
    
#     # 批量 tokenize instruction 和 response
#     inst_enc = tokenizer(examples['x'], add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
#     resp_enc = tokenizer(examples['y'], add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
    
#     for inst_ids, inst_mask, resp_ids in zip(inst_enc["input_ids"], inst_enc["attention_mask"], resp_enc["input_ids"]):
#         # ids = inst_ids + resp_ids + [tokenizer.pad_token_id]  # 不适用于GLUE的分类任务
#         # mask = inst_mask + [1] * (len(resp_ids) + 1)
#         # 只使用 instruction 的 input_ids
#         ids = inst_ids + [tokenizer.pad_token_id]
#         mask = inst_mask + [1] * 1  # 只保留 instruction 的 mask
#         label = ([-100] * len(inst_ids)) + resp_ids + [tokenizer.pad_token_id]
        
#         # 截断到 MAX_LENGTH
#         if len(ids) > MAX_LENGTH:
#             ids = ids[:MAX_LENGTH]
#             mask = mask[:MAX_LENGTH]
#             label = label[:MAX_LENGTH]
#         if len(label) > MAX_LENGTH:
#             label = label[:MAX_LENGTH]
        
#         # 填充到 MAX_LENGTH
#         padding_length = MAX_LENGTH - len(ids)
#         ids += [tokenizer.pad_token_id] * padding_length
#         mask += [0] * padding_length
        
#         padding_length_label = MAX_LENGTH - len(label)
#         label += [-100] * padding_length_label
        
#         # 确保长度一致
#         assert len(ids) == MAX_LENGTH, f"input_ids 长度不一致: {len(ids)}"
#         assert len(mask) == MAX_LENGTH, f"attention_mask 长度不一致: {len(mask)}"
#         assert len(label) == MAX_LENGTH, f"labels 长度不一致: {len(label)}"
        
#         input_ids.append(ids)
#         attention_mask.append(mask)
#         labels.append(label)
    
#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "labels": labels
#     }