import torch
import random
import numpy as np
from collections import Counter
from datasets import Dataset, DatasetDict, load_dataset, get_dataset_config_names, concatenate_datasets, load_from_disk
from tqdm import TqdmMonitorWarning

def set_seed_self(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def check_random_seeds():
    print("PyTorch random seed:", torch.initial_seed())
    if torch.cuda.is_available():
        print("CUDA random seed:", torch.cuda.initial_seed())

    print("Python random seed:", random.getstate()[1][0])

    print("NumPy random seed:", np.random.get_state()[1][0])
    
def load_sst2():
    dataset = load_dataset("glue", "sst2", cache_dir='data')
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
    dataset = load_dataset("glue", "cola", cache_dir='data')
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
    dataset = load_dataset("glue", "qqp", cache_dir='data')
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
    dataset = load_dataset("glue", "mrpc", cache_dir='data')
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
    dataset = load_dataset("glue", "mnli", cache_dir='data')
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
    dataset = load_dataset("glue", "qnli", cache_dir='data')
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
    combined_dataset = combined_dataset.shuffle(seed=42)
    # print(type(combined_dataset))
    # resplit
    train_size = int(0.95 * len(combined_dataset))
    # indices = random.sample(range(len(combined_dataset)), train_size)
    train_set = combined_dataset.select(range(train_size))
    test_set = combined_dataset.select(range(train_size, len(combined_dataset)))
    return train_set, validation_set, test_set


def load_boolq():
    dataset = load_dataset("boolq", cache_dir='data')
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
    dataset = load_dataset("piqa", cache_dir='data', trust_remote_code=True)
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
    dataset = load_dataset("social_i_qa", cache_dir='data', trust_remote_code=True)
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
    dataset = load_dataset("hellaswag", cache_dir='data', trust_remote_code=True)
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
        dataset = load_dataset("winogrande", subset, cache_dir='data', trust_remote_code=True)
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
    dataset = load_dataset("meta-math/MetaMathQA", cache_dir='data', trust_remote_code=True)
    
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
    
    train_set = formatted_dataset["train"]
    
    return train_set


def load_commongen():
    dataset = load_dataset("allenai/common_gen", cache_dir='data', trust_remote_code=True)
    
    instruction_template = "Given several concepts (i.e., nouns or verbs), write a short and simple sentence that contains *all* the required words. The sentence should describe a common scene in daily life, and the concepts should be used in a natural way.\n\n### Concepts:\n{concepts}\n\n### Sentence:"
    
    formatted_dataset = dataset.map(
        lambda e: {
            "x": instruction_template.format(concepts=", ".join(e["concepts"])),
            "y": e["target"]
        }
    )
    
    train_set = formatted_dataset["train"]
    validation_set = formatted_dataset["validation"]
    
    return train_set, validation_set, validation_set


def load_dart():
    dataset = load_dataset("gem/dart", cache_dir='data', trust_remote_code=True)
    instruction_template = "Given the following RDF triplets, write a detailed description that incorporates the information naturally and comprehensively.\n### RDF Triplets:\n{triplets}\n### Description:"
    
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
    
    train_set = formatted_dataset["train"]
    validation_set = formatted_dataset["validation"]
    
    return train_set, validation_set, validation_set


def load_cosmosqa():
    dataset = load_dataset("cosmos_qa", cache_dir='data', trust_remote_code=True)
    
    instruction = "Please read the context carefully and answer the question according to the context: "
    
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["context"]}\nQuestion: {e["question"]}\nOption1: {e["answer0"]}\nOption2: {e["answer1"]}\nOption3: {e["answer2"]}\nOption4: {e["answer3"]}\nAnswer format: option1/option2/option3/option4.\n Answer: ',
            "y": f'option{str(int(e["label"]) + 1)}',
        }
    )
    
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    
    return train_set, validation_set, validation_set


def load_obqa():
    dataset = load_dataset("allenai/openbookqa", "main", cache_dir='data', trust_remote_code=True)

    instruction = "Complete the sentence in the most reasonable way by choosing the best option: "

    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["question_stem"]} '
                 f'A: {e["choices"]["text"][0]} B: {e["choices"]["text"][1]} '
                 f'C: {e["choices"]["text"][2]} D: {e["choices"]["text"][3]} '
                 f'Answer format: A/B/C/D. Answer: ',
            "y": e["answerKey"],
        }
    )

    train_set = dataset["train"] 
    validation_set = dataset["validation"]
    test_set = dataset["test"]

    return train_set, validation_set, test_set


def load_arc():
    subsets = ["ARC-Challenge", "ARC-Easy"]
    splits = ["train", "validation", "test"]
    datasets = {split: [] for split in splits}  # 存储所有数据划分
    
    def process_dataset_split(dataset, split_name, instruction):
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
        dataset = load_dataset("ai2_arc", subset, cache_dir='data', trust_remote_code=True)
        
        for split in splits:
            datasets[split].append(process_dataset_split(dataset, split, instruction))

    # 分别返回训练集、验证集和测试集
    train_set = concatenate_datasets(datasets["train"])
    validation_set = concatenate_datasets(datasets["validation"])
    test_set = concatenate_datasets(datasets["test"])

    return train_set, validation_set, test_set


def load_nq():
    # 加载 natural-questions 数据集
    dataset = load_dataset("Sentence-transformers/natural-questions", split="train[:13000]", cache_dir='data', trust_remote_code=True)
    
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
    dataset = load_dataset("gsm8k", "main", split="test", cache_dir='data')
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



def check_label_distribution(dataset, dataset_name):
    if 'label' in dataset.column_names:
        label_counts = Counter(dataset['label'])
    elif 'y' in dataset.column_names:
        label_counts = Counter(dataset['y'])
    print(f"{dataset_name} 标签分布: {label_counts}")



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