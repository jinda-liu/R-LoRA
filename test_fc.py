import torch
import random
import math
import os
import re
from tqdm import tqdm
from collections import Counter
from datasets import Dataset
import pandas as pd
from typing import List, Union
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer

option_pattern = r"\((\w)\)"
def is_number(s):
    """检查字符串是否为整数或浮点数"""
    return re.match(r"^[-+]?(\d+|\d*\.\d+)$", s.strip()) is not None

def extract_last_option(text):
    # 从文本中提取括号内的选项例如 (A), (B)
    pattern1 = r"\((\w)\)"
    pattern2 = r"\b([A-Z])\b"
    matches = re.findall(pattern1, text)
    matches2 = re.findall(pattern2, text)
    if matches:
        return matches[-1].strip("()")
    elif matches2:
        return matches2[-1]

    return ""

def extract_number_from_text(text):
    """
    从文本中提取第一个数字（整数或浮点数），并将其转换为浮点数。
    如果未找到数字，返回 None。
    """
    # 正则表达式匹配整数或浮点数
    number_pattern = re.compile(r"[-+]?\d*\.?\d+")  # 匹配整数或浮点数
    match = number_pattern.search(text)
    if match:
        try:
            return float(match.group())  # 转换为浮点数
        except ValueError:
            return None
    return None


def is_option_format(text):
    # 检查文本是否包含选项形式，例如 (A), (B)
    pattern1 = r"\((\w)\)"
    pattern2 = r"\b([A-Z])\b"
    if re.search(pattern1, text):
        return True
    elif re.search(pattern2, text):
        return True
    return False

def preprocess_function(examples, tokenizer):
    """
    standard preprocess function for dataset
    """
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(
        inputs,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenizer(
        targets,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


def load_dataset(example_inputs, example_outputs, tokenizer):
    # add empty string if example_outputs is None
    if example_outputs is None:
        example_outputs = [""] * len(example_inputs)
    df = [
        {"input": example_inputs[i], "output": example_outputs[i]}
        for i in range(len(example_inputs))
    ]
    dataset = Dataset.from_pandas(pd.DataFrame(df))
    preprocess_func_with_tokenizer = partial(preprocess_function, tokenizer=tokenizer)
    processed_datasets = dataset.map(
        preprocess_func_with_tokenizer,
        batched=True,
        num_proc=1,
        desc="Running tokenizer on dataset",
    )
    return processed_datasets
    
def extract_last_answer(output):
    # 找到所有 A: 的位置
    a_indices = [m.start() for m in re.finditer(r'Answer:', output)]
    
    if not a_indices:
        return None  # 如果没有找到 A:，返回 None
    
    # 取最后一个 A: 的位置
    last_a_index = a_indices[-1]
    
    # 提取最后一个 A: 之后的内容
    last_answer = output[last_a_index + 7:].strip()  # +2 是为了跳过 "Answer:"
    # if len(last_answer) > 50:
    #     print("error_output", output)
    return last_answer

def accuracy_score(outputs, ground_truths):
    OPTIONS = {0:"A", 1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "H"}
    correct = 0
    total = 0
    labels = []
    preds = []
    if is_number(ground_truths[0]):
        print("number (int/float)")
        for output, truth in zip(outputs, ground_truths):
            try:
                # 从输出文本中提取数字
                output = extract_number_from_text(output)
                truth = float(truth)  # 真实值已经是数字，直接转换
                
                # 如果成功提取到数字，则进行比较
                if output is not None and math.isclose(output, truth, rel_tol=1e-9):
                    correct += 1
            except ValueError:
                # 转换失败时跳过
                print(f"Invalid number format: output={output}, truth={truth}")
            total += 1
            preds.append(output if output is not None else 0)
            labels.append(truth)

    elif is_option_format(ground_truths[0]):  # option
        print("options")
        for output, truth in zip(outputs, ground_truths):
            output_op = extract_last_option(output)
            if output_op == "" and output.isdigit():
                # print("output is digit")
                output_num = int(output)
                try:
                    output = OPTIONS[output_num]
                except:
                    output = output_op
            else:
                output = output_op
            # print(f"Output: {output}, Truth: {truth}")    
            truth = extract_last_option(truth)
            output = output.strip().lower().replace(".", "") 
            truth = truth.strip().lower().replace(".", "")
            if output == truth:
                correct += 1
            total += 1
            preds.append(output)
            labels.append(truth)
    else:  # text
        print("text")
        for output, truth in zip(outputs, ground_truths):
            output = output.strip().lower().replace(".", "")
            truth = truth.strip().lower().replace(".", "")
            if output == truth:
                correct += 1
            total += 1
            preds.append(output)
            labels.append(truth)

    # for i, (preds, labels) in enumerate(zip(preds, labels)):
    #     print(f"Pred: {preds}, Label: {labels}", end="\t")

    accuracy = correct / total * 100  if total > 0 else 0
    return accuracy, correct, total

    
def inference(example_inputs: List[str],
                      model: AutoModelForCausalLM,
                      tokenizer: AutoTokenizer,
                      batch_size: int,
                      example_outputs: List[str]=None):


    example_predictions = []
    # 设置 tokenizer 的 padding_side 为 'left'
    tokenizer.padding_side = 'left'

    # 如果 tokenizer 没有 pad_token，则设置为 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # process dataset
    dataset = load_dataset(example_inputs, example_outputs, tokenizer)
    # use gpu if available
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = model.to(device)
    device = model.device

    for i in range(0, len(dataset["input"]), batch_size):
        inputs = tokenizer(
            dataset["input"][i : i + batch_size],
            max_length=2048,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        
        outputs = model.generate(
            input_ids=inputs["input_ids"], max_new_tokens=256,
            attention_mask=inputs['attention_mask'],
            pad_token_id=tokenizer.pad_token_id  # 显式指定 pad_token_id
        )
        outputs = tokenizer.batch_decode(
            outputs.to("cpu"), skip_special_tokens=True
        )
        pred = [extract_last_answer(output) for output in outputs]
        # print("pred:", pred)
        example_predictions.extend(pred)
        # break
    # print("Labels:", example_outputs[0])
    # print("Predictions:", example_predictions)
    if example_outputs is not None:
        task_perf, correct, total = accuracy_score(example_predictions, example_outputs)
    else:
        task_perf, correct, total = None, 0, 0
        
    return example_predictions, task_perf, correct, total

def inference2(example_inputs: List[str],
                      model: AutoModelForCausalLM,
                      tokenizer: AutoTokenizer,
                      batch_size: int,
                      example_outputs: List[str]=None):


    example_predictions = []
    # 设置 tokenizer 的 padding_side 为 'left'
    tokenizer.padding_side = 'left'

    # 如果 tokenizer 没有 pad_token，则设置为 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # process dataset
    dataset = load_dataset(example_inputs, example_outputs, tokenizer)
    # use gpu if available
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = model.to(device)
    device = model.device

    inputs = tokenizer(
        dataset["input"],
        max_length=2048,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    
    # 分批次处理
    with torch.no_grad():
        for i in range(0, len(inputs['input_ids']), batch_size):
            batch = {
                'input_ids': inputs['input_ids'][i:i+batch_size],
                'attention_mask': inputs['attention_mask'][i:i+batch_size]
            }
            
            outputs = model.generate(
                **batch,
                pad_token_id=tokenizer.pad_token_id  # 显式指定 pad_token_id
            )
            outputs = tokenizer.batch_decode(
                outputs.to("cpu"), skip_special_tokens=True
            )
            pred = [extract_last_answer(output) for output in outputs]
            # print("pred:", pred)
            example_predictions.extend(pred)
        # break
    # print("Labels:", example_outputs[0])
    # print("Predictions:", example_predictions)
    if example_outputs is not None:
        task_perf, correct, total = accuracy_score(example_predictions, example_outputs)
    else:
        task_perf, correct, total = None, 0, 0
        
    return example_predictions, task_perf, correct, total

def parse_model_output(task, output_text):
    """
    从模型输出中提取预测结果。
    假设输出格式为："Answer: <预测结果>"
    """

    if "Answer:" in output_text:
        # 提取 "Answer:" 后面的部分
        return output_text.split("Answer:")[-1].strip()
    if "Result:" in output_text:
        # 提取 "Answer:" 后面的部分
        return output_text.split("Result:")[-1].strip()
    if "answer:" in output_text:
        # 提取 "Answer:" 后面的部分
        return output_text.split("answer:")[-1].strip()
    if "result:" in output_text:
        # 提取 "Answer:" 后面的部分
        return output_text.split("result:")[-1].strip()
    return output_text  # 如果格式不匹配，返回 None


def extract_number(text):
    # 定义正则表达式模式
    pattern = r'oxed\{(\d+)\}'
    pattern2 = r'####\s*(\d+)'
    pattern3 = r'answer is (\d+)'
    # 查找匹配项
    match = re.search(pattern, text)
    # match2 = re.search(pattern2, text)
    if match:
        result = match.group(1)
        print('box ', result)
        # return int(result)
        return result
        
    match2 = re.search(pattern2, text)
    if match2:
        result = match2.group(1)
        # print('# ', result)
        # return int(result)
        return result
    # print('no number',text)
    return text


def extract_num(text):
    # Regex pattern to find the number following '####'
    pattern = r'####\s*(\d+)'
    # Using re.search to find the first match
    match = re.search(pattern, text)
    if match:
        result = match.group(1)
        print(result)
    else:
        print(text)
        result = ""
    try:
        return int(result.replace(",", ""))
    except:
        print(f"'{result}' can't be converted")
        return 0


def extract_choice(sentence: str) -> str:
    # 去除句子两端的空格
    sentence = sentence.strip()
    
    # 提取 "Answer:" 之后的部分
    if "Answer:" in sentence:
        answer_part = sentence.split("Answer:")[-1].strip()
    else:
        answer_part = sentence
    # print(answer_part)
    # 提取第一个单词或字符串片段，并转换为小写
    pred_answer = answer_part.split()[0].lower()
    # print()
    # print(pred_answer)
    # 根据数据集类型，使用正则表达式进一步提取标签
    pattern = r"""
        true|false|          # boolq 数据集
        solution1|solution2| # piqa 数据集
        answer1|answer2|     # social_i_qa, ARC, openbookqa 数据集
        answer3|answer4|     # 同上
        answer5|             # 同上
        ending1|ending2|     # hellaswag 数据集
        ending3|ending4|     # 同上
        option1|option2      # winogrande 数据集
        yes|no      # winogrande 数据集
    """
    # 使用 re.VERBOSE 忽略空格和注释
    match = re.search(pattern, pred_answer, re.VERBOSE)
    
    # 如果匹配成功，返回匹配的标签；否则返回空字符串
    if match:
        return match.group(0)
    else:
        return pred_answer


def single_test(model, test_ds, tokenizer, device, filename=None):
    for name, dataset in test_ds.items():
        print(name)
        results = []  # 用于存储结果
        results.append(f"Task {name}\n")  # 添加任务名称
        correct_count = 0  # 正确预测的数量
        total_count = 0
        idx = 0
        preds = []
        labels = []
        # samples = test_ds[name].select(range(10))
        # for sample in samples:
        for sample in tqdm(dataset, desc=f"Testing {name}"):
            label = sample['y']
            input_text = sample['x']
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # 生成输出
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],  # 显式传递 attention_mask
                pad_token_id=tokenizer.pad_token_id
            )
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred = parse_model_output(name, pred_text)
            if name == "gsm8k":
                pred = extract_number(pred)
                # label = extract_number(label)

            preds.append(pred)
            labels.append(label)
            # 更新准确率
            if pred == label:
                correct_count += 1
            total_count += 1


            if (idx + 1) % 50 == 0:
                accuracy = correct_count / total_count
                print(f"{idx+1} Accuracy: {accuracy:.2%}")
                results.append(f"{idx+1} Accuracy: {accuracy:.2%}\n")  # 将结果存入变量

            idx += 1
        preds.append(pred)
        labels.append(label)
        for i, (preds, labels) in enumerate(zip(preds, labels)):
            print(f"Pred: {preds}, Label: {labels}", end = "\t")

        # 最终准确率
        accuracy = correct_count / total_count
        print(f"{idx} Accuracy: {accuracy:.2%}")
        results.append(f"{idx} Accuracy: {accuracy:.2%}\n")  # 将最终结果存入变量
        results.append("===\n")  # 添加分隔符
        # for i, (preds, labels) in enumerate(zip(preds, labels)):
        #     print(f"Pred: {preds}, Label: {labels}", end="\t")
        # 一次性写入文件
        with open(filename, "a") as file:
            file.writelines(results)  # 将结果写入文件

def single_test_mt(model, test_ds, tokenizer, device, filename=None):
    accs = []
    correct_count = 0
    total_count = 0
    for name, dataset in test_ds.items():
        print(name)
        results = []  # 用于存储结果
        results.append(f"Task {name}\n")  # 添加任务名称
        preds = []
        labels = []
        # samples = test_ds[name].select(range(10))
        # for sample in samples:
        for sample in tqdm(dataset, desc=f"Testing {name}"):
            label = sample['y']
            input_text = sample['x']
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # 生成输出
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],  # 显式传递 attention_mask
                pad_token_id=tokenizer.pad_token_id
            )
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred = parse_model_output(name, pred_text)
            if name == "gsm8k":
                # pred = extract_number(pred)
                label = extract_number(label)

            preds.append(pred)
            labels.append(label)
        if name == "gsm8k":
            print(preds[:10])
            print(labels[:10])
        # for i, (preds, labels) in enumerate(zip(preds, labels)):
        #     print(f"Pred: {preds}, Label: {labels}", end = "\t")
        
        accuracy, correct, total = accuracy_score(preds, labels)
        accs.append(accuracy)
        correct_count += correct
        total_count += total

        # 最终准确率
        # accuracy = correct_count / total_count
        print(f"Task {name} Accuracy: {accuracy:.2%}")
        results.append(f"Task {name} Accuracy: {accuracy:.2%}\n")  # 将最终结果存入变量
        results.append("===\n")  # 添加分隔符
        # for i, (preds, labels) in enumerate(zip(preds, labels)):
        #     print(f"Pred: {preds}, Label: {labels}", end="\t")
        # 一次性写入文件
        with open(filename, "a") as file:
            file.writelines(results)  # 将结果写入文件
    avg_accuracy = sum(accs) / len(accs)
    overall_accuracy_percentage = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"Average accuracy : {avg_accuracy:.2f}%")
    print(f"Overall accuracy : {overall_accuracy_percentage:.2f}%")
    with open(filename, "a") as file:
        file.write(f"Average accuracy : {avg_accuracy:.2f}%\n")
        file.write(f"Overall accuracy : {overall_accuracy_percentage:.2f}%\n")


def batch_test_mt(model, test_ds, tokenizer, device, batch_size=16, filename=None):
    accs = []
    correct_count = 0
    total_count = 0
    for name, dataset in test_ds.items():
        preds = []
        labels = []
        results = []
        # 将数据集转换为列表以便分批处理
        dataset_list = [dataset[i] for i in range(len(dataset))]
        
        # 分批处理数据
        for batch_idx in tqdm(range(0, len(dataset_list), batch_size), desc=f"Testing {name}"):
            batch_samples = dataset_list[batch_idx:batch_idx + batch_size]
            
            # 准备批量输入
            input_texts = [sample['x'] for sample in batch_samples]
            label_batch = [sample['y'] for sample in batch_samples]
            
            # 批量编码（自动填充和截断）
            inputs = tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            # 批量生成（启用内存优化选项）
            try:
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=128,  # 根据任务调整生成长度
                    use_cache=True,      # 启用缓存加速
                    do_sample=False      # 关闭采样加速生成
                )
            except RuntimeError as e:
                print(f"生成时出现错误: {str(e)}")
                continue
            
            # 批量解码
            pred_texts = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # 处理每个样本的预测结果
            for pred_text, label in zip(pred_texts, label_batch):
                # 任务特定的后处理
                processed_pred = parse_model_output(name, pred_text.strip())
                processed_label = label.strip()
                
                if name == "gsm8k":
                    processed_pred = extract_number(processed_pred)
                    processed_label = extract_number(processed_label)
                
                preds.append(processed_pred)
                labels.append(processed_label)

        # 计算任务准确率
        task_accuracy, task_correct, task_total = accuracy_score(preds, labels)
        accs.append(task_accuracy)
        correct_count += task_correct
        total_count += task_total
        
        print(f"Task {name} Accuracy: {task_accuracy:.2%}")
        results.append(f"Task {name} Accuracy: {task_accuracy:.2%}\n")  # 将最终结果存入变量
        results.append("===\n")  # 添加分隔符
        # for i, (preds, labels) in enumerate(zip(preds, labels)):
        #     print(f"Pred: {preds}, Label: {labels}", end="\t")
        # 一次性写入文件
        with open(filename, "a") as file:
            file.writelines(results)  # 将结果写入文件

    avg_accuracy = sum(accs) / len(accs)
    overall_accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"Average accuracy : {avg_accuracy:.2f}%")
    print(f"Overall accuracy : {overall_accuracy:.2f}%")
    with open(filename, "a") as file:
        file.write(f"Average accuracy : {avg_accuracy:.2f}%\n")
        file.write(f"Overall accuracy : {overall_accuracy:.2f}%\n")


    return {
        "average_accuracy": avg_accuracy,
        "overall_accuracy": overall_accuracy,
        "task_accuracies": dict(zip(test_ds.keys(), accs))
    }

def batch_test(model, test_ds, tokenizer, device, batch_size=5, filename=None):
    for name, dataset in test_ds.items():
        print(name)
        dataset = dataset.select(range(2*batch_size))
        results = [f"Task {name}\n"]
        correct_count = 0  # 正确预测的数量
        total_count = 0
        iter = 0
        for i in range(0, len(dataset), batch_size):
            batch_indices = range(i, min(i + batch_size, len(dataset)))
            batch = dataset.select(batch_indices)
            labels = [sample['y'] for sample in batch]
            input_texts = [sample['x'] for sample in batch]
            
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model.generate(inputs['input_ids'], max_length=512)
        
            pred_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            print(pred_texts)
            preds = [extract_choice(pred_text) for pred_text in pred_texts]
            print(preds)
            for pred, label in zip(preds, labels):
                if pred == label:
                    correct_count += 1
                total_count += 1
            
            if (iter) % 50 == 0:
                accuracy = correct_count / total_count
                print(f"Iter {iter} Accuracy: {accuracy:.2%}")
                results.append(f"Iter {iter} Accuracy: {accuracy:.2%}\n")
            
            iter += 1
        
        accuracy = correct_count / total_count
        print(f"Iter {iter} Accuracy: {accuracy:.2%}")
        results.append(f"Iter {iter} Accuracy: {accuracy:.2%}\n")
        print("===")
        
        with open(filename, "a") as file:
            file.writelines(results)