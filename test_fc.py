import math
import re
from tqdm import tqdm
from datasets import Dataset
import pandas as pd
from typing import List, Union
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer

def is_number(s):
    return re.match(r"^[-+]?(\d+|\d*\.\d+)$", s.strip()) is not None

def extract_last_option(text):
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
    number_pattern = re.compile(r"[-+]?\d*\.?\d+")
    match = number_pattern.search(text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def is_option_format(text):
    # check option label such as, (A), (B)
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
    a_indices = [m.start() for m in re.finditer(r'Answer:', output)]
    
    if not a_indices:
        return None 
    
    last_a_index = a_indices[-1]
    
    last_answer = output[last_a_index + 7:].strip()  # +7 to jump over "Answer:"
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
                truth = float(truth)
                
                if output is not None and math.isclose(output, truth, rel_tol=1e-9):
                    correct += 1
            except ValueError:
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
    tokenizer.padding_side = 'left'

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
            pad_token_id=tokenizer.pad_token_id
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
    extract label
    """

    if "Answer:" in output_text:
        return output_text.split("Answer:")[-1].strip()
    if "Result:" in output_text:
        return output_text.split("Result:")[-1].strip()
    if "answer:" in output_text:
        return output_text.split("answer:")[-1].strip()
    if "result:" in output_text:
        return output_text.split("result:")[-1].strip()
    return output_text 


def extract_number(text):
    pattern = r'oxed\{(\d+)\}'
    pattern2 = r'####\s*(\d+)'
    pattern3 = r'answer is (\d+)'
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
    pred_answer = answer_part.split()[0].lower()
    # print()
    # print(pred_answer)
    pattern = r"""
        true|false|          # boolq
        solution1|solution2| # piqa
        answer1|answer2|     # social_i_qa, ARC, openbookqa
        answer3|answer4|    
        answer5|             
        ending1|ending2|     # hellaswag
        ending3|ending4|
        option1|option2      # winogrande
        yes|no      # winogrande
    """
    # 使用 re.VERBOSE 忽略空格和注释
    match = re.search(pattern, pred_answer, re.VERBOSE)
    
    if match:
        return match.group(0)
    else:
        return pred_answer


def single_test(model, test_ds, tokenizer, device, filename=None):
    for name, dataset in test_ds.items():
        print(name)
        results = []
        results.append(f"Task {name}\n")
        correct_count = 0 
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


            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pad_token_id=tokenizer.pad_token_id
            )
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred = parse_model_output(name, pred_text)
            if name == "gsm8k":
                pred = extract_number(pred)
                # label = extract_number(label)

            preds.append(pred)
            labels.append(label)
            if pred == label:
                correct_count += 1
            total_count += 1


            if (idx + 1) % 50 == 0:
                accuracy = correct_count / total_count
                print(f"{idx+1} Accuracy: {accuracy:.2%}")
                results.append(f"{idx+1} Accuracy: {accuracy:.2%}\n")

            idx += 1
        preds.append(pred)
        labels.append(label)
        for i, (preds, labels) in enumerate(zip(preds, labels)):
            print(f"Pred: {preds}, Label: {labels}", end = "\t")

        # 最终准确率
        accuracy = correct_count / total_count
        print(f"{idx} Accuracy: {accuracy:.2%}")
        results.append(f"{idx} Accuracy: {accuracy:.2%}\n")
        results.append("===\n") 
        # for i, (preds, labels) in enumerate(zip(preds, labels)):
        #     print(f"Pred: {preds}, Label: {labels}", end="\t")
        with open(filename, "a") as file:
            file.writelines(results)

def single_test_mt(model, test_ds, tokenizer, device, filename=None):
    accs = []
    correct_count = 0
    total_count = 0
    for name, dataset in test_ds.items():
        print(name)
        results = [] 
        results.append(f"Task {name}\n") 
        preds = []
        labels = []
        # samples = test_ds[name].select(range(10))
        # for sample in samples:
        for sample in tqdm(dataset, desc=f"Testing {name}"):
            label = sample['y']
            input_text = sample['x']
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pad_token_id=tokenizer.pad_token_id
            )
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred = parse_model_output(name, pred_text)
            if name == "gsm8k":
                # pred = extract_number(pred)
                label = extract_number(label)

            preds.append(pred)
            labels.append(label)

        # for i, (preds, labels) in enumerate(zip(preds, labels)):
        #     print(f"Pred: {preds}, Label: {labels}", end = "\t")
        
        accuracy, correct, total = accuracy_score(preds, labels)
        accs.append(accuracy)
        correct_count += correct
        total_count += total

        # accuracy = correct_count / total_count
        print(f"Task {name} Accuracy: {accuracy:.2%}")
        results.append(f"Task {name} Accuracy: {accuracy:.2%}\n")
        results.append("===\n")
        # for i, (preds, labels) in enumerate(zip(preds, labels)):
        #     print(f"Pred: {preds}, Label: {labels}", end="\t")

        with open(filename, "a") as file:
            file.writelines(results)
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

        dataset_list = [dataset[i] for i in range(len(dataset))]
        
        for batch_idx in tqdm(range(0, len(dataset_list), batch_size), desc=f"Testing {name}"):
            batch_samples = dataset_list[batch_idx:batch_idx + batch_size]
            
            input_texts = [sample['x'] for sample in batch_samples]
            label_batch = [sample['y'] for sample in batch_samples]
            
            inputs = tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            try:
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=128,
                    use_cache=True,
                    do_sample=False
                )
            except RuntimeError as e:
                print(f"error: {str(e)}")
                continue
            
            pred_texts = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            for pred_text, label in zip(pred_texts, label_batch):
                processed_pred = parse_model_output(name, pred_text.strip())
                processed_label = label.strip()
                
                if name == "gsm8k":
                    processed_pred = extract_number(processed_pred)
                    processed_label = extract_number(processed_label)
                
                preds.append(processed_pred)
                labels.append(processed_label)


        task_accuracy, task_correct, task_total = accuracy_score(preds, labels)
        accs.append(task_accuracy)
        correct_count += task_correct
        total_count += task_total
        
        print(f"Task {name} Accuracy: {task_accuracy:.2%}")
        results.append(f"Task {name} Accuracy: {task_accuracy:.2%}\n")
        results.append("===\n") 
        # for i, (preds, labels) in enumerate(zip(preds, labels)):
        #     print(f"Pred: {preds}, Label: {labels}", end="\t")
        with open(filename, "a") as file:
            file.writelines(results) 

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
