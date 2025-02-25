from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import json
import torch
import time
from typing import List, Union
import re

from test_fc import *
    
def accuracy_score(outputs, ground_truths):
    correct = 0
    total = 0
    labels = []
    preds = []
    if is_number(ground_truths[0]):
        print("number (int/float)")
        for output, truth in zip(outputs, ground_truths):
            try:
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
            output = extract_last_option(output)
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

    for i, (pres, labels) in enumerate(zip(preds, labels)):
        print(f"Pred: {pres}, Label: {labels}", end="\t")

    accuracy = correct / total * 100 if total > 0 else 0
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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    for i in range(0, len(dataset["input"]), batch_size):
        inputs = tokenizer(
            dataset["input"][i : i + batch_size],
            max_length=2048,
            return_tensors="pt",
            padding=True,
        ).to(device)
        outputs = model.generate(
            input_ids=inputs["input_ids"], max_new_tokens=256,
            attention_mask=inputs['attention_mask'])
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


def evaluate_few_shot(folder, model, tokenizer, filename=None):
    sub_dirs = os.listdir(folder)
    overall_accuracy = []
    task_results = []
    total_correct = 0
    total_count = 0
    for sub_dir in sub_dirs:
        test_file_path = os.path.join(folder, sub_dir, "few_shot.jsonl")
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            # task_inputs.append(example["context"])
            # task_outputs.append(example["completion"])
            
            # change Q:to Question: A: Answer:
            prompt = "I will give you three examples, please answer the last question based on them.\n"
            modified_context = prompt + example["context"].replace("Q:", "question:").replace("A:", "Answer:")
            task_inputs.append(modified_context)
            
            modified_completion = example["completion"].replace("A:", "\nAnswer:")
            task_outputs.append(modified_completion)
        # print("task_inputs:", task_inputs)
        # print("task_outputs:", task_outputs[0])
        print("Evaluating on task (few shot): ", sub_dir)
        _, task_perf, correct, total = inference(task_inputs,
                                         model,
                                         tokenizer,
                                         96,
                                         task_outputs)
        if task_perf is not None:
            overall_accuracy.append(task_perf)
            total_correct += correct
            total_count += total
            print(f"Accuracy for task {sub_dir}: {task_perf:.2f}%")
            
            task_results.append({
                'task_name': sub_dir,
                'accuracy': task_perf,
                'correct': correct,
                'total': total
            })


    if overall_accuracy:
        avg_accuracy = sum(overall_accuracy) / len(overall_accuracy)
        overall_accuracy_percentage = (total_correct / total_count) * 100 if total_count > 0 else 0
        print(f"Average accuracy across all zero-shot tasks: {avg_accuracy:.2f}%")
        print(f"Overall accuracy across all zero-shot tasks: {overall_accuracy_percentage:.2f}%")
    
    with open(filename, 'w') as f:
        for result in task_results:
            f.write(f"Accuracy: {result['accuracy']:.2f}%, Task: {result['task_name']}, Correct: {result['correct']}, Total: {result['total']}\n")
        
        f.write(f"\nAverage accuracy across all tasks: {avg_accuracy:.2f}%\n")
        f.write(f"Overall accuracy across all tasks: {overall_accuracy_percentage:.2f}%\n")



if __name__ == "__main__":
    start_time = time.time()
    if not os.path.exists("data_bbh"):
        # download dataset
        os.system("wget https://github.com/sail-sg/lorahub/releases/download/0.1/data_bbh.zip")
        # unzip
        os.system("unzip data_bbh.zip")
    
    # Load the model and tokenizer once
    torch_dtype = torch.bfloat16
    model_path = 'Qwen/Qwen2.5-0.5B'
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                  device_map="auto",
                                                  torch_dtype=torch_dtype,
                                                  output_hidden_states=True
                                                  )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, 
                                              use_fast=False, trust_remote_code=True)
    lora_path = 'path/to/lora'
    
    experiment_params = os.path.basename(os.path.dirname(lora_path))
    filename = f"result/bbh_{experiment_params}.txt"
    # filename = f"result/bbh_Q-B-0.5.txt"
    
    model = PeftModel.from_pretrained(model, model_id=lora_path)
    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16:
            param.data = param.data.to(torch_dtype)
    # Evaluate the model
    # evaluate_zero_shot("data_bbh", model, tokenizer)
    evaluate_few_shot("data_bbh", model, tokenizer, filename=filename)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time} seconds")