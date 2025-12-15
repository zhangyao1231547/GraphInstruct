#!/usr/bin/env python3
"""
图推理模型评测脚本
支持对训练后的模型进行基准测试
"""
import json
import re
import os
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Graph Reasoning Model Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Base model path")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="LoRA adapter path (if using LoRA)")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Test dataset file (JSON)")
    parser.add_argument("--output_file", type=str, default="eval_results.json",
                        help="Output file for results")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum new tokens to generate")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (None for all)")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task names to evaluate (None for all)")
    return parser.parse_args()


def extract_answer(text: str) -> Optional[str]:
    """从模型输出中提取<<<ANSWER>>>格式的答案"""
    # 匹配 <<<...>>> 格式
    pattern = r'<<<(.+?)>>>'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()  # 取最后一个匹配
    return None


def normalize_answer(answer: str, task: str) -> Any:
    """根据任务类型标准化答案"""
    if answer is None:
        return None

    answer = answer.strip()

    # 布尔类型任务
    bool_tasks = ['connectivity', 'cycle', 'edge', 'bipartite']
    if task in bool_tasks:
        answer_lower = answer.lower()
        if 'yes' in answer_lower or 'true' in answer_lower:
            return True
        elif 'no' in answer_lower or 'false' in answer_lower:
            return False
        return answer

    # 数值类型任务
    numeric_tasks = ['degree', 'diameter', 'common_neighbor', 'maximum_flow', 'MST', 'shortest_path']
    if task in numeric_tasks:
        try:
            # 尝试提取数字
            nums = re.findall(r'-?\d+\.?\d*', answer)
            if nums:
                return float(nums[0]) if '.' in nums[0] else int(nums[0])
        except:
            pass
        return answer

    # 浮点数任务
    float_tasks = ['jaccard', 'clustering_coefficient', 'page_rank']
    if task in float_tasks:
        try:
            nums = re.findall(r'-?\d+\.?\d*', answer)
            if nums:
                return float(nums[0])
        except:
            pass
        return answer

    # 列表类型任务 (BFS, DFS, neighbor, predecessor, topological_sort, connected_component)
    list_tasks = ['BFS', 'DFS', 'neighbor', 'predecessor', 'topological_sort', 'connected_component']
    if task in list_tasks:
        try:
            # 尝试解析列表
            if answer.startswith('[') and answer.endswith(']'):
                return eval(answer)
            # 尝试提取数字列表
            nums = re.findall(r'\d+', answer)
            if nums:
                return [int(n) for n in nums]
        except:
            pass
        return answer

    return answer


def compare_answers(pred: Any, gold: Any, task: str) -> bool:
    """比较预测答案和标准答案"""
    if pred is None:
        return False

    # 标准化gold answer
    gold_normalized = normalize_answer(str(gold), task)

    # 布尔类型
    if isinstance(pred, bool) and isinstance(gold_normalized, bool):
        return pred == gold_normalized

    # 数值类型 (允许小误差)
    if isinstance(pred, (int, float)) and isinstance(gold_normalized, (int, float)):
        if task in ['jaccard', 'clustering_coefficient', 'page_rank']:
            return abs(pred - gold_normalized) < 0.01  # 浮点数允许0.01误差
        return pred == gold_normalized

    # 列表类型
    if isinstance(pred, list) and isinstance(gold_normalized, list):
        # 对于BFS/DFS等有序任务，需要完全匹配
        if task in ['BFS', 'DFS', 'topological_sort']:
            return pred == gold_normalized
        # 对于neighbor等无序任务，集合比较
        return set(pred) == set(gold_normalized)

    # 字符串比较
    return str(pred).strip().lower() == str(gold_normalized).strip().lower()


def load_model(model_path: str, lora_path: Optional[str], device: str):
    """加载模型和tokenizer"""
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )

    if lora_path:
        print(f"Loading LoRA adapter from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()  # 合并LoRA权重

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, instruction: str, max_new_tokens: int) -> str:
    """生成模型响应"""
    messages = [
        {"role": "user", "content": instruction}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # 禁用thinking模式以获得更直接的答案
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 使用greedy decoding保证可复现
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 只取生成的部分
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    return response


def main():
    args = parse_args()

    # 加载测试数据
    print(f"Loading test data from {args.test_file}...")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 过滤任务
    if args.tasks:
        target_tasks = set(args.tasks.split(','))
        test_data = [d for d in test_data if d.get('task', '') in target_tasks]
        print(f"Filtered to {len(test_data)} samples for tasks: {target_tasks}")

    # 限制样本数
    if args.num_samples:
        test_data = test_data[:args.num_samples]

    print(f"Total samples to evaluate: {len(test_data)}")

    # 加载模型
    model, tokenizer = load_model(args.model_path, args.lora_path, args.device)

    # 评测
    results = []
    task_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for sample in tqdm(test_data, desc="Evaluating"):
        instruction = sample['instruction']
        gold_output = sample['output']
        task = sample.get('task', 'unknown')

        # 从gold_output提取标准答案
        gold_answer = extract_answer(gold_output)

        # 生成预测
        pred_output = generate_response(model, tokenizer, instruction, args.max_new_tokens)
        pred_answer = extract_answer(pred_output)

        # 标准化并比较
        pred_normalized = normalize_answer(str(pred_answer) if pred_answer else "", task)
        gold_normalized = normalize_answer(str(gold_answer) if gold_answer else "", task)

        is_correct = compare_answers(pred_normalized, gold_normalized, task)

        # 记录结果
        result = {
            'task': task,
            'instruction': instruction[:200] + '...' if len(instruction) > 200 else instruction,
            'gold_answer': str(gold_answer),
            'pred_answer': str(pred_answer),
            'pred_output': pred_output[:500] + '...' if len(pred_output) > 500 else pred_output,
            'correct': is_correct
        }
        results.append(result)

        # 更新统计
        task_stats[task]['total'] += 1
        if is_correct:
            task_stats[task]['correct'] += 1

    # 计算总体准确率
    total_correct = sum(s['correct'] for s in task_stats.values())
    total_samples = sum(s['total'] for s in task_stats.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

    # 打印结果
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n{'Task':<30} {'Accuracy':<15} {'Correct/Total':<15}")
    print("-" * 60)

    for task in sorted(task_stats.keys()):
        stats = task_stats[task]
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{task:<30} {acc*100:>6.2f}%       {stats['correct']:>4}/{stats['total']:<4}")

    print("-" * 60)
    print(f"{'OVERALL':<30} {overall_accuracy*100:>6.2f}%       {total_correct:>4}/{total_samples:<4}")
    print("=" * 60)

    # 保存结果
    output = {
        'overall_accuracy': overall_accuracy,
        'task_accuracies': {
            task: {
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
                'correct': stats['correct'],
                'total': stats['total']
            }
            for task, stats in task_stats.items()
        },
        'detailed_results': results
    }

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
