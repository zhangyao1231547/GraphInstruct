#!/usr/bin/env python3
"""
纯文本评测脚本 - 使用原始Qwen3-4B模型(无图编码)
用于评测baseline模型在图推理任务上的表现
"""

import json
import os
import re
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_final_answer(text: str) -> str:
    """提取<<<...>>>中的最终答案"""
    pattern = r'<<<(.+?)>>>'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def normalize_answer(answer: str) -> str:
    """标准化答案用于比较"""
    return answer.lower().replace(' ', '').replace('[', '').replace(']', '')


class TextOnlyEvaluator:
    """纯文本评测器"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 512,
        bf16: bool = True,
        verbose: bool = False
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose

        logger.info(f"Loading model from {model_path}...")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 加载模型
        dtype = torch.bfloat16 if bf16 else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

        logger.info("Model loaded successfully!")

    def build_prompt(self, instruction: str) -> str:
        """构建Qwen chat格式的prompt"""
        system_prompt = (
            "You are a helpful assistant that solves graph reasoning tasks. "
            "Follow the instructions carefully and explain your answers in detail."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    def generate(self, instruction: str) -> str:
        """生成回答"""
        prompt = self.build_prompt(instruction)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 解码生成的文本
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()

    def evaluate(
        self,
        eval_data: List[Dict],
        output_dir: str
    ) -> Dict:
        """运行评测"""
        os.makedirs(output_dir, exist_ok=True)

        results = []
        correct_by_task = {}
        total_correct = 0

        logger.info(f"Starting evaluation on {len(eval_data)} samples...")

        for i, sample in enumerate(tqdm(eval_data, desc="Evaluating")):
            task_type = sample.get('task_type', 'unknown')
            instruction = sample.get('instruction', '')
            ground_truth = sample.get('output', '')
            sample_id = sample.get('id', f'sample_{i}')

            # 初始化任务统计
            if task_type not in correct_by_task:
                correct_by_task[task_type] = {'total': 0, 'correct': 0}
            correct_by_task[task_type]['total'] += 1

            # 生成预测
            try:
                prediction = self.generate(instruction)
            except Exception as e:
                logger.error(f"Error generating for sample {sample_id}: {e}")
                prediction = ""

            # 提取答案并比较
            gt_answer = extract_final_answer(ground_truth)
            pred_answer = extract_final_answer(prediction)

            gt_norm = normalize_answer(gt_answer)
            pred_norm = normalize_answer(pred_answer)

            is_correct = gt_norm == pred_norm
            if is_correct:
                total_correct += 1
                correct_by_task[task_type]['correct'] += 1

            # 保存结果
            result = {
                'id': sample_id,
                'task_type': task_type,
                'instruction': instruction[:500] + '...' if len(instruction) > 500 else instruction,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'gt_answer': gt_answer,
                'pred_answer': pred_answer,
                'is_correct': is_correct
            }
            results.append(result)

            # 详细输出
            if self.verbose:
                logger.info(f"\n{'='*60}")
                logger.info(f"Sample {sample_id} (Task: {task_type})")
                logger.info(f"GT Answer: {gt_answer}")
                logger.info(f"Pred Answer: {pred_answer}")
                logger.info(f"Correct: {is_correct}")

        # 计算指标
        total = len(eval_data)
        accuracy = total_correct / total if total > 0 else 0

        metrics = {
            'model': 'Qwen3-4B-Instruct (Original)',
            'total': total,
            'correct': total_correct,
            'accuracy': accuracy,
            'exact_match': 0,  # 用于兼容报告生成器
            'partial_match': 0,
            'exact_match_rate': 0.0,
            'partial_match_rate': 0.0,
            'by_task': correct_by_task,
            'by_task_type': {}  # 用于兼容报告生成器
        }

        # 转换格式以兼容报告生成器
        for task, stats in correct_by_task.items():
            metrics['by_task_type'][task] = {
                'total': stats['total'],
                'exact_match': 0,
                'partial_match': 0,
                'exact_match_rate': 0.0,
                'partial_match_rate': 0.0
            }

        # 保存结果
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"Metrics saved to {metrics_path}")

        # 保存详细结果
        detailed_path = os.path.join(output_dir, 'detailed_results.json')
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed results saved to {detailed_path}")

        # 打印统计
        logger.info(f"\n{'='*60}")
        logger.info("Evaluation Results")
        logger.info(f"{'='*60}")
        logger.info(f"Total samples: {total}")
        logger.info(f"Correct: {total_correct}")
        logger.info(f"Accuracy: {accuracy*100:.2f}%")
        logger.info(f"\nBy task type:")

        sorted_tasks = sorted(
            correct_by_task.items(),
            key=lambda x: x[1]['correct'] / x[1]['total'] if x[1]['total'] > 0 else 0,
            reverse=True
        )

        for task, stats in sorted_tasks:
            task_acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            logger.info(f"  {task}: {stats['correct']}/{stats['total']} ({task_acc:.1f}%)")

        return metrics


def main():
    parser = argparse.ArgumentParser(description='Text-only evaluation using original Qwen model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the Qwen model')
    parser.add_argument('--eval-data-path', type=str, required=True,
                        help='Path to evaluation JSON file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--max-new-tokens', type=int, default=512,
                        help='Maximum new tokens to generate')
    parser.add_argument('--bf16', action='store_true', default=True,
                        help='Use bfloat16')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')

    args = parser.parse_args()

    # 加载评测数据
    logger.info(f"Loading evaluation data from {args.eval_data_path}...")
    with open(args.eval_data_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    logger.info(f"Loaded {len(eval_data)} samples")

    # 创建评测器
    evaluator = TextOnlyEvaluator(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        bf16=args.bf16,
        verbose=args.verbose
    )

    # 运行评测
    metrics = evaluator.evaluate(eval_data, args.output_dir)

    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()
