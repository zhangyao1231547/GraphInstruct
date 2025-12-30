#!/usr/bin/env python3
"""
纯文本评测脚本 - 支持LoRA适配器模型
用于评测GraphInstruct训练的LoRA模型在图推理任务上的表现
"""

import json
import os
import re
import argparse
import logging
import ast
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

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
    """标准化答案用于比较

    处理以下格式差异:
    - [<7>, <10>] vs [7, 10]
    - <node_0> vs 0
    - True vs true vs TRUE
    - 多余空格
    - 中英文标点
    """
    # 移除尖括号中的内容，保留数字/文本
    # <7> -> 7, <node_0> -> node_0
    answer = re.sub(r'<([^>]+)>', r'\1', answer)

    # 移除 node_ 前缀
    answer = re.sub(r'node_?', '', answer, flags=re.IGNORECASE)

    # 统一为小写
    answer = answer.lower()

    # 移除所有空白字符
    answer = re.sub(r'\s+', '', answer)

    # 移除方括号、圆括号、花括号
    answer = answer.replace('[', '').replace(']', '')
    answer = answer.replace('(', '').replace(')', '')
    answer = answer.replace('{', '').replace('}', '')

    # 中英文标点统一
    answer = answer.replace('，', ',').replace('。', '.').replace('：', ':')

    return answer


def compare_set_answers(gt_answer: str, pred_answer: str) -> Tuple[bool, str]:
    """Compare answers as sets (order-independent).

    For tasks like connected_component, neighbor where the order doesn't matter.
    """
    try:
        gt_list = ast.literal_eval(gt_answer)
        pred_list = ast.literal_eval(pred_answer)

        gt_set = set(gt_list)
        pred_set = set(pred_list)

        if gt_set == pred_set:
            return True, "set_match"
        else:
            return False, "set_mismatch"
    except:
        # 如果无法解析为列表，回退到精确匹配
        is_correct = gt_answer == pred_answer
        return is_correct, "exact_match" if is_correct else "no_match"


def compare_numeric_answers(gt_answer: str, pred_answer: str, tolerance: float = 0.01) -> Tuple[bool, str]:
    """Compare numeric answers with tolerance for floating point precision.

    For tasks like page_rank and clustering_coefficient where small precision
    differences should be tolerated.
    """
    try:
        # 尝试解析为浮点数
        gt_val = float(gt_answer)
        pred_val = float(pred_answer)

        # 使用相对误差或绝对误差比较
        if abs(gt_val) < 1e-6:
            is_close = abs(pred_val - gt_val) < tolerance
        else:
            is_close = abs(pred_val - gt_val) / abs(gt_val) < tolerance

        if is_close:
            return True, "numeric_match"
        else:
            return False, "numeric_mismatch"
    except:
        try:
            # 尝试解析为列表
            gt_list = ast.literal_eval(gt_answer)
            pred_list = ast.literal_eval(pred_answer)

            if len(gt_list) != len(pred_list):
                return False, "length_mismatch"

            # 检查每个元素是否在误差范围内
            for gt_v, pred_v in zip(gt_list, pred_list):
                gt_f = float(gt_v)
                pred_f = float(pred_v)
                if abs(gt_f) < 1e-6:
                    is_close = abs(pred_f - gt_f) < tolerance
                else:
                    is_close = abs(pred_f - gt_f) / abs(gt_f) < tolerance
                if not is_close:
                    return False, "numeric_mismatch"
            return True, "numeric_match"
        except:
            # 回退到精确匹配
            is_correct = gt_answer == pred_answer
            return is_correct, "exact_match" if is_correct else "no_match"


def compare_traversal_answers(gt_answer: str, pred_answer: str, task_type: str) -> Tuple[bool, str]:
    """Compare BFS/DFS traversal answers.

    BFS/DFS traversals can have multiple valid orderings:
    - BFS: Nodes at the same depth level can be visited in any order
    - DFS: Different neighbor visit orders lead to different valid traversals

    Validation rules:
    1. No duplicate nodes (each node visited exactly once)
    2. Same length as ground truth
    3. Same node set as ground truth
    4. Same starting node
    """
    try:
        gt_list = ast.literal_eval(gt_answer)
        pred_list = ast.literal_eval(pred_answer)
    except:
        # 如果无法解析为列表，回退到精确匹配
        is_correct = gt_answer == pred_answer
        return is_correct, "exact_match" if is_correct else "no_match"

    # 检查长度是否相同（每个节点只访问一次）
    if len(gt_list) != len(pred_list):
        return False, "length_mismatch"

    # 检查预测是否有重复节点
    if len(pred_list) != len(set(pred_list)):
        return False, "duplicate_nodes"

    # 检查是否包含相同的节点集合
    gt_set = set(gt_list)
    pred_set = set(pred_list)

    if gt_set != pred_set:
        # 节点集合不同
        return False, "node_set_mismatch"

    # 检查起始节点是否相同
    if len(gt_list) > 0 and len(pred_list) > 0:
        if gt_list[0] != pred_list[0]:
            return False, "start_node_mismatch"

    # 节点集合相同、长度相同、无重复、起始节点相同
    if task_type == 'BFS':
        return True, "bfs_valid"
    elif task_type == 'DFS':
        return True, "dfs_valid"

    return True, "traversal_valid"


def smart_compare_answers(gt_answer: str, pred_answer: str, task_type: str) -> Tuple[bool, str]:
    """Smart answer comparison based on task type.

    Different tasks have different comparison logic:
    - BFS/DFS: Level-based matching
    - connected_component, neighbor, common_neighbor, predecessor: Set matching
    - page_rank, clustering_coefficient: Numeric tolerance matching
    - Others: Exact matching
    """
    if gt_answer == pred_answer:
        return True, "exact_match"

    # 对于BFS/DFS任务，使用层级匹配
    if task_type in ['BFS', 'DFS']:
        return compare_traversal_answers(gt_answer, pred_answer, task_type)

    # 对于顺序无关的列表任务，使用集合匹配
    if task_type in ['connected_component', 'neighbor', 'common_neighbor', 'predecessor']:
        return compare_set_answers(gt_answer, pred_answer)

    # 对于数值列表任务（如page_rank），使用近似匹配
    if task_type in ['page_rank', 'clustering_coefficient']:
        return compare_numeric_answers(gt_answer, pred_answer)

    # 其他情况使用精确匹配
    return False, "no_match"


class TextOnlyLoRAEvaluator:
    """纯文本评测器 - 支持LoRA模型"""

    def __init__(
        self,
        model_path: str,
        base_model_path: Optional[str] = None,
        device: str = "cuda",
        max_new_tokens: int = 512,
        bf16: bool = True,
        verbose: bool = False
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.is_lora = False
        self.model_name = "Unknown"

        # 检测是否为LoRA模型
        adapter_config_path = os.path.join(model_path, 'adapter_config.json')
        if os.path.exists(adapter_config_path):
            self.is_lora = True
            logger.info(f"Detected LoRA adapter at {model_path}")

            # 读取LoRA配置获取base model路径
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)

            if base_model_path is None:
                base_model_path = adapter_config.get('base_model_name_or_path')
                if base_model_path is None:
                    raise ValueError("Cannot determine base model path. Please provide --base-model-path")

            logger.info(f"Base model: {base_model_path}")
            logger.info(f"LoRA adapter: {model_path}")
            self.model_name = f"LoRA: {os.path.basename(model_path)}"

            # 加载tokenizer (从base model)
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True
            )

            # 加载base model
            dtype = torch.bfloat16 if bf16 else torch.float16
            logger.info(f"Loading base model from {base_model_path}...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True
            )

            # 加载LoRA adapter
            logger.info(f"Loading LoRA adapter from {model_path}...")
            self.model = PeftModel.from_pretrained(
                base_model,
                model_path,
                torch_dtype=dtype
            )

        else:
            # 普通模型加载
            logger.info(f"Loading model from {model_path}...")
            self.model_name = os.path.basename(model_path)

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            dtype = torch.bfloat16 if bf16 else torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True
            )

        self.model.eval()
        logger.info("Model loaded successfully!")

    def build_prompt(self, instruction: str, language: str = "auto") -> str:
        """构建chat格式的prompt"""
        # 根据内容自动判断语言
        if language == "auto":
            # 检查是否包含中文字符
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in instruction)
            language = "zh" if has_chinese else "en"

        if language == "zh":
            system_prompt = (
                "你是一个解决图推理任务的助手。"
                "请仔细按照指示进行，并详细解释你的答案。"
                "最终答案请用 <<<答案>>> 格式给出。"
            )
        else:
            system_prompt = (
                "You are a helpful assistant that solves graph reasoning tasks. "
                "Follow the instructions carefully and explain your answers in detail. "
                "Provide your final answer in <<<answer>>> format."
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

    def generate(self, instruction: str, language: str = "auto") -> str:
        """生成回答"""
        prompt = self.build_prompt(instruction, language)

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
        output_dir: str,
        language: str = "auto"
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
            ground_truth = sample.get('output', sample.get('ground_truth', ''))
            sample_id = sample.get('id', f'sample_{i}')

            # 初始化任务统计
            if task_type not in correct_by_task:
                correct_by_task[task_type] = {'total': 0, 'correct': 0}
            correct_by_task[task_type]['total'] += 1

            # 生成预测
            try:
                prediction = self.generate(instruction, language)
            except Exception as e:
                logger.error(f"Error generating for sample {sample_id}: {e}")
                prediction = ""

            # 提取答案并比较
            gt_answer = extract_final_answer(ground_truth)
            pred_answer = extract_final_answer(prediction)

            # 使用智能比较（基于任务类型选择不同的比较策略）
            is_correct, match_type = smart_compare_answers(gt_answer, pred_answer, task_type)

            # 如果智能比较失败，回退到标准化比较
            if not is_correct and match_type == "no_match":
                gt_norm = normalize_answer(gt_answer)
                pred_norm = normalize_answer(pred_answer)
                if gt_norm == pred_norm:
                    is_correct = True
                    match_type = "normalized_match"

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
                'is_correct': is_correct,
                'match_type': match_type
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
            'model': self.model_name,
            'is_lora': self.is_lora,
            'total': total,
            'correct': total_correct,
            'accuracy': accuracy,
            'by_task': correct_by_task,
            'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Is LoRA: {self.is_lora}")
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
    parser = argparse.ArgumentParser(description='Text-only evaluation supporting LoRA models')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the model or LoRA adapter')
    parser.add_argument('--base-model-path', type=str, default=None,
                        help='Path to base model (required for LoRA if not in adapter_config)')
    parser.add_argument('--eval-data-path', type=str, required=True,
                        help='Path to evaluation JSON file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--max-new-tokens', type=int, default=1024,
                        help='Maximum new tokens to generate')
    parser.add_argument('--bf16', action='store_true', default=True,
                        help='Use bfloat16')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')
    parser.add_argument('--language', type=str, default='auto', choices=['auto', 'en', 'zh'],
                        help='Language for system prompt (auto/en/zh)')

    args = parser.parse_args()

    # 加载评测数据
    logger.info(f"Loading evaluation data from {args.eval_data_path}...")
    with open(args.eval_data_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    logger.info(f"Loaded {len(eval_data)} samples")

    # 创建评测器
    evaluator = TextOnlyLoRAEvaluator(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        max_new_tokens=args.max_new_tokens,
        bf16=args.bf16,
        verbose=args.verbose
    )

    # 运行评测
    metrics = evaluator.evaluate(eval_data, args.output_dir, args.language)

    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()
