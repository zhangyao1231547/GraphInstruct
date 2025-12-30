#!/usr/bin/env python3
"""
API云端模型评测脚本 - 支持火山引擎(豆包/DeepSeek)和阿里云(Qwen)
用于评测云端大模型在图推理任务上的表现

支持断点续传：使用 --resume 参数可以从上次中断的位置继续评测
"""

import json
import os
import re
import argparse
import logging
import time
import asyncio
import aiohttp
import ast
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm.asyncio import tqdm_asyncio
from concurrent.futures import ThreadPoolExecutor
import requests

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


def is_numeric(s: str) -> bool:
    """检查字符串是否为数值"""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def normalize_numeric(s: str, precision: int = 4) -> str:
    """标准化数值字符串，处理浮点精度问题"""
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(round(val, precision))
    except (ValueError, TypeError):
        return s


def normalize_answer(answer: str) -> str:
    """标准化答案用于比较

    处理以下格式差异:
    - [<7>, <10>] vs [7, 10]
    - <node_0> vs 0
    - True vs true vs TRUE
    - 多余空格
    - 中英文标点
    - 数值精度差异: 0.5833 vs 0.5833333333333334
    - 整数/浮点格式: 1.0 vs 1
    """
    if not answer:
        return ""

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

    # 处理数值精度问题
    if is_numeric(answer):
        answer = normalize_numeric(answer)
    else:
        # 如果是逗号分隔的列表，对每个元素进行数值标准化
        parts = answer.split(',')
        if len(parts) > 1:
            normalized_parts = []
            for part in parts:
                part = part.strip()
                if is_numeric(part):
                    normalized_parts.append(normalize_numeric(part))
                else:
                    normalized_parts.append(part)
            answer = ','.join(normalized_parts)

    return answer


def compare_set_answers(gt_answer: str, pred_answer: str) -> Tuple[bool, str]:
    """比较集合类型的答案（顺序无关）

    适用于: neighbor, common_neighbor, predecessor, connected_component
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
        # 尝试从标准化答案解析
        try:
            gt_norm = normalize_answer(gt_answer)
            pred_norm = normalize_answer(pred_answer)
            gt_parts = set(gt_norm.split(','))
            pred_parts = set(pred_norm.split(','))
            if gt_parts == pred_parts:
                return True, "set_match"
        except:
            pass
        return gt_answer == pred_answer, "exact_match" if gt_answer == pred_answer else "no_match"


def compare_numeric_answers(gt_answer: str, pred_answer: str, tolerance: float = 0.01) -> Tuple[bool, str]:
    """比较数值类型的答案（允许精度误差）

    适用于: clustering_coefficient, page_rank
    """
    try:
        gt_val = float(gt_answer)
        pred_val = float(pred_answer)

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
            return gt_answer == pred_answer, "exact_match" if gt_answer == pred_answer else "no_match"


def compare_traversal_answers(gt_answer: str, pred_answer: str, task_type: str) -> Tuple[bool, str]:
    """比较BFS/DFS遍历答案

    BFS/DFS遍历可以有多个有效顺序：
    - BFS: 同一层级的节点可以按任意顺序访问
    - DFS: 不同的邻居访问顺序导致不同的有效遍历

    验证规则：
    1. 无重复节点（每个节点只访问一次）
    2. 长度与ground truth相同
    3. 节点集合与ground truth相同
    4. 起始节点相同
    """
    try:
        gt_list = ast.literal_eval(gt_answer)
        pred_list = ast.literal_eval(pred_answer)
    except:
        # 尝试从标准化答案解析
        try:
            gt_norm = normalize_answer(gt_answer)
            pred_norm = normalize_answer(pred_answer)
            gt_list = [int(x) for x in gt_norm.split(',') if x]
            pred_list = [int(x) for x in pred_norm.split(',') if x]
        except:
            return gt_answer == pred_answer, "exact_match" if gt_answer == pred_answer else "no_match"

    # 检查长度是否相同
    if len(gt_list) != len(pred_list):
        return False, "length_mismatch"

    # 检查预测是否有重复节点
    if len(pred_list) != len(set(pred_list)):
        return False, "duplicate_nodes"

    # 检查是否包含相同的节点集合
    gt_set = set(gt_list)
    pred_set = set(pred_list)

    if gt_set != pred_set:
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


def compare_answers(gt_answer: str, pred_answer: str, task_type: str = None) -> Tuple[bool, str]:
    """智能答案比较 - 根据任务类型选择合适的比较方法

    Args:
        gt_answer: 标准答案
        pred_answer: 预测答案
        task_type: 任务类型

    Returns:
        (is_correct, match_type)
    """
    if gt_answer == "NULL" and pred_answer == "NULL":
        return True, "both_null"
    if gt_answer == "NULL" or pred_answer == "NULL":
        return False, "one_null"

    # 根据任务类型选择比较方法
    if task_type in ['BFS', 'DFS']:
        return compare_traversal_answers(gt_answer, pred_answer, task_type)
    elif task_type in ['connected_component', 'neighbor', 'common_neighbor', 'predecessor']:
        return compare_set_answers(gt_answer, pred_answer)
    elif task_type in ['page_rank', 'clustering_coefficient']:
        return compare_numeric_answers(gt_answer, pred_answer)
    else:
        # 默认精确匹配（标准化后）
        gt_norm = normalize_answer(gt_answer)
        pred_norm = normalize_answer(pred_answer)
        is_correct = gt_norm == pred_norm
        if is_correct:
            return True, "exact_match"
        elif gt_answer.lower() == pred_answer.lower():
            return True, "case_insensitive_match"
        else:
            return False, "no_match"


class VolcanoAPIClient:
    """火山引擎API客户端 - 支持豆包和DeepSeek

    火山引擎支持直接使用模型名称或Endpoint ID调用。
    API文档: https://www.volcengine.com/docs/82379/1222542

    模型名称示例:
    - doubao-seed-1-8-251215 (豆包1.8)
    - deepseek-v3-2-251201 (DeepSeek V3.2)
    """

    def __init__(self, api_key: str, model: str = "doubao-seed-1-8-251215"):
        self.api_key = api_key
        self.model = model
        # 火山引擎API endpoint - 使用V3版本
        self.base_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

    def chat(self, instruction: str, max_tokens: int = 2048) -> str:
        """同步调用API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        system_prompt = (
            "You are a helpful assistant that solves graph reasoning tasks. "
            "Follow the instructions carefully and provide your answer in the specified format."
        )

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=180
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            logger.error(f"Volcano API HTTP error: {e.response.status_code} - {e.response.text}")
            return ""
        except Exception as e:
            logger.error(f"Volcano API error: {e}")
            return ""


class AliyunQwenClient:
    """阿里云Qwen API客户端"""

    def __init__(self, api_key: str, model: str = "qwen-max"):
        self.api_key = api_key
        self.model = model
        # 阿里云DashScope API endpoint
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def chat(self, instruction: str, max_tokens: int = 2048) -> str:
        """同步调用API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        system_prompt = (
            "You are a helpful assistant that solves graph reasoning tasks. "
            "Follow the instructions carefully and provide your answer in the specified format."
        )

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Aliyun API error: {e}")
            return ""


class DeepSeekAPIClient:
    """DeepSeek官方API客户端"""

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.deepseek.com/chat/completions"

    def chat(self, instruction: str, max_tokens: int = 2048) -> str:
        """同步调用API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        system_prompt = (
            "You are a helpful assistant that solves graph reasoning tasks. "
            "Follow the instructions carefully and provide your answer in the specified format."
        )

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            return ""


class APIModelEvaluator:
    """API模型评测器 - 支持断点续传"""

    def __init__(
        self,
        client,
        model_name: str,
        max_tokens: int = 2048,
        verbose: bool = False,
        rate_limit: float = 0.5,  # 请求间隔秒数
        checkpoint_interval: int = 1  # 每处理多少样本保存一次检查点
    ):
        self.client = client
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.rate_limit = rate_limit
        self.checkpoint_interval = checkpoint_interval

    def evaluate_sample(self, sample: Dict, idx: int) -> Dict:
        """评测单个样本"""
        task_type = sample.get('task_type', 'unknown')
        instruction = sample.get('instruction', '')
        ground_truth = sample.get('output', '')
        sample_id = sample.get('id', f'sample_{idx}')

        # 调用API
        try:
            prediction = self.client.chat(instruction, self.max_tokens)
            time.sleep(self.rate_limit)  # 限速
        except Exception as e:
            logger.error(f"Error generating for sample {sample_id}: {e}")
            prediction = ""

        # 提取答案并比较
        gt_answer = extract_final_answer(ground_truth)
        pred_answer = extract_final_answer(prediction)

        # 使用智能答案比较（支持集合匹配、数值容差、遍历验证）
        is_correct, match_type = compare_answers(gt_answer, pred_answer, task_type)

        gt_norm = normalize_answer(gt_answer)
        pred_norm = normalize_answer(pred_answer)

        result = {
            'id': sample_id,
            'task_type': task_type,
            'instruction': instruction[:500] + '...' if len(instruction) > 500 else instruction,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'gt_answer': gt_answer,
            'pred_answer': pred_answer,
            'gt_normalized': gt_norm,
            'pred_normalized': pred_norm,
            'is_correct': is_correct,
            'match_type': match_type
        }

        if self.verbose:
            logger.info(f"Sample {sample_id} ({task_type}): {'✓' if is_correct else '✗'} [{match_type}]")

        return result

    def load_checkpoint(self, output_dir: str) -> Tuple[List[Dict], set]:
        """加载检查点，返回已完成的结果和已处理的样本ID集合"""
        checkpoint_path = os.path.join(output_dir, 'checkpoint.json')
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                results = checkpoint.get('results', [])
                processed_ids = set(checkpoint.get('processed_ids', []))
                logger.info(f"Loaded checkpoint with {len(results)} completed samples")
                return results, processed_ids
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return [], set()

    def save_checkpoint(self, output_dir: str, results: List[Dict], processed_ids: set):
        """保存检查点"""
        checkpoint_path = os.path.join(output_dir, 'checkpoint.json')
        checkpoint = {
            'results': results,
            'processed_ids': list(processed_ids),
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name
        }
        try:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def evaluate(
        self,
        eval_data: List[Dict],
        output_dir: str,
        resume: bool = False
    ) -> Dict:
        """运行评测，支持断点续传"""
        os.makedirs(output_dir, exist_ok=True)

        # 加载检查点（如果启用续传）
        if resume:
            results, processed_ids = self.load_checkpoint(output_dir)
            if processed_ids:
                logger.info(f"Resuming from checkpoint: {len(processed_ids)}/{len(eval_data)} samples already completed")
        else:
            results = []
            processed_ids = set()

        correct_by_task = {}
        total_correct = 0

        # 重新计算已完成样本的统计
        for result in results:
            task_type = result.get('task_type', 'unknown')
            if task_type not in correct_by_task:
                correct_by_task[task_type] = {'total': 0, 'correct': 0}
            correct_by_task[task_type]['total'] += 1
            if result.get('is_correct', False):
                total_correct += 1
                correct_by_task[task_type]['correct'] += 1

        # 计算需要处理的样本
        samples_to_process = []
        for i, sample in enumerate(eval_data):
            sample_id = sample.get('id', f'sample_{i}')
            if sample_id not in processed_ids:
                samples_to_process.append((i, sample, sample_id))

        if not samples_to_process:
            logger.info("All samples already processed!")
        else:
            logger.info(f"Starting evaluation on {len(samples_to_process)} remaining samples with {self.model_name}...")

            from tqdm import tqdm
            for i, sample, sample_id in tqdm(samples_to_process, desc=f"Evaluating {self.model_name}"):
                task_type = sample.get('task_type', 'unknown')

                # 初始化任务统计
                if task_type not in correct_by_task:
                    correct_by_task[task_type] = {'total': 0, 'correct': 0}
                correct_by_task[task_type]['total'] += 1

                # 评测样本
                result = self.evaluate_sample(sample, i)
                results.append(result)
                processed_ids.add(sample_id)

                if result['is_correct']:
                    total_correct += 1
                    correct_by_task[task_type]['correct'] += 1

                # 保存检查点
                if len(results) % self.checkpoint_interval == 0:
                    self.save_checkpoint(output_dir, results, processed_ids)

        # 最终保存检查点
        self.save_checkpoint(output_dir, results, processed_ids)

        # 计算指标
        total = len(eval_data)
        accuracy = total_correct / total if total > 0 else 0

        metrics = {
            'model': self.model_name,
            'total': total,
            'correct': total_correct,
            'accuracy': accuracy,
            'by_task': correct_by_task,
            'timestamp': datetime.now().isoformat()
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
        logger.info(f"Evaluation Results - {self.model_name}")
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
    parser = argparse.ArgumentParser(description='API model evaluation for graph reasoning tasks')
    parser.add_argument('--provider', type=str, required=True,
                        choices=['volcano', 'aliyun', 'deepseek'],
                        help='API provider: volcano (火山引擎), aliyun (阿里云), deepseek')
    parser.add_argument('--api-key', type=str, required=True,
                        help='API key')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name or endpoint ID')
    parser.add_argument('--eval-data-path', type=str, required=True,
                        help='Path to evaluation JSON file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--max-tokens', type=int, default=2048,
                        help='Maximum tokens to generate')
    parser.add_argument('--rate-limit', type=float, default=0.5,
                        help='Seconds between API calls')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if available')
    parser.add_argument('--checkpoint-interval', type=int, default=1,
                        help='Save checkpoint every N samples (default: 1)')

    args = parser.parse_args()

    # 加载评测数据
    logger.info(f"Loading evaluation data from {args.eval_data_path}...")
    with open(args.eval_data_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    logger.info(f"Loaded {len(eval_data)} samples")

    # 创建API客户端
    if args.provider == 'volcano':
        client = VolcanoAPIClient(args.api_key, args.model)
        model_name = f"Volcano-{args.model}"
    elif args.provider == 'aliyun':
        client = AliyunQwenClient(args.api_key, args.model)
        model_name = f"Aliyun-{args.model}"
    elif args.provider == 'deepseek':
        client = DeepSeekAPIClient(args.api_key, args.model)
        model_name = f"DeepSeek-{args.model}"
    else:
        raise ValueError(f"Unknown provider: {args.provider}")

    # 创建评测器
    evaluator = APIModelEvaluator(
        client=client,
        model_name=model_name,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        rate_limit=args.rate_limit,
        checkpoint_interval=args.checkpoint_interval
    )

    # 运行评测
    metrics = evaluator.evaluate(eval_data, args.output_dir, resume=args.resume)

    logger.info("Evaluation completed!")
    return metrics


if __name__ == '__main__':
    main()
