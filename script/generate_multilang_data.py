#!/usr/bin/env python3
"""
多语言图推理数据生成脚本
支持生成中文和英文版本的训练/评测数据

Usage:
    # 生成中文数据
    python generate_multilang_data.py --lang zh --output-dir ./data/zh --num-samples 100

    # 生成英文数据
    python generate_multilang_data.py --lang en --output-dir ./data/en --num-samples 100

    # 只生成特定任务
    python generate_multilang_data.py --lang zh --tasks BFS,DFS,degree --num-samples 50
"""

import json
import os
import sys
import random
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GTG.utils.utils import set_random_seed
from GTG.utils.language import LANG_EN, LANG_ZH, get_task_templates, graph_to_natural_language_multilang


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 所有支持的任务类型
ALL_TASKS = [
    'BFS', 'DFS', 'shortest_path', 'page_rank', 'topological_sort',
    'cycle', 'degree', 'connectivity', 'edge', 'neighbor',
    'bipartite', 'clustering_coefficient', 'common_neighbor',
    'connected_component', 'diameter', 'jaccard', 'maximum_flow',
    'MST', 'predecessor'
]

# 默认配置
DEFAULT_CONFIG = {
    'num_nodes_range': '[8, 15]',
}


def import_task_module(task_name: str):
    """动态导入任务模块"""
    try:
        # 尝试导入多语言版本
        module = __import__(
            f'GTG.tasks.{task_name}.{task_name}_multilang',
            fromlist=['generate_a_sample_multilang']
        )
        return module, True
    except ImportError:
        # 回退到原始版本
        try:
            module = __import__(
                f'GTG.tasks.{task_name}.{task_name}',
                fromlist=['generate_a_sample']
            )
            return module, False
        except ImportError:
            logger.warning(f"Could not import task module: {task_name}")
            return None, False


def generate_sample_for_task(task_name: str, config: Dict, lang: str = LANG_EN) -> Optional[Dict]:
    """
    为指定任务生成样本

    Args:
        task_name: 任务名称
        config: 配置字典
        lang: 语言代码

    Returns:
        生成的样本字典,失败返回None
    """
    module, is_multilang = import_task_module(task_name)
    if module is None:
        return None

    try:
        if is_multilang and hasattr(module, 'generate_a_sample_multilang'):
            sample = module.generate_a_sample_multilang(config, lang=lang)
        else:
            # 使用原始英文版本
            sample = module.generate_a_sample(config)
            # 如果需要中文但只有英文版本,记录警告
            if lang == LANG_ZH:
                logger.debug(f"Task {task_name} does not have multilang support, using English")

        sample['language'] = lang
        return sample

    except Exception as e:
        logger.error(f"Error generating sample for {task_name}: {e}")
        return None


def generate_dataset(
    tasks: List[str],
    config: Dict,
    lang: str,
    samples_per_task: int,
    output_dir: str,
    seed: int = 42
) -> List[Dict]:
    """
    批量生成数据集

    Args:
        tasks: 任务列表
        config: 配置字典
        lang: 语言代码
        samples_per_task: 每个任务的样本数
        output_dir: 输出目录
        seed: 随机种子

    Returns:
        生成的所有样本列表
    """
    set_random_seed(seed)

    all_samples = []
    stats = {'total': 0, 'success': 0, 'by_task': {}}

    logger.info(f"Generating {samples_per_task} samples per task for {len(tasks)} tasks")
    logger.info(f"Language: {lang}")

    for task_name in tasks:
        logger.info(f"Processing task: {task_name}")
        stats['by_task'][task_name] = {'total': 0, 'success': 0}

        for i in range(samples_per_task):
            stats['total'] += 1
            stats['by_task'][task_name]['total'] += 1

            sample = generate_sample_for_task(task_name, config, lang)

            if sample is not None:
                sample['id'] = f"{task_name}_{lang}_{i}"
                all_samples.append(sample)
                stats['success'] += 1
                stats['by_task'][task_name]['success'] += 1
            else:
                logger.warning(f"Failed to generate sample {i} for {task_name}")

        task_stats = stats['by_task'][task_name]
        logger.info(f"  {task_name}: {task_stats['success']}/{task_stats['total']} samples generated")

    # 保存数据集
    os.makedirs(output_dir, exist_ok=True)

    # 保存为JSON格式
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"graphinstruct_{lang}_{len(tasks)}tasks_{samples_per_task}samples.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    logger.info(f"\n{'='*60}")
    logger.info("Generation Statistics:")
    logger.info(f"  Total attempted: {stats['total']}")
    logger.info(f"  Total success: {stats['success']}")
    logger.info(f"  Success rate: {stats['success']/stats['total']*100:.1f}%")
    logger.info(f"  Output file: {output_path}")
    logger.info(f"{'='*60}")

    # 保存统计信息
    stats_path = os.path.join(output_dir, f"generation_stats_{lang}_{timestamp}.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    return all_samples


def convert_to_alpaca_format(samples: List[Dict], output_path: str, system_prompt: str = None):
    """
    转换为Alpaca训练格式

    Args:
        samples: 样本列表
        output_path: 输出路径
        system_prompt: 系统提示词
    """
    if system_prompt is None:
        system_prompt = "You are a helpful assistant that solves graph reasoning tasks."

    alpaca_data = []
    for sample in samples:
        # 构建instruction
        graph_nl = sample.get('graph_nl', '')
        question = sample.get('question', '')

        instruction = f"{graph_nl}\n\n{question}"

        # 构建output
        steps = sample.get('steps', '')
        answer = sample.get('answer', '')
        output = f"{steps}<<<{answer}>>>"

        alpaca_item = {
            'instruction': instruction,
            'input': '',
            'output': output,
            'task_type': sample.get('task', ''),
            'language': sample.get('language', 'en'),
        }
        alpaca_data.append(alpaca_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Alpaca format saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate multilingual graph reasoning data')
    parser.add_argument('--lang', type=str, default='zh', choices=['en', 'zh'],
                        help='Language for generation (en/zh)')
    parser.add_argument('--tasks', type=str, default=None,
                        help='Comma-separated list of tasks (default: all)')
    parser.add_argument('--samples-per-task', type=int, default=10,
                        help='Number of samples per task')
    parser.add_argument('--output-dir', type=str,
                        default='/nvme0/work/workspaces-zy/GraphInstruct/data/generated',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num-nodes-range', type=str, default='[8, 15]',
                        help='Range of number of nodes in graphs')
    parser.add_argument('--alpaca-format', action='store_true',
                        help='Also output in Alpaca format')

    args = parser.parse_args()

    # 解析任务列表
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(',')]
    else:
        tasks = ALL_TASKS

    # 验证任务
    invalid_tasks = [t for t in tasks if t not in ALL_TASKS]
    if invalid_tasks:
        logger.warning(f"Invalid tasks will be skipped: {invalid_tasks}")
        tasks = [t for t in tasks if t in ALL_TASKS]

    # 配置
    config = {
        'num_nodes_range': args.num_nodes_range,
    }

    # 生成数据集
    samples = generate_dataset(
        tasks=tasks,
        config=config,
        lang=args.lang,
        samples_per_task=args.samples_per_task,
        output_dir=args.output_dir,
        seed=args.seed
    )

    # 可选: 转换为Alpaca格式
    if args.alpaca_format and samples:
        alpaca_path = os.path.join(
            args.output_dir,
            f"graphinstruct_{args.lang}_alpaca.json"
        )
        convert_to_alpaca_format(samples, alpaca_path)

    logger.info("Done!")


if __name__ == '__main__':
    main()
