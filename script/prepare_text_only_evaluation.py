#!/usr/bin/env python3
"""
GraphInstruct 纯文本评测数据准备脚本
从GraphInstruct测试数据中采样19个任务各10条数据,保存为纯文本格式(JSON)
用于原始Qwen模型的baseline评测
"""

import json
import os
import random
import logging
import argparse
from typing import Dict, List

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 19个图任务类型对应的测试文件映射
TASK_FILE_MAPPING = {
    'BFS': 'BFS-int_id_test.json',
    'DFS': 'DFS-int_id_test.json',
    'bipartite': 'bipartite-int_id_test.json',
    'clustering_coefficient': 'clustering_coefficient-int_id_test.json',
    'common_neighbor': 'common_neighbor-int_id_test.json',
    'connected_component': 'connected_component-int_id_test.json',
    'connectivity': 'connectivity-int_id_test.json',
    'cycle': 'cycle-int_id_test.json',
    'degree': 'degree-int_id_test.json',
    'diameter': 'diameter-int_id_test.json',
    'edge': 'edge-int_id_test.json',
    'jaccard': 'jaccard-int_id_test.json',
    'maximum_flow': 'maximum_flow-int_id_test.json',
    'MST': 'MST-int_id_test.json',
    'neighbor': 'neighbor-int_id_test.json',
    'page_rank': 'page_rank-int_id_test.json',
    'predecessor': 'predecessor-int_id_test.json',
    'shortest_path': 'shortest_path-int_id_test.json',
    'topological_sort': 'topological_sort-int_id_test.json',
}


def sample_and_convert(
    test_data_dir: str,
    output_path: str,
    samples_per_task: int = 10,
    seed: int = 42
) -> List[Dict]:
    """从测试数据中采样并转换为纯文本格式"""
    random.seed(seed)

    all_samples = []
    stats = {'total': 0, 'success': 0}

    for task_type, filename in TASK_FILE_MAPPING.items():
        input_path = os.path.join(test_data_dir, filename)

        if not os.path.exists(input_path):
            logger.warning(f"Skipping {task_type}: {input_path} not found")
            continue

        logger.info(f"Processing {task_type}...")

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 随机采样
        if len(data) > samples_per_task:
            sampled_data = random.sample(data, samples_per_task)
        else:
            sampled_data = data
            logger.warning(f"{task_type} has only {len(data)} samples (requested {samples_per_task})")

        # 转换每个样本
        for sample in sampled_data:
            stats['total'] += 1

            result = {
                'id': sample.get('id', f'{task_type}_{stats["total"]}'),
                'task_type': task_type,
                'instruction': sample.get('instruction', ''),
                'output': sample.get('output', ''),
            }

            all_samples.append(result)
            stats['success'] += 1

        logger.info(f"  {task_type}: {len(sampled_data)} samples converted")

    # 保存
    logger.info(f"Saving {len(all_samples)} samples to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info("Processing Statistics:")
    logger.info(f"  Total processed: {stats['total']}")
    logger.info(f"  Success: {stats['success']}")
    logger.info(f"{'='*60}")

    return all_samples


def main():
    parser = argparse.ArgumentParser(description='Prepare text-only evaluation data from GraphInstruct')
    parser.add_argument('--test-data-dir', type=str,
                        default='/nvme0/work/workspaces-zy/GraphInstruct/data/test',
                        help='Directory containing test JSON files')
    parser.add_argument('--output-path', type=str,
                        default='/nvme0/work/workspaces-zy/GraphInstruct/data/eval/graphinstruct_eval_19tasks_10samples_text.json',
                        help='Output path for evaluation JSON file')
    parser.add_argument('--samples-per-task', type=int, default=10,
                        help='Number of samples per task type')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')

    args = parser.parse_args()

    sample_and_convert(
        test_data_dir=args.test_data_dir,
        output_path=args.output_path,
        samples_per_task=args.samples_per_task,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
