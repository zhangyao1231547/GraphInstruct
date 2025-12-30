#!/usr/bin/env python3
"""
将多语言数据转换为GraphAgent训练格式

多语言数据格式 (multilang):
{
    'task': 'BFS',
    'graph': '[(<5>, <4>), ...]',    # 边列表字符串
    'graph_adj': '{<5>: [...]}',     # 邻接表字符串
    'graph_nl': '节点 <5> 连接到...',  # 自然语言描述
    'question': '从节点 <5> 开始...',
    'answer': '[<5>, <4>, ...]',
    'steps': '让我们逐步...',
    'language': 'zh'
}

训练数据格式 (GraphAgent):
{
    'id': 'sample_id',
    'graph': {
        'node_idx': 0,
        'edge_index': [[src], [dst]],
        'node_list': [0, 1, 2, ...]
    },
    'conversations': [
        {'from': 'human', 'value': '<graph>\\n问题...'},
        {'from': 'gpt', 'value': '步骤...<<<答案>>>'}
    ]
}

Usage:
    python convert_multilang_to_training_format.py \\
        --input data/generated/graphinstruct_zh_19tasks_100samples.json \\
        --output data/training/graphinstruct_zh_training.json
"""

import json
import os
import sys
import re
import argparse
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_graph_adj_str(graph_adj_str: str) -> Dict[int, List[int]]:
    """
    解析邻接表字符串为字典

    输入: '{<0>: [<1>, <2>], <1>: [<0>, <3>], ...}'
    输出: {0: [1, 2], 1: [0, 3], ...}
    """
    adj_dict = {}

    # 提取节点ID模式 <数字>
    pattern = r'<(\d+)>:\s*\[([^\]]*)\]'
    matches = re.findall(pattern, graph_adj_str)

    for node_str, neighbors_str in matches:
        node_id = int(node_str)
        # 解析邻居列表
        neighbor_pattern = r'<(\d+)>'
        neighbors = [int(n) for n in re.findall(neighbor_pattern, neighbors_str)]
        adj_dict[node_id] = neighbors

    return adj_dict


def adj_to_edge_index(adj_dict: Dict[int, List[int]], directed: bool = False) -> List[List[int]]:
    """
    将邻接表转换为edge_index格式 (PyG格式)

    返回: [[src_nodes], [dst_nodes]]
    """
    src_nodes = []
    dst_nodes = []
    seen_edges = set()

    for src, neighbors in adj_dict.items():
        for dst in neighbors:
            if directed:
                src_nodes.append(src)
                dst_nodes.append(dst)
            else:
                # 无向图：避免重复边
                edge = tuple(sorted([src, dst]))
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    src_nodes.append(src)
                    dst_nodes.append(dst)

    return [src_nodes, dst_nodes]


def convert_sample_to_training_format(sample: Dict, sample_idx: int) -> Optional[Dict]:
    """
    将单个多语言样本转换为训练格式
    """
    try:
        task = sample.get('task', 'unknown')
        lang = sample.get('language', 'en')

        # 1. 解析图结构
        graph_adj_str = sample.get('graph_adj', '')
        adj_dict = parse_graph_adj_str(graph_adj_str)

        if not adj_dict:
            logger.warning(f"Sample {sample_idx}: Could not parse graph_adj")
            return None

        # 获取节点列表
        all_nodes = set(adj_dict.keys())
        for neighbors in adj_dict.values():
            all_nodes.update(neighbors)
        node_list = sorted(list(all_nodes))

        # 转换为edge_index
        directed = sample.get('directed', False)
        edge_index = adj_to_edge_index(adj_dict, directed)

        # 2. 构建对话
        graph_nl = sample.get('graph_nl', '')
        question = sample.get('question', '')
        steps = sample.get('steps', '')
        answer = sample.get('answer', '')

        # 构建human消息: <graph> + 图描述 + 问题
        human_value = f"<graph>\n{graph_nl}\n\n{question}"

        # 构建gpt消息: 步骤 + <<<答案>>>
        gpt_value = f"{steps}<<<{answer}>>>"

        conversations = [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": gpt_value}
        ]

        # 3. 构建最终样本
        sample_id = sample.get('id', f"{task}_{lang}_{sample_idx}")

        training_sample = {
            "id": sample_id,
            "graph": {
                "node_idx": 0,  # 通常表示中心节点
                "edge_index": edge_index,
                "node_list": node_list
            },
            "conversations": conversations,
            # 额外元数据
            "task_type": task,
            "language": lang,
            "num_nodes": len(node_list),
            "num_edges": len(edge_index[0])
        }

        return training_sample

    except Exception as e:
        logger.error(f"Error converting sample {sample_idx}: {e}")
        return None


def convert_dataset(input_path: str, output_path: str, max_samples: Optional[int] = None):
    """
    转换整个数据集
    """
    logger.info(f"Loading data from {input_path}...")

    with open(input_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    if max_samples:
        samples = samples[:max_samples]

    logger.info(f"Converting {len(samples)} samples...")

    converted_samples = []
    stats = {'total': 0, 'success': 0, 'failed': 0}

    for i, sample in enumerate(samples):
        stats['total'] += 1

        converted = convert_sample_to_training_format(sample, i)

        if converted:
            converted_samples.append(converted)
            stats['success'] += 1
        else:
            stats['failed'] += 1

    # 保存
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_samples, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info("Conversion Statistics:")
    logger.info(f"  Total: {stats['total']}")
    logger.info(f"  Success: {stats['success']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"{'='*60}")

    return converted_samples


def verify_conversion(output_path: str, num_samples: int = 2):
    """验证转换结果"""
    logger.info(f"\n===== 验证转换结果 =====")

    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i in range(min(num_samples, len(data))):
        sample = data[i]
        logger.info(f"\n--- 样本 {i}: {sample['id']} ---")
        logger.info(f"Task: {sample.get('task_type', 'N/A')}")
        logger.info(f"Language: {sample.get('language', 'N/A')}")
        logger.info(f"Nodes: {sample['graph']['node_list'][:10]}... ({len(sample['graph']['node_list'])} total)")
        logger.info(f"Edges: {len(sample['graph']['edge_index'][0])}")

        # 打印对话片段
        human_msg = sample['conversations'][0]['value'][:200]
        gpt_msg = sample['conversations'][1]['value'][:200]
        logger.info(f"Human (first 200 chars): {human_msg}...")
        logger.info(f"GPT (first 200 chars): {gpt_msg}...")


def main():
    parser = argparse.ArgumentParser(description='Convert multilang data to GraphAgent training format')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum samples to convert')
    parser.add_argument('--verify', action='store_true', help='Verify conversion results')

    args = parser.parse_args()

    converted = convert_dataset(args.input, args.output, args.max_samples)

    if args.verify and converted:
        verify_conversion(args.output)


if __name__ == '__main__':
    main()
