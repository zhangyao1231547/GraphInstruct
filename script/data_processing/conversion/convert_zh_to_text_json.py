#!/usr/bin/env python3
"""
将原始中文数据转换为与英文一致的纯文本 JSON 格式

原始格式 (graphinstruct_zh_19tasks_10000samples.json):
{
    "task": "BFS",
    "graph_nl": "节点 <1> 连接到节点 <2>, <0>...",
    "question": "从节点 <3> 开始，输出广度优先搜索...",
    "steps": "让我们逐步执行...",
    "answer": "[<3>, <1>, <7>, ...]",
    "id": "BFS_zh_0",
    "language": "zh"
}

目标格式 (与英文数据一致):
{
    "id": "BFS_zh_0",
    "task_type": "BFS",
    "instruction": "以下是一个关于图推理任务 BFS 的问题...",
    "output": "让我们逐步执行...<<<[<3>, <1>, <7>, ...]>>>"
}
"""

import json
import os
import argparse
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 中文任务类型映射
TASK_TYPE_ZH = {
    'BFS': '广度优先搜索 (BFS)',
    'DFS': '深度优先搜索 (DFS)',
    'shortest_path': '最短路径',
    'page_rank': '页面排名 (PageRank)',
    'topological_sort': '拓扑排序',
    'cycle': '环检测',
    'degree': '节点度数',
    'connectivity': '连通性检测',
    'edge': '边检测',
    'neighbor': '邻居节点',
    'bipartite': '二分图最大匹配',
    'clustering_coefficient': '聚类系数',
    'common_neighbor': '共同邻居',
    'connected_component': '连通分量',
    'diameter': '图直径',
    'jaccard': 'Jaccard 相似度',
    'maximum_flow': '最大流',
    'MST': '最小生成树',
    'mst': '最小生成树',
    'predecessor': '前驱节点'
}


def convert_sample(sample: dict) -> dict:
    """转换单个样本到目标格式"""
    task = sample.get('task', 'UNKNOWN')
    graph_nl = sample.get('graph_nl', '')
    question = sample.get('question', '')
    steps = sample.get('steps', '')
    answer = sample.get('answer', '')
    sample_id = sample.get('id', '')
    is_directed = sample.get('directed', False)

    # 获取任务类型中文名
    task_name_zh = TASK_TYPE_ZH.get(task, task)

    # 构建 instruction
    graph_type = '有向图' if is_directed else '无向图'
    instruction = f"以下是一个关于图推理任务 {task.upper()} ({task_name_zh}) 的问题。请逐步分析并解答，最终答案请用 <<<答案>>> 的格式给出。\n\n"
    instruction += f"给定一个{graph_type}，其连接关系如下：\n{graph_nl}\n\n"
    instruction += question

    # 构建 output (包含完整的推理步骤)
    if steps:
        output = f"{steps}\n<<<{answer}>>>"
    else:
        output = f"<<<{answer}>>>"

    return {
        'id': sample_id,
        'task_type': task,
        'instruction': instruction,
        'output': output
    }


def convert_file(input_path: str, output_path: str):
    """转换整个文件"""
    logger.info(f"读取数据: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"共 {len(data):,} 条样本")

    converted = []
    skipped = 0

    for sample in tqdm(data, desc="转换中"):
        try:
            converted_sample = convert_sample(sample)
            converted.append(converted_sample)
        except Exception as e:
            logger.warning(f"转换失败 (ID: {sample.get('id', 'unknown')}): {e}")
            skipped += 1
            continue

    logger.info(f"\n转换完成:")
    logger.info(f"  成功转换: {len(converted):,} 条样本")
    logger.info(f"  跳过: {skipped} 条样本")

    # 统计任务分布
    from collections import Counter
    task_counts = Counter(s['task_type'] for s in converted)
    logger.info(f"\n任务分布:")
    for task, count in sorted(task_counts.items()):
        task_name = TASK_TYPE_ZH.get(task, task)
        logger.info(f"  {task:25s} ({task_name}): {count:6,d}")

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    logger.info(f"\n已保存到: {output_path}")

    # 显示示例
    if converted:
        logger.info(f"\n第一个样本示例:")
        logger.info(f"  ID: {converted[0]['id']}")
        logger.info(f"  Task: {converted[0]['task_type']}")
        logger.info(f"  Instruction 长度: {len(converted[0]['instruction'])} 字符")
        logger.info(f"  Output 长度: {len(converted[0]['output'])} 字符")
        logger.info(f"  Output 开头: {converted[0]['output'][:150]}...")

    return len(converted)


def main():
    parser = argparse.ArgumentParser(description='将原始中文数据转换为与英文一致的格式')
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='/nvme0/work/workspaces-zy/GraphInstruct/data/chinese_training/graphinstruct_zh_19tasks_10000samples.json',
        help='输入文件路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='/mnt/yrfs/GraphAgent_model/zy/data/graphInstruct/graphinstruct_chinese_combined_text.json',
        help='输出文件路径'
    )
    args = parser.parse_args()

    convert_file(args.input, args.output)


if __name__ == '__main__':
    main()
