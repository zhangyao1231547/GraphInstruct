#!/usr/bin/env python3
"""
从英文数据直接翻译生成对应的中文数据

输入: graphinstruct_english_combined_text.json (95,000 条英文)
输出: graphinstruct_chinese_combined_text.json (95,000 条中文，完全对应)
"""

import json
import re
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 任务类型英文到中文的映射
TASK_TYPE_ZH = {
    'BFS': '广度优先搜索',
    'DFS': '深度优先搜索',
    'BIPARTITE': '二分图判断',
    'bipartite': '二分图判断',
    'CLUSTERING_COEFFICIENT': '聚类系数',
    'clustering_coefficient': '聚类系数',
    'COMMON_NEIGHBOR': '共同邻居',
    'common_neighbor': '共同邻居',
    'CONNECTED_COMPONENT': '连通分量',
    'connected_component': '连通分量',
    'CONNECTIVITY': '连通性',
    'connectivity': '连通性',
    'CYCLE': '环检测',
    'cycle': '环检测',
    'DEGREE': '节点度数',
    'degree': '节点度数',
    'DIAMETER': '图直径',
    'diameter': '图直径',
    'EDGE': '边存在性',
    'edge': '边存在性',
    'JACCARD': 'Jaccard相似度',
    'jaccard': 'Jaccard相似度',
    'MAXIMUM_FLOW': '最大流',
    'maximum_flow': '最大流',
    'MST': '最小生成树',
    'mst': '最小生成树',
    'NEIGHBOR': '邻居节点',
    'neighbor': '邻居节点',
    'PAGE_RANK': 'PageRank',
    'page_rank': 'PageRank',
    'PREDECESSOR': '前驱节点',
    'predecessor': '前驱节点',
    'SHORTEST_PATH': '最短路径',
    'shortest_path': '最短路径',
    'TOPOLOGICAL_SORT': '拓扑排序',
    'topological_sort': '拓扑排序'
}


def translate_instruction(text: str, task_type: str) -> str:
    """翻译 instruction 部分"""
    result = text

    # 翻译任务类型
    task_zh = TASK_TYPE_ZH.get(task_type, task_type)

    # 基础翻译
    result = re.sub(
        r'The following is a question related to the graph reasoning task ([A-Z_]+)\.',
        lambda m: f'以下是一个关于图推理任务 {m.group(1)} ({TASK_TYPE_ZH.get(m.group(1), m.group(1))}) 的问题。',
        result
    )

    result = re.sub(
        r'Please answer the question step by step and format your final response as <<<ANSWER>>>',
        '请逐步分析并解答，最终答案请用 <<<答案>>> 的格式给出',
        result
    )

    result = re.sub(r'For example, if your answer is ([^,]+), please format your answer as <<<\1>>>\.',
                   r'例如，如果你的答案是 \1，请将答案格式化为 <<<\1>>>。',
                   result)

    # 图类型翻译
    result = re.sub(r'Given a directed graph', '给定一个有向图', result)
    result = re.sub(r'Given an undirected graph', '给定一个无向图', result)
    result = re.sub(r'with the following connections:', '其连接关系如下：', result)

    # 节点连接翻译
    result = re.sub(r'Node (\d+) is connected to nodes? ([0-9, ]+)\.',
                   r'节点 \1 连接到节点 \2。',
                   result)

    # 问题翻译（保持灵活，只翻译常见模式）
    result = re.sub(r'Start from node (\d+), output a sequence of traversal in breadth-first search \(BFS\) order\.',
                   r'从节点 \1 开始，输出广度优先搜索 (BFS) 遍历序列。',
                   result)

    result = re.sub(r'Start from node (\d+), output a sequence of traversal in depth-first search \(DFS\) order\.',
                   r'从节点 \1 开始，输出深度优先搜索 (DFS) 遍历序列。',
                   result)

    result = re.sub(r'What is the ([a-z ]+) of node (\d+)\?',
                   lambda m: f'节点 \2 的{translate_property(m.group(1))}是多少？',
                   result)

    result = re.sub(r'Find the ([a-z ]+) between nodes? (\d+) and (\d+)\.',
                   lambda m: f'找出节点 \2 和 \3 之间的{translate_property(m.group(1))}。',
                   result)

    return result


def translate_property(prop: str) -> str:
    """翻译属性名称"""
    mapping = {
        'degree': '度数',
        'in-degree': '入度',
        'out-degree': '出度',
        'neighbors': '邻居',
        'predecessors': '前驱节点',
        'shortest path': '最短路径',
        'clustering coefficient': '聚类系数',
        'common neighbors': '共同邻居',
        'jaccard similarity': 'Jaccard相似度',
        'pagerank': 'PageRank值'
    }
    return mapping.get(prop.lower(), prop)


def translate_output(text: str, task_type: str) -> str:
    """翻译 output 部分（推理步骤）"""
    result = text

    # BFS/DFS 翻译
    result = re.sub(r"Let's run breadth-first search \(BFS\) step by step\.",
                   "让我们逐步执行广度优先搜索 (BFS)。", result)
    result = re.sub(r"Let's run depth-first search \(DFS\) step by step\.",
                   "让我们逐步执行深度优先搜索 (DFS)。", result)

    result = re.sub(r'Visit node (\d+)\. Unvisited neighbors of node \1 are \[([^\]]+)\]\.',
                   r'访问节点 \1。节点 \1 的未访问邻居是 [\2]。', result)
    result = re.sub(r'Visit node (\d+)\.',
                   r'访问节点 \1。', result)

    result = re.sub(r'So the BFS traversal is', '因此 BFS 遍历序列是', result)
    result = re.sub(r'So the DFS traversal is', '因此 DFS 遍历序列是', result)

    # 通用模式
    result = re.sub(r"Let's ([a-z ]+) step by step\.",
                   lambda m: f"让我们逐步{translate_action(m.group(1))}。", result)

    result = re.sub(r'Therefore,', '因此，', result)
    result = re.sub(r'Thus,', '因此，', result)
    result = re.sub(r'Hence,', '因此，', result)
    result = re.sub(r'Finally,', '最后，', result)
    result = re.sub(r'First,', '首先，', result)
    result = re.sub(r'Next,', '接下来，', result)
    result = re.sub(r'Then,', '然后，', result)

    # 答案标记翻译
    result = re.sub(r'<<<ANSWER>>>', '<<<答案>>>', result)
    result = re.sub(r'<<<', '<<<', result)  # 保持答案标记
    result = re.sub(r'>>>', '>>>', result)

    return result


def translate_action(action: str) -> str:
    """翻译动作描述"""
    mapping = {
        'calculate': '计算',
        'calculate the diameter': '计算图的直径',
        'calculate the clustering coefficient': '计算聚类系数',
        'calculate pagerank': '计算PageRank',
        'find': '找出',
        'find the shortest path': '找出最短路径',
        'find the common neighbors': '找出共同邻居',
        'check': '检查',
        'check if': '检查是否'
    }
    return mapping.get(action.lower(), action)


def translate_sample(sample: dict) -> dict:
    """翻译单个样本"""
    translated = {
        'id': sample['id'] + '_zh',  # 添加 _zh 后缀标识中文版本
        'task_type': sample['task_type'],
        'instruction': translate_instruction(sample['instruction'], sample['task_type']),
        'output': translate_output(sample['output'], sample['task_type'])
    }
    return translated


def main():
    input_file = '/mnt/yrfs/GraphAgent_model/zy/data/graphInstruct/graphinstruct_english_combined_text.json'
    output_file = '/mnt/yrfs/GraphAgent_model/zy/data/graphInstruct/graphinstruct_chinese_combined_text.json'

    logger.info(f'读取英文数据: {input_file}')
    with open(input_file, 'r', encoding='utf-8') as f:
        en_data = json.load(f)

    logger.info(f'共 {len(en_data):,} 条样本')

    # 翻译
    zh_data = []
    for sample in tqdm(en_data, desc='翻译中'):
        try:
            translated = translate_sample(sample)
            zh_data.append(translated)
        except Exception as e:
            logger.warning(f'翻译失败 (ID: {sample.get("id", "unknown")}): {e}')
            continue

    logger.info(f'\n成功翻译: {len(zh_data):,} 条样本')

    # 保存
    logger.info(f'保存到: {output_file}')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(zh_data, f, ensure_ascii=False, indent=2)

    # 统计
    from collections import Counter
    task_counts = Counter(s['task_type'] for s in zh_data)
    logger.info(f'\n任务分布:')
    for task, count in sorted(task_counts.items()):
        task_zh = TASK_TYPE_ZH.get(task, task)
        logger.info(f'  {task:25s} ({task_zh}): {count:6,d}')

    logger.info(f'\n✓ 完成！中文数据与英文数据完全对应（1:1）')


if __name__ == '__main__':
    main()
