#!/usr/bin/env python3
"""
修复中文数据中的结构问题

需要修复的任务：
1. MAXIMUM_FLOW - instruction 中的英文节点连接描述
2. shortest_path - output 中翻译不完整
3. page_rank - output 中翻译不完整
"""

import json
import re
from tqdm import tqdm


def translate_node_connections(text):
    """翻译节点连接描述"""
    result = text

    # "Node <X> is connected to nodes <Y> (weight: Z), ..."
    result = re.sub(
        r'Node <(\d+)> is connected to nodes? ([^.\n]+)\.',
        lambda m: translate_connection_line(m.group(1), m.group(2)),
        result
    )

    # 没有尖括号的版本
    result = re.sub(
        r'Node (\d+) is connected to nodes? ([^.\n]+)\.',
        lambda m: translate_connection_line(m.group(1), m.group(2)),
        result
    )

    return result


def translate_connection_line(node, connections_str):
    """翻译单行连接"""
    # 翻译 "(weight: X)" 为 "(权重: X)"
    connections_translated = re.sub(r'\(weight: (\d+)\)', r'(权重: \1)', connections_str)
    return f'节点 <{node}> 连接到节点 {connections_translated}。'


def fix_maximum_flow_sample(sample):
    """修复 MAXIMUM_FLOW 样本"""
    # 翻译 instruction 中的节点连接
    instruction = sample['instruction']
    instruction = translate_node_connections(instruction)

    sample['instruction'] = instruction
    return sample


def fix_shortest_path_sample(sample):
    """修复 shortest_path 样本（改进翻译）"""
    # 这个任务已经翻译了基本结构，但output中有大量英文
    # 只翻译最关键的部分
    output = sample['output']

    # 翻译常见模式
    output = re.sub(r'The unvisited nodes are:', '未访问的节点：', output)
    output = re.sub(r'node <(\d+)>: inf', r'节点 <\1>: inf', output)
    output = re.sub(r'node <(\d+)>: (\d+)', r'节点 <\1>: \2', output)
    output = re.sub(r'Visit node <(\d+)>', r'访问节点 <\1>', output)
    output = re.sub(r'from node <(\d+)> to node <(\d+)>', r'从节点 <\1> 到节点 <\2>', output)
    output = re.sub(r'The shortest distance is', '最短距离是', output)
    output = re.sub(r'The answer is', '答案是', output)

    sample['output'] = output
    return sample


def fix_page_rank_sample(sample):
    """修复 page_rank 样本（改进翻译）"""
    output = sample['output']

    # 基础翻译 - 必须在其他翻译之前
    output = re.sub(r"Let's calculate", '让我们计算', output)
    output = re.sub(r'In case of devide by zero', '为避免除以零', output)
    output = re.sub(r'we add a small constant', '我们添加一个小常数', output)

    # 复杂句式翻译（优先级高，避免被拆分）
    output = re.sub(r'should be 归一化的 on axis (\d+) through the process of', r'应在轴 \1 上通过以下过程进行归一化', output)
    output = re.sub(r'should be normalized on axis (\d+) through the process of', r'应在轴 \1 上通过以下过程进行归一化', output)
    output = re.sub(r'so as to (the )?归一化邻接矩阵', r'从而得到归一化邻接矩阵', output)
    output = re.sub(r'so as to the normalized adjacency matrix', '从而得到归一化邻接矩阵', output)
    output = re.sub(r'to the denominator while calculating', '到分母中以计算', output)

    # "According to" 句式翻译
    output = re.sub(r'According to the graph inst[ru]ct?ure,?\s*', '根据图结构，', output)
    output = re.sub(r'According to (the )?graph,?\s*', '根据图，', output)

    # "the X of the Y is" 句式翻译
    output = re.sub(r'the (adjacency )?矩阵 of the graph is:', '图的邻接矩阵是：', output)
    output = re.sub(r'the adjacency matrix of the graph is:', '图的邻接矩阵是：', output)

    # 矩阵相关术语翻译（在复杂句式之后）
    output = re.sub(r'the normalized adjacency matrix M', '归一化邻接矩阵 M', output)
    output = re.sub(r'the normalized adjacency matrix', '归一化邻接矩阵', output)
    output = re.sub(r'归一化邻接矩阵 M is:', '归一化邻接矩阵 M 是：', output)
    output = re.sub(r'\bmatrix\b', '矩阵', output, flags=re.IGNORECASE)
    output = re.sub(r'adjacency 矩阵', '邻接矩阵', output)
    output = re.sub(r'adjacency matrix', '邻接矩阵', output)
    output = re.sub(r'\bnormalized\b', '归一化的', output)

    # 连接词和过渡词
    output = re.sub(r'For instance,?\s*', '例如，', output)
    output = re.sub(r'For example,?\s*', '例如，', output)
    output = re.sub(r'according to the formula', '根据公式', output)
    output = re.sub(r'As a result,?\s*', '因此，', output)
    output = re.sub(r'Therefore,?\s*', '因此，', output)
    output = re.sub(r'Thus,?\s*', '因此，', output)

    # "is" 后置翻译
    output = re.sub(r' is:', ' 是：', output)
    output = re.sub(r' is $', ' 是', output)

    # PageRank 特定翻译
    output = re.sub(r'PageRank values of each node', '各节点的 PageRank 值', output)
    output = re.sub(r'at round (\d+) are:', r'在第 \1 轮是：', output)
    output = re.sub(r'After (\d+) rounds of iteration', r'经过 \1 轮迭代后', output)
    output = re.sub(r'the PageRank values converge', 'PageRank 值收敛', output)
    output = re.sub(r'So the node with the largest PageRank value is', '因此 PageRank 值最大的节点是', output)
    output = re.sub(r'The answer is', '答案是', output)

    sample['output'] = output
    return sample


def main():
    # 读取中文数据
    print('读取中文数据...')
    input_file = '/mnt/yrfs/GraphAgent_model/zy/data/graphInstruct/graphinstruct_chinese_combined_text.json'
    with open(input_file, 'r', encoding='utf-8') as f:
        zh_data = json.load(f)

    print(f'原始数据: {len(zh_data):,} 条\n')

    # 修复数据
    print('开始修复...\n')

    fixed_count = {'MAXIMUM_FLOW': 0, 'shortest_path': 0, 'page_rank': 0}

    for sample in tqdm(zh_data, desc='修复中'):
        task = sample['task_type']

        if task == 'MAXIMUM_FLOW':
            fix_maximum_flow_sample(sample)
            fixed_count['MAXIMUM_FLOW'] += 1

        elif task == 'shortest_path':
            fix_shortest_path_sample(sample)
            fixed_count['shortest_path'] += 1

        elif task == 'page_rank':
            fix_page_rank_sample(sample)
            fixed_count['page_rank'] += 1

    print(f'\n修复完成:')
    for task, count in fixed_count.items():
        if count > 0:
            print(f'  {task:20s}: {count:,} 条')

    # 保存修复后的数据
    output_file = '/mnt/yrfs/GraphAgent_model/zy/data/graphInstruct/graphinstruct_chinese_combined_text.json'
    print(f'\n保存到: {output_file}')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(zh_data, f, ensure_ascii=False, indent=2)

    # 验证修复效果
    print('\n验证修复效果...')

    def count_chinese_chars(text):
        return sum(1 for c in text if '\u4e00' <= c <= '\u9fff')

    from collections import defaultdict
    task_stats = defaultdict(lambda: {'total': 0, 'zh_chars': 0, 'count': 0})

    for sample in zh_data:
        task = sample['task_type']
        text = sample['instruction'] + sample['output']
        task_stats[task]['total'] += len(text)
        task_stats[task]['zh_chars'] += count_chinese_chars(text)
        task_stats[task]['count'] += 1

    print(f'\n{"任务":<30s} {"样本数":>8s} {"修复前占比":>12s} {"修复后占比":>12s}')
    print('-' * 70)

    original_ratios = {
        'MAXIMUM_FLOW': 7.1,
        'shortest_path': 7.5,
        'page_rank': 4.6
    }

    for task in ['MAXIMUM_FLOW', 'shortest_path', 'page_rank']:
        if task in task_stats:
            stats = task_stats[task]
            new_ratio = stats['zh_chars'] / stats['total'] * 100 if stats['total'] > 0 else 0
            old_ratio = original_ratios.get(task, 0)
            print(f'{task:<30s} {stats["count"]:>8,d} {old_ratio:>11.1f}% {new_ratio:>11.1f}%')

    print('\n✓ 修复完成！')


if __name__ == '__main__':
    main()
