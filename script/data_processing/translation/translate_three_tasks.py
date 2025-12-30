#!/usr/bin/env python3
"""
从英文数据中提取 shortest_path, page_rank, mst 三个任务并翻译成中文
替换中文数据中对应的任务
"""

import json
import re
from tqdm import tqdm

def translate_shortest_path_instruction(text: str) -> str:
    """翻译 shortest_path 的 instruction"""
    result = text

    # "In a weighted undirected graph with X nodes:"
    result = re.sub(r'In a weighted undirected graph with (\d+) nodes?:',
                   r'在一个包含 \1 个节点的加权无向图中：', result)

    # "Node <X> is connected to nodes <Y> (weight: Z), ..."
    result = re.sub(r'Node <(\d+)> is connected to nodes? ([^.]+)\.',
                   lambda m: f'节点 <{m.group(1)}> 连接到节点 {translate_connections(m.group(2))}。',
                   result)

    # "Calculate the distance of the shortest path from node <X> to node <Y>."
    result = re.sub(r'Calculate the distance of the shortest path from node <(\d+)> to node <(\d+)>\.',
                   r'计算从节点 <\1> 到节点 <\2> 的最短路径距离。', result)

    return result

def translate_connections(conn_text: str) -> str:
    """翻译连接列表"""
    # 替换 "(weight: X)" 为 "(权重: X)"
    result = re.sub(r'\(weight: (\d+)\)', r'(权重: \1)', conn_text)
    return result

def translate_mst_instruction(text: str) -> str:
    """翻译 mst 的 instruction"""
    result = text

    # "Consider an undirected graph with X nodes and Y weighted edges."
    result = re.sub(r'Consider an undirected graph with (\d+) nodes? and (\d+) weighted edges?\.',
                   r'考虑一个包含 \1 个节点和 \2 条加权边的无向图。', result)

    # "Node X is connected to nodes Y (weight: Z), ..."
    result = re.sub(r'Node (\d+) is connected to nodes? ([^.]+)\.',
                   lambda m: f'节点 {m.group(1)} 连接到节点 {translate_connections(m.group(2))}。',
                   result)

    # "Output the total weight of the minimum spanning tree (MST) for this graph."
    result = re.sub(r'Output the total weight of the minimum spanning tree \(MST\) for this graph\.',
                   r'输出该图的最小生成树 (MST) 的总权重。', result)

    return result

def translate_pagerank_instruction(text: str) -> str:
    """翻译 page_rank 的 instruction"""
    result = text

    # 基础翻译
    result = re.sub(r'The following is a question related to the graph reasoning task PAGE_RANK\.',
                   '以下是一个关于图推理任务 PAGE_RANK (页面排名) 的问题。', result)

    result = re.sub(r'Please answer the question step by step and format your final response as <<<ANSWER>>>',
                   '请逐步分析并解答，最终答案请用 <<<答案>>> 的格式给出', result)

    result = re.sub(r'For example, if your answer is ([^,]+), please format your answer as <<<\1>>>',
                   r'例如，如果你的答案是 \1，请将答案格式化为 <<<\1>>>', result)

    # "Given a undirected graph with the following connections:"
    result = re.sub(r'Given a(?:n)? (undirected|directed) graph with the following connections:',
                   lambda m: f'给定一个{"无向图" if m.group(1) == "undirected" else "有向图"}，其连接关系如下：',
                   result)

    # "Node X is connected to nodes Y, Z."
    result = re.sub(r'Node (\d+) is connected to nodes? ([^.]+)\.',
                   r'节点 \1 连接到节点 \2。', result)

    # "Which node has the largest PageRank value?"
    result = re.sub(r'Which node has the largest PageRank value\?',
                   '哪个节点的 PageRank 值最大？', result)

    # "The dampling factor is X."
    result = re.sub(r'The dampl?ing factor is ([0-9.]+)\.',
                   r'阻尼因子是 \1。', result)

    # "The number of iterations is X."
    result = re.sub(r'The number of iterations is (\d+)\.',
                   r'迭代次数是 \1。', result)

    # "The initial PageRank values..."
    result = re.sub(r'The initial PageRank values for all nodes are initialized equally as 1/N, where N is the number of nodes\.',
                   '所有节点的初始 PageRank 值均初始化为 1/N，其中 N 是节点数。', result)

    return result

def translate_output(text: str, task_type: str) -> str:
    """翻译 output 部分"""
    result = text

    # 通用翻译
    result = re.sub(r"Let's solve (it|this) step by step\.",
                   "让我们逐步解决。", result)
    result = re.sub(r"Let's calculate PageRank step by step\.",
                   "让我们逐步计算 PageRank。", result)
    result = re.sub(r"Let's solve this step by step using Prim's algorithm",
                   "让我们使用 Prim 算法逐步解决", result)

    result = re.sub(r'We can use the Dijsktra algorithm\.',
                   '我们可以使用 Dijkstra 算法。', result)
    result = re.sub(r"using Prim's algorithm", "使用 Prim 算法", result)

    # Round/Step 翻译
    result = re.sub(r'Round (\d+):', r'第 \1 轮：', result)
    result = re.sub(r'Step ?(\d+):', r'步骤 \1：', result)
    result = re.sub(r'Step (\d+):', r'步骤 \1：', result)

    # 其他常见翻译
    result = re.sub(r'All the nodes?:', '所有节点：', result)
    result = re.sub(r'The unvisited nodes are:', '未访问的节点：', result)
    result = re.sub(r'Current MST nodes?:', '当前 MST 节点：', result)
    result = re.sub(r'Candidate edges to unvisited nodes?:', '候选边到未访问节点：', result)
    result = re.sub(r'Select minimum weight edge:', '选择最小权重边：', result)
    result = re.sub(r'Add node (\d+) to MST\.', r'将节点 \1 添加到 MST。', result)

    result = re.sub(r'Therefore,', '因此，', result)
    result = re.sub(r'Thus,', '因此，', result)
    result = re.sub(r'Finally,', '最后，', result)
    result = re.sub(r'The answer is', '答案是', result)

    # 答案标记
    result = re.sub(r'<<<ANSWER>>>', '<<<答案>>>', result)

    return result

def translate_sample(sample: dict) -> dict:
    """翻译单个样本"""
    task_type = sample['task_type']

    # 翻译 instruction
    if task_type == 'shortest_path':
        instruction = translate_shortest_path_instruction(sample['instruction'])
    elif task_type == 'page_rank':
        instruction = translate_pagerank_instruction(sample['instruction'])
    elif task_type == 'mst':
        instruction = translate_mst_instruction(sample['instruction'])
    else:
        instruction = sample['instruction']

    # 翻译 output
    output = translate_output(sample['output'], task_type)

    return {
        'id': sample['id'] + '_zh',
        'task_type': task_type,
        'instruction': instruction,
        'output': output
    }

def main():
    # 读取英文数据
    print('读取英文数据...')
    with open('/mnt/yrfs/GraphAgent_model/zy/data/graphInstruct/graphinstruct_english_combined_text.json', 'r') as f:
        en_data = json.load(f)

    # 读取现有中文数据
    print('读取现有中文数据...')
    with open('/mnt/yrfs/GraphAgent_model/zy/data/graphInstruct/graphinstruct_chinese_combined_text.json', 'r') as f:
        zh_data = json.load(f)

    print(f'英文数据: {len(en_data):,} 条')
    print(f'中文数据: {len(zh_data):,} 条\n')

    # 提取需要翻译的三个任务
    target_tasks = ['shortest_path', 'page_rank', 'mst']
    print(f'提取并翻译这三个任务: {target_tasks}\n')

    new_zh_samples = []
    for sample in tqdm(en_data, desc='翻译中'):
        if sample['task_type'] in target_tasks:
            translated = translate_sample(sample)
            new_zh_samples.append(translated)

    print(f'\n翻译完成: {len(new_zh_samples):,} 条\n')

    # 移除中文数据中的这三个任务
    print('移除中文数据中的旧任务...')
    zh_data_filtered = [s for s in zh_data if s['task_type'] not in target_tasks]
    print(f'移除后: {len(zh_data_filtered):,} 条\n')

    # 合并
    print('合并数据...')
    final_data = zh_data_filtered + new_zh_samples
    print(f'最终数据: {len(final_data):,} 条\n')

    # 保存
    output_file = '/mnt/yrfs/GraphAgent_model/zy/data/graphInstruct/graphinstruct_chinese_combined_text.json'
    print(f'保存到: {output_file}')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    # 统计
    from collections import Counter
    task_counts = Counter(s['task_type'] for s in final_data)
    print(f'\n最终任务分布:')
    for task, count in sorted(task_counts.items()):
        print(f'  {task:25s}: {count:6,d}')

    print(f'\n✓ 完成！三个任务已从英文数据翻译并替换')

if __name__ == '__main__':
    main()
