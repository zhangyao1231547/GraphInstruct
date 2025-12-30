#!/usr/bin/env python3
"""
将19万GraphInstruct英文训练数据转换为中文版本

输入: data/converted_fixed/merged_graphinstruct.json (190000样本)
输出: data/training/chinese/graphinstruct_chinese_190k.json

转换内容:
1. 保持图结构不变
2. 将英文推理步骤和答案转换为中文
"""

import json
import os
import re
import argparse
import logging
from typing import Dict, List
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 任务类型到中文的映射
TASK_TYPE_ZH = {
    'BFS': '广度优先搜索',
    'DFS': '深度优先搜索',
    'bipartite': '二分图判断',
    'clustering_coefficient': '聚类系数',
    'common_neighbor': '共同邻居',
    'connected_component': '连通分量',
    'connectivity': '连通性',
    'cycle': '环检测',
    'degree': '节点度数',
    'diameter': '图直径',
    'edge': '边存在性',
    'jaccard': 'Jaccard相似度',
    'maximum_flow': '最大流',
    'MST': '最小生成树',
    'neighbor': '邻居节点',
    'page_rank': 'PageRank',
    'predecessor': '前驱节点',
    'shortest_path': '最短路径',
    'topological_sort': '拓扑排序'
}

# 英文到中文的翻译映射
EN_TO_ZH_PATTERNS = [
    # BFS/DFS 相关
    (r"Let's run breadth-first search \(BFS\) step by step\.", "让我们逐步执行广度优先搜索(BFS)。"),
    (r"Let's run depth-first search \(DFS\) step by step\.", "让我们逐步执行深度优先搜索(DFS)。"),
    (r"Visit node (\d+)\.", r"访问节点 \1。"),
    (r"Unvisited neighbors of node (\d+) are \[([^\]]+)\]\.", r"节点 \1 的未访问邻居是 [\2]。"),
    (r"Unvisited neighbors of node (\d+) are \[([^\]]+)\]", r"节点 \1 的未访问邻居是 [\2]"),
    (r"So the BFS traversal is", "因此BFS遍历顺序是"),
    (r"So the DFS traversal is", "因此DFS遍历顺序是"),
    (r"Node (\d+) has no unvisited neighbors\.", r"节点 \1 没有未访问的邻居。"),
    (r"Backtrack to node (\d+)\.", r"回溯到节点 \1。"),

    # 二分图匹配相关 (Hungarian algorithm)
    (r"To find a maximum matching in the bipartite graph, let's run the Hungarian algorithm step by step\.",
     "为了找到二分图的最大匹配，让我们逐步运行匈牙利算法。"),
    (r"Firstly, initialize an empty matching \{\}\.", "首先，初始化一个空匹配 {}。"),
    (r"Iterate over nodes in set 1:", "遍历集合1中的节点:"),
    (r"Search from node (\d+)\. Updated matching: \{([^}]+)\}\.", r"从节点 \1 搜索。更新匹配: {\2}。"),
    (r"So the maximum matching has size (\d+)\.", r"因此最大匹配的大小是 \1。"),
    (r"The maximum matching is", "最大匹配是"),

    # 二分图检查相关
    (r"Let's check if the graph is bipartite step by step\.", "让我们逐步检查图是否为二分图。"),
    (r"Let's check if the graph is bipartite\.", "让我们检查图是否为二分图。"),
    (r"Start by coloring node (\d+) with color (\d+)\.", r"首先将节点 \1 染成颜色 \2。"),
    (r"Color node (\d+) with color (\d+)\.", r"将节点 \1 染成颜色 \2。"),
    (r"Node (\d+) is already colored with color (\d+)\.", r"节点 \1 已经被染成颜色 \2。"),
    (r"The graph is bipartite\.", "该图是二分图。"),
    (r"The graph is not bipartite", "该图不是二分图"),
    (r"Conflict found", "发现冲突"),
    (r"So the answer is", "因此答案是"),

    # 聚类系数相关
    (r"Let's calculate the clustering coefficient of node (\d+)\.", r"让我们计算节点 \1 的聚类系数。"),
    (r"Node (\d+) has (\d+) neighbors?: \[([^\]]+)\]\.", r"节点 \1 有 \2 个邻居: [\3]。"),
    (r"The number of edges between neighbors is (\d+)\.", r"邻居之间的边数是 \1。"),
    (r"The maximum possible edges between (\d+) neighbors is (\d+)\.", r"\1 个邻居之间的最大可能边数是 \2。"),
    (r"The clustering coefficient is (\d+)/(\d+) = ([0-9.]+)\.", r"聚类系数是 \1/\2 = \3。"),
    (r"The clustering coefficient is", "聚类系数是"),

    # 共同邻居相关
    (r"Let's find the common neighbors of nodes? (\d+) and (\d+)\.", r"让我们找出节点 \1 和 \2 的共同邻居。"),
    (r"Neighbors of node (\d+): \[([^\]]+)\]\.", r"节点 \1 的邻居: [\2]。"),
    (r"Neighbors of node (\d+): \[([^\]]+)\]", r"节点 \1 的邻居: [\2]"),
    (r"The common neighbors are", "共同邻居是"),
    (r"There are no common neighbors", "没有共同邻居"),

    # 连通分量相关
    (r"Let's find all connected components\.", "让我们找出所有连通分量。"),
    (r"Starting from node (\d+), we can reach nodes?: \[([^\]]+)\]\.", r"从节点 \1 开始，我们可以到达节点: [\2]。"),
    (r"This forms component (\d+)\.", r"这形成了分量 \1。"),
    (r"The connected components are", "连通分量是"),
    (r"Component (\d+): \[([^\]]+)\]", r"分量 \1: [\2]"),

    # 连通性相关
    (r"Let's check if node (\d+) can reach node (\d+)\.", r"让我们检查节点 \1 是否能到达节点 \2。"),
    (r"Starting BFS from node (\d+)\.", r"从节点 \1 开始BFS。"),
    (r"We can reach node (\d+) from node (\d+)\.", r"我们可以从节点 \2 到达节点 \1。"),
    (r"We cannot reach node (\d+) from node (\d+)\.", r"我们无法从节点 \2 到达节点 \1。"),
    (r"Node (\d+) is reachable\.", r"节点 \1 可达。"),
    (r"Node (\d+) is not reachable\.", r"节点 \1 不可达。"),

    # 环检测相关
    (r"Let's check if the graph contains a cycle\.", "让我们检查图中是否存在环。"),
    (r"Using DFS to detect cycles\.", "使用DFS检测环。"),
    (r"A cycle is detected", "检测到环"),
    (r"No cycle is detected", "未检测到环"),
    (r"The graph contains a cycle", "图中存在环"),
    (r"The graph does not contain a cycle", "图中不存在环"),

    # 度数相关
    (r"Let's calculate the degree of node (\d+)\.", r"让我们计算节点 \1 的度数。"),
    (r"Node (\d+) is connected to nodes?: \[([^\]]+)\]\.", r"节点 \1 连接到节点: [\2]。"),
    (r"The degree of node (\d+) is (\d+)\.", r"节点 \1 的度数是 \2。"),
    (r"The in-degree of node (\d+) is (\d+)\.", r"节点 \1 的入度是 \2。"),
    (r"The out-degree of node (\d+) is (\d+)\.", r"节点 \1 的出度是 \2。"),

    # 直径相关
    (r"Let's calculate the diameter of the graph\.", "让我们计算图的直径。"),
    (r"The shortest path from node (\d+) to node (\d+) has length (\d+)\.", r"从节点 \1 到节点 \2 的最短路径长度是 \3。"),
    (r"The diameter of the graph is", "图的直径是"),
    (r"The eccentricity of node (\d+) is (\d+)\.", r"节点 \1 的离心率是 \2。"),

    # 边存在性相关
    (r"Let's check if there is an edge between nodes? (\d+) and (\d+)\.", r"让我们检查节点 \1 和 \2 之间是否存在边。"),
    (r"Looking at the adjacency list of node (\d+): \[([^\]]+)\]\.", r"查看节点 \1 的邻接表: [\2]。"),
    (r"Yes, there is an edge between nodes? (\d+) and (\d+)\.", r"是的，节点 \1 和 \2 之间存在边。"),
    (r"No, there is no edge between nodes? (\d+) and (\d+)\.", r"不，节点 \1 和 \2 之间不存在边。"),
    (r"There is an edge", "存在边"),
    (r"There is no edge", "不存在边"),

    # Jaccard相似度相关
    (r"Let's calculate the Jaccard similarity between nodes? (\d+) and (\d+)\.", r"让我们计算节点 \1 和 \2 之间的Jaccard相似度。"),
    (r"Let's calculate the Jaccard coefficient step by step\.", "让我们逐步计算Jaccard系数。"),
    (r"The intersection of neighbors is \[([^\]]+)\] with size (\d+)\.", r"邻居的交集是 [\1]，大小为 \2。"),
    (r"The union of neighbors is \[([^\]]+)\] with size (\d+)\.", r"邻居的并集是 [\1]，大小为 \2。"),
    (r"Jaccard similarity = (\d+)/(\d+) = ([0-9.]+)\.", r"Jaccard相似度 = \1/\2 = \3。"),
    (r"The Jaccard similarity is", "Jaccard相似度是"),
    (r"So the Jaccard coefficient is", "因此Jaccard系数是"),
    (r"The neighbors of node (\d+) are \[([^\]]+)\]\.", r"节点 \1 的邻居是 [\2]。"),
    (r"The neighbors of node (\d+) are \[([^\]]+)\]", r"节点 \1 的邻居是 [\2]"),
    (r"The intersection of the neighbors of node (\d+) and node (\d+) is \[([^\]]+)\]\.",
     r"节点 \1 和节点 \2 邻居的交集是 [\3]。"),
    (r"The union of the neighbors of node (\d+) and node (\d+) is \[([^\]]+)\]\.",
     r"节点 \1 和节点 \2 邻居的并集是 [\3]。"),

    # 最大流相关
    (r"Let's calculate the maximum flow from node (\d+) to node (\d+)\.", r"让我们计算从节点 \1 到节点 \2 的最大流。"),
    (r"Using Ford-Fulkerson algorithm\.", "使用Ford-Fulkerson算法。"),
    (r"We will use the Edmonds-Karp algorithm\.", "我们将使用Edmonds-Karp算法。"),
    (r"Found augmenting path: \[([^\]]+)\] with capacity (\d+)\.", r"找到增广路径: [\1]，容量为 \2。"),
    (r"Updated the flow along this path\. Current total flow is (\d+)\.", r"沿此路径更新流量。当前总流量是 \1。"),
    (r"Updated the flow along this path\.", "沿此路径更新流量。"),
    (r"The maximum flow is", "最大流是"),
    (r"Total flow: (\d+)", r"总流量: \1"),
    (r"So the maximum flow from node (\d+) to node (\d+) is (\d+)\.", r"因此从节点 \1 到节点 \2 的最大流是 \3。"),

    # MST相关
    (r"Let's find the minimum spanning tree\.", "让我们找出最小生成树。"),
    (r"Using Kruskal's algorithm\.", "使用Kruskal算法。"),
    (r"Using Prim's algorithm\.", "使用Prim算法。"),
    (r"Add edge \((\d+), (\d+)\) with weight ([0-9.]+)\.", r"添加边 (\1, \2)，权重为 \3。"),
    (r"The MST edges are", "最小生成树的边是"),
    (r"The total weight of MST is", "最小生成树的总权重是"),
    (r"These edges make up its minimum spanning tree\.", "这些边构成了它的最小生成树。"),
    (r"The total weight of the MST is", "最小生成树的总权重是"),

    # 邻居节点相关
    (r"Let's find the neighbors of node (\d+)\.", r"让我们找出节点 \1 的邻居。"),
    (r"From the adjacency list, node (\d+) is connected to \[([^\]]+)\]\.", r"从邻接表可知，节点 \1 连接到 [\2]。"),
    (r"The neighbors of node (\d+) are", r"节点 \1 的邻居是"),

    # PageRank相关
    (r"Let's calculate the PageRank of node (\d+)\.", r"让我们计算节点 \1 的PageRank值。"),
    (r"Let's calculate PageRank step by step\.", "让我们逐步计算PageRank。"),
    (r"All the nodes:", "所有节点:"),
    (r"The normalized adjacency matrix M is:", "归一化邻接矩阵M是:"),
    (r"According to M_hat = \(d \* M \+ \(1 - d\) / N\), where d is the damping factor ([0-9.]+) and N is the number of nodes, the transition probability is:",
     r"根据 M_hat = (d * M + (1 - d) / N)，其中d是阻尼因子 \1，N是节点数，转移概率是:"),
    (r"PageRank values of each node from node (\d+) to node (\d+) at round (\d+) are:",
     r"第 \3 轮从节点 \1 到节点 \2 的各节点PageRank值是:"),
    (r"Finally, after (\d+) rounds of iteration, the PageRank values of each node from node (\d+) to node (\d+) are:",
     r"最后，经过 \1 轮迭代后，从节点 \2 到节点 \3 的各节点PageRank值是:"),
    (r"So the node with the largest PageRank value is", "因此PageRank值最大的节点是"),
    (r"After (\d+) iterations, the PageRank values converge\.", r"经过 \1 次迭代后，PageRank值收敛。"),
    (r"The PageRank of node (\d+) is", r"节点 \1 的PageRank值是"),
    (r"PageRank values:", "PageRank值:"),

    # 前驱节点相关
    (r"Let's find the predecessors of node (\d+)\.", r"让我们找出节点 \1 的前驱节点。"),
    (r"Looking at incoming edges to node (\d+)\.", r"查看指向节点 \1 的入边。"),
    (r"The predecessors of node (\d+) are", r"节点 \1 的前驱节点是"),
    (r"Node (\d+) has no predecessors", r"节点 \1 没有前驱节点"),

    # 最短路径相关
    (r"Let's find the shortest path from node (\d+) to node (\d+)\.", r"让我们找出从节点 \1 到节点 \2 的最短路径。"),
    (r"Using Dijkstra's algorithm\.", "使用Dijkstra算法。"),
    (r"We can use the Dijsktra's algorithm to find the shortest path from node (\d+) to node (\d+)\.",
     r"我们可以使用Dijkstra算法找出从节点 \1 到节点 \2 的最短路径。"),
    (r"We can use the Dijkstra's algorithm", "我们可以使用Dijkstra算法"),
    (r"Using BFS for unweighted graph\.", "使用BFS处理无权图。"),
    (r"The shortest path is \[([^\]]+)\] with length (\d+)\.", r"最短路径是 [\1]，长度为 \2。"),
    (r"The shortest path is", "最短路径是"),
    (r"The path length is", "路径长度是"),
    (r"No path exists", "不存在路径"),
    (r"The shortest distance from node (\d+) to node (\d+) is", r"从节点 \1 到节点 \2 的最短距离是"),
    (r"So the shortest path from node (\d+) to node (\d+) is", r"因此从节点 \1 到节点 \2 的最短路径是"),

    # 拓扑排序相关
    (r"Let's perform topological sort on the directed graph\.", "让我们对有向图执行拓扑排序。"),
    (r"We can use the topological sort to find a valid ordering of nodes\.", "我们可以使用拓扑排序找出节点的有效顺序。"),
    (r"Using Kahn's algorithm\.", "使用Kahn算法。"),
    (r"Node (\d+) has in-degree 0, add to result\.", r"节点 \1 的入度为0，加入结果。"),
    (r"The topological order is", "拓扑排序顺序是"),
    (r"Remove node (\d+) and update in-degrees\.", r"移除节点 \1 并更新入度。"),
    (r"So the topological order is", "因此拓扑排序顺序是"),
    (r"A valid topological order is", "一个有效的拓扑排序顺序是"),

    # 连通分量相关 (Tarjan)
    (r"We can use the Tarjan's algorithm to find strongly connected components\.",
     "我们可以使用Tarjan算法找出强连通分量。"),
    (r"Using Tarjan's algorithm\.", "使用Tarjan算法。"),
    (r"Tarjan algorithm\.", "使用Tarjan算法。"),
    (r"The strongly connected components are", "强连通分量是"),
    (r"So the strongly connected components are", "因此强连通分量是"),

    # 直径相关
    (r"Let's calculate the diameter of the graph\.", "让我们计算图的直径。"),
    (r"The shortest path from node (\d+) to node (\d+) has length (\d+)\.", r"从节点 \1 到节点 \2 的最短路径长度是 \3。"),
    (r"The diameter of the graph is", "图的直径是"),
    (r"The eccentricity of node (\d+) is (\d+)\.", r"节点 \1 的离心率是 \2。"),
    (r"So the diameter of the graph is", "因此图的直径是"),
    (r"The longest shortest path in the graph has length", "图中最长的最短路径长度是"),

    # 环检测相关
    (r"Let's check if the graph contains a cycle\.", "让我们检查图中是否存在环。"),
    (r"Using DFS to detect cycles\.", "使用DFS检测环。"),
    (r"A cycle is detected", "检测到环"),
    (r"No cycle is detected", "未检测到环"),
    (r"The graph contains a cycle", "图中存在环"),
    (r"The graph does not contain a cycle", "图中不存在环"),
    (r"So the graph contains a cycle", "因此图中存在环"),
    (r"So the graph does not contain a cycle", "因此图中不存在环"),

    # 聚类系数相关 (补充)
    (r"Let's calculate the clustering coefficient step by step\.", "让我们逐步计算聚类系数。"),
    (r"The number of triangles T is", "三角形数量T是"),
    (r"T is the number of triangles", "T是三角形的数量"),
    (r"So the clustering coefficient of node (\d+) is", r"因此节点 \1 的聚类系数是"),

    # 通用模式
    (r"Step (\d+):", r"步骤 \1:"),
    (r"Therefore,", "因此，"),
    (r"Thus,", "因此，"),
    (r"Hence,", "因此，"),
    (r"Finally,", "最后，"),
    (r"In conclusion,", "总之，"),
    (r"The answer is", "答案是"),
    (r"True", "True"),  # 保持布尔值不变
    (r"False", "False"),
    (r"Yes\b", "是"),
    (r"No\b", "否"),
    (r"Firstly,", "首先，"),
    (r"Secondly,", "其次，"),
    (r"First,", "首先，"),
    (r"Second,", "其次，"),
    (r"Next,", "接下来，"),
    (r"Then,", "然后，"),
]


def translate_response(text: str, task_type: str) -> str:
    """将英文回答翻译为中文"""
    result = text

    # 应用翻译模式
    for pattern, replacement in EN_TO_ZH_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


def convert_sample(sample: Dict) -> Dict:
    """转换单个样本为中文"""
    converted = sample.copy()

    # 获取任务类型
    task_type = sample.get('task_type', '')

    # 转换对话
    if 'conversations' in converted:
        new_conversations = []
        for conv in converted['conversations']:
            new_conv = conv.copy()
            if conv['from'] == 'gpt':
                # 翻译GPT的回答
                new_conv['value'] = translate_response(conv['value'], task_type)
            new_conversations.append(new_conv)
        converted['conversations'] = new_conversations

    # 添加语言标记
    converted['language'] = 'zh'

    # 更新ID
    if 'id' in converted:
        converted['id'] = converted['id'].replace('graphinstruct_', 'graphinstruct_zh_')

    return converted


def process_batch(batch: List[Dict]) -> List[Dict]:
    """处理一批样本"""
    return [convert_sample(sample) for sample in batch]


def convert_dataset(
    input_path: str,
    output_path: str,
    num_workers: int = 8,
    batch_size: int = 1000
):
    """转换整个数据集"""
    logger.info(f"Loading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_samples = len(data)
    logger.info(f"Loaded {total_samples} samples")

    # 分批处理
    batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

    converted_data = []

    logger.info(f"Converting {total_samples} samples using {num_workers} workers...")

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}

            with tqdm(total=len(batches), desc="Converting") as pbar:
                for future in as_completed(futures):
                    batch_result = future.result()
                    converted_data.extend(batch_result)
                    pbar.update(1)
    else:
        for batch in tqdm(batches, desc="Converting"):
            converted_data.extend(process_batch(batch))

    # 按原始顺序排序
    converted_data.sort(key=lambda x: x.get('id', ''))

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"Saving {len(converted_data)} converted samples to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

    # 统计任务分布
    task_counts = {}
    for sample in converted_data:
        task_type = sample.get('task_type', 'unknown')
        task_counts[task_type] = task_counts.get(task_type, 0) + 1

    logger.info("\n任务分布:")
    for task_type, count in sorted(task_counts.items()):
        zh_name = TASK_TYPE_ZH.get(task_type, task_type)
        logger.info(f"  {task_type} ({zh_name}): {count}")

    logger.info(f"\n转换完成! 共 {len(converted_data)} 样本")

    return converted_data


def main():
    parser = argparse.ArgumentParser(description='将GraphInstruct英文数据转换为中文')
    parser.add_argument('--input', type=str,
                        default='/nvme0/work/workspaces-zy/GraphInstruct/data/converted_fixed/merged_graphinstruct.json',
                        help='输入JSON文件路径')
    parser.add_argument('--output', type=str,
                        default='/nvme0/work/workspaces-zy/GraphInstruct/data/training/chinese/graphinstruct_chinese_190k.json',
                        help='输出JSON文件路径')
    parser.add_argument('--workers', type=int, default=8,
                        help='并行工作进程数')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='每批处理的样本数')

    args = parser.parse_args()

    convert_dataset(
        input_path=args.input,
        output_path=args.output,
        num_workers=args.workers,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
