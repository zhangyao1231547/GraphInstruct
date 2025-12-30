#!/usr/bin/env python3
"""
GraphInstruct 数据转换脚本
将 GraphInstruct 的 Alpaca 格式数据转换为 GraphAgent 训练所需的格式

GraphInstruct 格式:
{
    "instruction": "Given a undirected graph... Node X connected to nodes Y, Z...",
    "input": "",
    "output": "Let's solve it step by step... <<<answer>>>"
}

GraphAgent 格式:
{
    "id": "graphinstruct_task_xxx",
    "graph": {
        "node_list": [...],
        "edge_index": [[...], [...]]
    },
    "conversations": [
        {"from": "human", "value": "...<graph>..."},
        {"from": "gpt", "value": "..."}
    ]
}
"""

import json
import re
import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import argparse
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GraphData:
    """图数据结构"""
    nodes: List[int]
    edges: List[Tuple[int, int]]
    weights: Optional[List[float]] = None
    is_directed: bool = False


class GraphTextParser:
    """从文本描述中解析图结构"""

    # 节点连接模式 (无权重)
    # "Node X is connected to nodes Y, Z, W."
    # "Node X is connected to node Y."
    PATTERN_UNDIRECTED = re.compile(
        r'Node\s+(\d+)\s+is\s+connected\s+to\s+nodes?\s+([\d,\s]+)\.',
        re.IGNORECASE
    )

    # 节点连接模式 (带权重)
    # "Node X is connected to nodes Y (weight: W1), Z (weight: W2)."
    PATTERN_WEIGHTED = re.compile(
        r'Node\s+(\d+)\s+is\s+connected\s+to\s+nodes?\s+(.+?)\.',
        re.IGNORECASE
    )

    # 权重提取模式
    PATTERN_WEIGHT = re.compile(r'(\d+)\s*\(weight:\s*([\d.]+)\)')

    # 图类型检测
    PATTERN_DIRECTED = re.compile(r'directed\s+graph', re.IGNORECASE)
    PATTERN_UNDIRECTED_TYPE = re.compile(r'undirected\s+graph', re.IGNORECASE)

    @classmethod
    def parse(cls, text: str) -> GraphData:
        """从文本中解析图结构"""
        is_directed = bool(cls.PATTERN_DIRECTED.search(text))

        nodes = set()
        edges = []
        weights = []
        has_weights = False

        # 尝试解析带权重的连接
        for line in text.split('\n'):
            line = line.strip()
            if not line.startswith('Node'):
                continue

            # 提取源节点
            match = re.match(r'Node\s+(\d+)\s+is\s+connected\s+to', line)
            if not match:
                continue
            source = int(match.group(1))
            nodes.add(source)

            # 提取目标节点和权重
            connection_part = line[match.end():].strip()

            # 检查是否有权重
            weight_matches = cls.PATTERN_WEIGHT.findall(connection_part)

            if weight_matches:
                # 带权重的边
                has_weights = True
                for target_str, weight_str in weight_matches:
                    target = int(target_str)
                    weight = float(weight_str)
                    nodes.add(target)
                    edges.append((source, target))
                    weights.append(weight)

                    # 无向图添加反向边
                    if not is_directed:
                        edges.append((target, source))
                        weights.append(weight)
            else:
                # 无权重的边 - 提取所有数字
                targets = re.findall(r'\d+', connection_part)
                for target_str in targets:
                    target = int(target_str)
                    nodes.add(target)
                    edges.append((source, target))

                    # 无向图添加反向边
                    if not is_directed:
                        edges.append((target, source))

        # 去重边（保留第一次出现的）
        seen_edges = set()
        unique_edges = []
        unique_weights = []

        for i, edge in enumerate(edges):
            if edge not in seen_edges:
                seen_edges.add(edge)
                unique_edges.append(edge)
                if has_weights and i < len(weights):
                    unique_weights.append(weights[i])

        return GraphData(
            nodes=sorted(nodes),
            edges=unique_edges,
            weights=unique_weights if has_weights else None,
            is_directed=is_directed
        )


class TaskConverter:
    """任务类型转换器基类"""

    TASK_TEMPLATES = {
        'BFS': {
            'zh_prompt': '给定以下图结构:\n<graph>\n\n请从节点 {start_node} 开始，执行广度优先搜索(BFS)遍历，输出访问节点的顺序。',
            'en_prompt': 'Given the following graph structure:\n<graph>\n\nStarting from node {start_node}, perform a breadth-first search (BFS) traversal and output the sequence of visited nodes.'
        },
        'DFS': {
            'zh_prompt': '给定以下图结构:\n<graph>\n\n请从节点 {start_node} 开始，执行深度优先搜索(DFS)遍历，输出访问节点的顺序。',
            'en_prompt': 'Given the following graph structure:\n<graph>\n\nStarting from node {start_node}, perform a depth-first search (DFS) traversal and output the sequence of visited nodes.'
        },
        'SHORTEST_PATH': {
            'zh_prompt': '给定以下图结构:\n<graph>\n\n请计算从节点 {source} 到节点 {target} 的最短路径距离。',
            'en_prompt': 'Given the following graph structure:\n<graph>\n\nCalculate the shortest path distance from node {source} to node {target}.'
        },
        'DEGREE': {
            'zh_prompt': '给定以下图结构:\n<graph>\n\n请计算节点 {node} 的度数。',
            'en_prompt': 'Given the following graph structure:\n<graph>\n\nCalculate the degree of node {node}.'
        },
        'NEIGHBOR': {
            'zh_prompt': '给定以下图结构:\n<graph>\n\n请列出节点 {node} 的所有邻居节点。',
            'en_prompt': 'Given the following graph structure:\n<graph>\n\nList all neighbor nodes of node {node}.'
        },
        'CYCLE': {
            'zh_prompt': '给定以下图结构:\n<graph>\n\n请判断该图中是否存在环。',
            'en_prompt': 'Given the following graph structure:\n<graph>\n\nDetermine whether the graph contains a cycle.'
        },
        'CONNECTIVITY': {
            'zh_prompt': '给定以下图结构:\n<graph>\n\n请判断节点 {source} 和节点 {target} 是否连通。',
            'en_prompt': 'Given the following graph structure:\n<graph>\n\nDetermine whether node {source} and node {target} are connected.'
        },
        'BIPARTITE': {
            'zh_prompt': '给定以下图结构:\n<graph>\n\n请判断该图是否为二分图。',
            'en_prompt': 'Given the following graph structure:\n<graph>\n\nDetermine whether the graph is bipartite.'
        },
        'CONNECTED_COMPONENT': {
            'zh_prompt': '给定以下图结构:\n<graph>\n\n请计算该图的连通分量数量。',
            'en_prompt': 'Given the following graph structure:\n<graph>\n\nCalculate the number of connected components in the graph.'
        },
        'MST': {
            'zh_prompt': '给定以下加权图结构:\n<graph>\n\n请计算最小生成树的总权重。',
            'en_prompt': 'Given the following weighted graph structure:\n<graph>\n\nCalculate the total weight of the minimum spanning tree.'
        },
        'MAXIMUM_FLOW': {
            'zh_prompt': '给定以下图结构:\n<graph>\n\n请计算从源节点 {source} 到汇节点 {sink} 的最大流。',
            'en_prompt': 'Given the following graph structure:\n<graph>\n\nCalculate the maximum flow from source node {source} to sink node {sink}.'
        },
        'PAGE_RANK': {
            'zh_prompt': '给定以下图结构:\n<graph>\n\n请计算哪个节点的PageRank值最大。(阻尼系数=0.85，迭代次数=3)',
            'en_prompt': 'Given the following graph structure:\n<graph>\n\nWhich node has the largest PageRank value? (damping factor=0.85, iterations=3)'
        },
        'TOPOLOGICAL_SORT': {
            'zh_prompt': '给定以下有向无环图结构:\n<graph>\n\n请输出一个有效的拓扑排序序列。',
            'en_prompt': 'Given the following directed acyclic graph:\n<graph>\n\nOutput a valid topological sorting sequence.'
        },
        'DIAMETER': {
            'zh_prompt': '给定以下图结构:\n<graph>\n\n请计算该图的直径(最长最短路径)。',
            'en_prompt': 'Given the following graph structure:\n<graph>\n\nCalculate the diameter (longest shortest path) of the graph.'
        },
        'CLUSTERING_COEFFICIENT': {
            'zh_prompt': '给定以下图结构:\n<graph>\n\n请计算节点 {node} 的聚类系数。',
            'en_prompt': 'Given the following graph structure:\n<graph>\n\nCalculate the clustering coefficient of node {node}.'
        },
        'COMMON_NEIGHBOR': {
            'zh_prompt': '给定以下图结构:\n<graph>\n\n请找出节点 {node1} 和节点 {node2} 的共同邻居。',
            'en_prompt': 'Given the following graph structure:\n<graph>\n\nFind the common neighbors of node {node1} and node {node2}.'
        },
        'JACCARD': {
            'zh_prompt': '给定以下图结构:\n<graph>\n\n请计算节点 {node1} 和节点 {node2} 之间的Jaccard相似度。',
            'en_prompt': 'Given the following graph structure:\n<graph>\n\nCalculate the Jaccard similarity between node {node1} and node {node2}.'
        },
        'PREDECESSOR': {
            'zh_prompt': '给定以下有向图结构:\n<graph>\n\n请列出节点 {node} 的所有前驱节点。',
            'en_prompt': 'Given the following directed graph:\n<graph>\n\nList all predecessor nodes of node {node}.'
        },
        'EDGE': {
            'zh_prompt': '给定以下图结构:\n<graph>\n\n请判断节点 {node1} 和节点 {node2} 之间是否存在边。',
            'en_prompt': 'Given the following graph structure:\n<graph>\n\nDetermine whether there is an edge between node {node1} and node {node2}.'
        },
    }

    @classmethod
    def extract_task_type(cls, instruction: str) -> str:
        """从指令中提取任务类型"""
        # 匹配 "graph reasoning task XXX"
        match = re.search(r'graph\s+reasoning\s+task\s+(\w+)', instruction, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return 'UNKNOWN'

    @classmethod
    def extract_task_params(cls, instruction: str, task_type: str) -> Dict[str, Any]:
        """从指令中提取任务参数"""
        params = {}

        if task_type in ['BFS', 'DFS']:
            # "Start from node X"
            match = re.search(r'Start\s+from\s+node\s+(\d+)', instruction, re.IGNORECASE)
            if match:
                params['start_node'] = int(match.group(1))

        elif task_type == 'SHORTEST_PATH':
            # "from node X to node Y"
            match = re.search(r'from\s+node\s+(\d+)\s+to\s+node\s+(\d+)', instruction, re.IGNORECASE)
            if match:
                params['source'] = int(match.group(1))
                params['target'] = int(match.group(2))

        elif task_type == 'DEGREE':
            # "degree of node X" or "What is the degree of node X?"
            match = re.search(r'degree of node\s+(\d+)', instruction, re.IGNORECASE)
            if match:
                params['node'] = int(match.group(1))

        elif task_type == 'NEIGHBOR':
            # "neighbor nodes of node X" or "Which are the neighbor nodes of node X?"
            match = re.search(r'neighbor nodes of node\s+(\d+)', instruction, re.IGNORECASE)
            if match:
                params['node'] = int(match.group(1))

        elif task_type == 'PREDECESSOR':
            # "predecessor nodes of node X" or "Which are the predecessor nodes of node X?"
            match = re.search(r'predecessor nodes of node\s+(\d+)', instruction, re.IGNORECASE)
            if match:
                params['node'] = int(match.group(1))

        elif task_type == 'CLUSTERING_COEFFICIENT':
            # "clustering coefficient of node X" or "What is the clustering coefficient of node X?"
            match = re.search(r'clustering coefficient of node\s+(\d+)', instruction, re.IGNORECASE)
            if match:
                params['node'] = int(match.group(1))

        elif task_type == 'PAGE_RANK':
            # PAGE_RANK asks "Which node has the largest PageRank value?" - no specific node
            # The template doesn't require a node parameter
            pass

        elif task_type == 'CONNECTIVITY':
            # "node X and node Y are connected"
            match = re.search(r'node\s+(\d+)\s+and\s+node\s+(\d+)', instruction, re.IGNORECASE)
            if match:
                params['source'] = int(match.group(1))
                params['target'] = int(match.group(2))

        elif task_type == 'MAXIMUM_FLOW':
            # "from node X to node Y"
            match = re.search(r'from\s+node\s+(\d+)\s+to\s+node\s+(\d+)', instruction, re.IGNORECASE)
            if match:
                params['source'] = int(match.group(1))
                params['sink'] = int(match.group(2))

        elif task_type in ['COMMON_NEIGHBOR', 'JACCARD', 'EDGE']:
            # "node X and node Y"
            match = re.search(r'node\s+(\d+)\s+and\s+node\s+(\d+)', instruction, re.IGNORECASE)
            if match:
                params['node1'] = int(match.group(1))
                params['node2'] = int(match.group(2))

        return params


class GraphInstructConverter:
    """GraphInstruct 到 GraphAgent 格式转换器"""

    def __init__(self, use_chinese: bool = True, keep_original_response: bool = True):
        """
        Args:
            use_chinese: 是否使用中文提示
            keep_original_response: 是否保留原始回答（包含推理过程）
        """
        self.use_chinese = use_chinese
        self.keep_original_response = keep_original_response
        self.parser = GraphTextParser()
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'by_task': {}
        }

    def convert_sample(self, sample: Dict, task_type: str, sample_id: int) -> Optional[Dict]:
        """转换单个样本"""
        try:
            instruction = sample['instruction']
            output = sample['output']

            # 解析图结构
            graph_data = self.parser.parse(instruction)

            if not graph_data.nodes:
                logger.warning(f"Sample {sample_id}: No nodes parsed from instruction")
                return None

            # 构建 edge_index 格式 [[src...], [dst...]]
            edge_index = [[], []]
            for src, dst in graph_data.edges:
                edge_index[0].append(src)
                edge_index[1].append(dst)

            # 提取任务参数
            task_params = TaskConverter.extract_task_params(instruction, task_type)

            # 构建对话
            if self.use_chinese:
                template = TaskConverter.TASK_TEMPLATES.get(task_type, {}).get('zh_prompt', '')
            else:
                template = TaskConverter.TASK_TEMPLATES.get(task_type, {}).get('en_prompt', '')

            if template and task_params:
                try:
                    human_message = template.format(**task_params)
                except KeyError:
                    # 使用原始指令
                    human_message = f"<graph>\n\n{instruction}"
            else:
                # 使用简化版指令
                human_message = f"<graph>\n\n{self._simplify_instruction(instruction)}"

            # 处理回答
            if self.keep_original_response:
                gpt_message = output
            else:
                # 只提取最终答案
                answer_match = re.search(r'<<<(.+?)>>>', output)
                if answer_match:
                    gpt_message = answer_match.group(1)
                else:
                    gpt_message = output

            # 构建 GraphAgent 格式
            result = {
                'id': f'graphinstruct_{task_type.lower()}_{sample_id}',
                'graph': {
                    'node_list': graph_data.nodes,
                    'edge_index': edge_index,
                    'is_directed': graph_data.is_directed,
                },
                'conversations': [
                    {'from': 'human', 'value': human_message},
                    {'from': 'gpt', 'value': gpt_message}
                ],
                'task_type': task_type,
                'task_params': task_params
            }

            # 添加权重信息
            if graph_data.weights:
                result['graph']['edge_weights'] = graph_data.weights

            return result

        except Exception as e:
            logger.error(f"Sample {sample_id}: Conversion failed - {e}")
            return None

    def _simplify_instruction(self, instruction: str) -> str:
        """简化指令文本"""
        # 移除图连接描述部分，只保留问题
        lines = instruction.split('\n')
        question_lines = []
        in_graph_desc = False

        for line in lines:
            if 'Node' in line and 'connected' in line:
                in_graph_desc = True
                continue
            if in_graph_desc and line.strip() == '':
                in_graph_desc = False
                continue
            if not in_graph_desc:
                question_lines.append(line)

        return '\n'.join(question_lines).strip()

    def convert_file(self, input_path: str, output_path: str, task_type: str, max_samples: Optional[int] = None):
        """转换单个文件"""
        logger.info(f"Converting {input_path}...")

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if max_samples:
            data = data[:max_samples]

        converted = []
        for i, sample in enumerate(tqdm(data, desc=f"Converting {task_type}")):
            self.stats['total'] += 1

            result = self.convert_sample(sample, task_type, sample.get('id', i))
            if result:
                converted.append(result)
                self.stats['success'] += 1
                self.stats['by_task'][task_type] = self.stats['by_task'].get(task_type, 0) + 1
            else:
                self.stats['failed'] += 1

        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)

        logger.info(f"Converted {len(converted)}/{len(data)} samples to {output_path}")
        return converted

    def convert_all_tasks(self, input_dir: str, output_dir: str, max_samples_per_task: Optional[int] = None):
        """转换所有任务类型"""
        task_dirs = {
            'BFS-int_id': 'BFS',
            'DFS-int_id': 'DFS',
            'shortest_path-int_id': 'SHORTEST_PATH',
            'degree-int_id': 'DEGREE',
            'neighbor-int_id': 'NEIGHBOR',
            'cycle-int_id': 'CYCLE',
            'connectivity-int_id': 'CONNECTIVITY',
            'bipartite-int_id': 'BIPARTITE',
            'connected_component-int_id': 'CONNECTED_COMPONENT',
            'MST-int_id': 'MST',
            'maximum_flow-int_id': 'MAXIMUM_FLOW',
            'page_rank-int_id': 'PAGE_RANK',
            'topological_sort-int_id': 'TOPOLOGICAL_SORT',
            'diameter-int_id': 'DIAMETER',
            'clustering_coefficient-int_id': 'CLUSTERING_COEFFICIENT',
            'common_neighbor-int_id': 'COMMON_NEIGHBOR',
            'jaccard-int_id': 'JACCARD',
            'predecessor-int_id': 'PREDECESSOR',
            'edge-int_id': 'EDGE',
        }

        all_converted = []

        for dir_name, task_type in task_dirs.items():
            input_path = os.path.join(input_dir, dir_name, 'train.json')
            if not os.path.exists(input_path):
                logger.warning(f"Skipping {task_type}: {input_path} not found")
                continue

            output_path = os.path.join(output_dir, f'{task_type.lower()}_converted.json')
            converted = self.convert_file(input_path, output_path, task_type, max_samples_per_task)
            all_converted.extend(converted)

        # 保存合并后的数据
        merged_path = os.path.join(output_dir, 'merged_graphinstruct.json')
        with open(merged_path, 'w', encoding='utf-8') as f:
            json.dump(all_converted, f, ensure_ascii=False, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"Conversion Statistics:")
        logger.info(f"  Total processed: {self.stats['total']}")
        logger.info(f"  Success: {self.stats['success']}")
        logger.info(f"  Failed: {self.stats['failed']}")
        logger.info(f"\nBy task type:")
        for task, count in sorted(self.stats['by_task'].items()):
            logger.info(f"  {task}: {count}")
        logger.info(f"\nMerged output: {merged_path}")
        logger.info(f"Total samples in merged file: {len(all_converted)}")

        return all_converted


def main():
    parser = argparse.ArgumentParser(description='Convert GraphInstruct data to GraphAgent format')
    parser.add_argument('--input-dir', type=str,
                        default='/nvme0/work/workspaces-zy/GraphInstruct/LLaMAFactory/data/reasoning',
                        help='Input directory containing GraphInstruct task folders')
    parser.add_argument('--output-dir', type=str,
                        default='/nvme0/work/workspaces-zy/GraphInstruct/data/converted',
                        help='Output directory for converted data')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples per task type (for testing)')
    parser.add_argument('--chinese', action='store_true', default=True,
                        help='Use Chinese prompts')
    parser.add_argument('--english', action='store_true',
                        help='Use English prompts')
    parser.add_argument('--keep-reasoning', action='store_true', default=True,
                        help='Keep full reasoning in response')

    args = parser.parse_args()

    use_chinese = not args.english

    converter = GraphInstructConverter(
        use_chinese=use_chinese,
        keep_original_response=args.keep_reasoning
    )

    converter.convert_all_tasks(
        args.input_dir,
        args.output_dir,
        args.max_samples
    )


if __name__ == '__main__':
    main()
