#!/usr/bin/env python3
"""
GraphInstruct 评测数据准备脚本
从GraphInstruct测试数据中采样19个任务各10条数据,转换为GraphAgent评测格式
"""

import json
import copy
import os
import sys
import random
import logging
from typing import Dict, List, Optional
from tqdm import tqdm
import argparse

import torch
import transformers
from torch_geometric.data import HeteroData

# 添加 GraphAgent 训练代码路径
GRAPHAGENT_TRAINING_PATH = "/nvme0/work/workspaces-zy/GraphAgent-zy/GraphAGent-training"
sys.path.insert(0, GRAPHAGENT_TRAINING_PATH)

# 添加当前目录用于导入sample_graph_tokenizer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model.graph_action_agent import conversation as conversation_lib
except ImportError:
    conversation_lib = None
    logging.warning("Could not import conversation_lib from GraphAgent training")

try:
    from sample_graph_tokenizer import SampleGraphTokenizer
except ImportError:
    SampleGraphTokenizer = None
    logging.warning("Could not import SampleGraphTokenizer")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 常量定义
IGNORE_TOKEN_ID = -100
DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"

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


class GraphTextParser:
    """从文本描述中解析图结构"""

    import re

    PATTERN_DIRECTED = re.compile(r'directed\s+graph', re.IGNORECASE)

    @classmethod
    def parse(cls, text: str) -> Dict:
        """从文本中解析图结构"""
        import re
        is_directed = bool(cls.PATTERN_DIRECTED.search(text))

        nodes = set()
        edges = []
        weights = []
        has_weights = False

        # 权重提取模式
        PATTERN_WEIGHT = re.compile(r'(\d+)\s*\(weight:\s*([\d.]+)\)')

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
            weight_matches = PATTERN_WEIGHT.findall(connection_part)

            if weight_matches:
                has_weights = True
                for target_str, weight_str in weight_matches:
                    target = int(target_str)
                    weight = float(weight_str)
                    nodes.add(target)
                    edges.append((source, target))
                    weights.append(weight)
                    if not is_directed:
                        edges.append((target, source))
                        weights.append(weight)
            else:
                targets = re.findall(r'\d+', connection_part)
                for target_str in targets:
                    target = int(target_str)
                    nodes.add(target)
                    edges.append((source, target))
                    if not is_directed:
                        edges.append((target, source))

        # 去重边
        seen_edges = set()
        unique_edges = []
        unique_weights = []

        for i, edge in enumerate(edges):
            if edge not in seen_edges:
                seen_edges.add(edge)
                unique_edges.append(edge)
                if has_weights and i < len(weights):
                    unique_weights.append(weights[i])

        # 构建 edge_index 格式
        edge_index = [[], []]
        for src, dst in unique_edges:
            edge_index[0].append(src)
            edge_index[1].append(dst)

        return {
            'node_list': sorted(nodes),
            'edge_index': edge_index,
            'is_directed': is_directed,
            'edge_weights': unique_weights if has_weights else None
        }


class EvaluationDataPreparer:
    """评测数据准备器"""

    def __init__(
        self,
        model_path: str,
        max_length: int = 4096,
        node_feature_dim: int = 768,
        use_graph_start_end: bool = True,
        use_graph_encoding: bool = True
    ):
        self.model_path = model_path
        self.max_length = max_length
        self.node_feature_dim = node_feature_dim
        self.use_graph_start_end = use_graph_start_end
        self.use_graph_encoding = use_graph_encoding

        # 初始化tokenizer
        self.tokenizer = self._setup_tokenizer()

        # 初始化图编码器
        self.graph_tokenizer = None
        if use_graph_encoding and SampleGraphTokenizer is not None:
            try:
                self.graph_tokenizer = SampleGraphTokenizer(
                    device='cuda:0' if torch.cuda.is_available() else 'cpu',
                    sentence_transformer_path='/nvme0/work/workspaces-zy/model/GraphAgent/all-mpnet-base-v2',
                    pretrained_gnn_path='/nvme0/work/workspaces-zy/model/GraphAgent/GraphTokenizer'
                )
                logger.info("Graph tokenizer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize graph tokenizer: {e}")
                self.graph_tokenizer = None

        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'too_long': 0,
            'no_graph': 0,
            'empty_labels': 0
        }

    def _setup_tokenizer(self) -> transformers.PreTrainedTokenizer:
        """设置Qwen tokenizer"""
        logger.info(f"Loading Qwen tokenizer from {self.model_path}...")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=self.max_length,
            padding_side="right",
        )

        # Qwen使用<|im_end|>或<|endoftext|>作为EOS和pad token
        eos_token = "<|im_end|>"
        eos_id = tokenizer.convert_tokens_to_ids(eos_token)
        if eos_id is None or eos_id == tokenizer.unk_token_id:
            eos_token = "<|endoftext|>"
            eos_id = tokenizer.convert_tokens_to_ids(eos_token)

        tokenizer.pad_token = eos_token
        tokenizer.pad_token_id = eos_id
        logger.info(f"Using pad token: {eos_token} (id={eos_id})")

        # 设置Qwen对话模板
        if conversation_lib is not None:
            conversation_lib.default_conversation = conversation_lib.conv_templates["qwen"]

        # 添加特殊token
        tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
        tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)

        logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
        return tokenizer

    def _preprocess_graph_token(self, text: str, num_graph_tokens: int) -> str:
        """将<graph>替换为具体的patch token序列"""
        if DEFAULT_GRAPH_TOKEN not in text:
            return text

        replace_token = DEFAULT_GRAPH_PATCH_TOKEN * num_graph_tokens
        if self.use_graph_start_end:
            replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN

        return text.replace(DEFAULT_GRAPH_TOKEN, replace_token)

    def _apply_prompt_template(self, sources: List[List[Dict]]):
        """应用Qwen对话模板"""
        if conversation_lib is None:
            conversations = []
            for source in sources:
                conv_text = "<|im_start|>system\nYou are GraphAgent, a helpful assistant for graph reasoning tasks.<|im_end|>\n"
                for msg in source:
                    role = "user" if msg["from"] == "human" else "assistant"
                    conv_text += f"<|im_start|>{role}\n{msg['value']}<|im_end|>\n"
                conversations.append(conv_text)
            return conversations, None

        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        conversations = []
        for source in sources:
            if roles[source[0]["from"]] != conv.roles[0]:
                source = source[1:]

            conv.messages = []
            for sentence in source:
                role = roles[sentence["from"]]
                conv.append_message(role, sentence["value"])

            prompt = conv.get_prompt()
            conversations.append(prompt)

        return conversations, conv

    def _tokenize_and_mask_qwen(self, conversations: List[str], conv):
        """Tokenize并mask instruction部分"""
        input_ids = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False
        ).input_ids

        targets = input_ids.clone()

        im_start = "<|im_start|>"
        im_end = "<|im_end|>"

        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(self.tokenizer.pad_token_id).sum())
            cur_len = 0

            parts = conversation.split(im_start)

            for i, part in enumerate(parts):
                if not part:
                    continue

                if i > 0:
                    part = im_start + part

                part_ids = self.tokenizer(part, add_special_tokens=False).input_ids
                part_len = len(part_ids)

                if part.startswith(f"{im_start}assistant"):
                    header = f"{im_start}assistant\n"
                    header_len = len(self.tokenizer(header, add_special_tokens=False).input_ids)
                    target[cur_len:cur_len + header_len] = IGNORE_TOKEN_ID
                else:
                    target[cur_len:cur_len + part_len] = IGNORE_TOKEN_ID

                cur_len += part_len

            target[cur_len:] = IGNORE_TOKEN_ID

        return input_ids, targets

    def _create_hetero_data(self, graph_info: Dict) -> HeteroData:
        """从graph_info创建HeteroData图数据"""
        node_list = graph_info.get('node_list', [])
        edge_index = graph_info.get('edge_index', [[], []])

        num_nodes = len(node_list) if node_list else 1

        hetero_data = HeteroData()
        hetero_data["node"].x = torch.randn(num_nodes, self.node_feature_dim)
        hetero_data["node"].description = [f"node {i}" for i in range(num_nodes)]

        if edge_index and len(edge_index) == 2 and len(edge_index[0]) > 0:
            node_to_idx = {node: idx for idx, node in enumerate(node_list)}

            mapped_src = []
            mapped_dst = []
            for src, dst in zip(edge_index[0], edge_index[1]):
                if src in node_to_idx and dst in node_to_idx:
                    mapped_src.append(node_to_idx[src])
                    mapped_dst.append(node_to_idx[dst])

            if mapped_src:
                edge_index_tensor = torch.tensor([mapped_src, mapped_dst], dtype=torch.long)
                hetero_data["node", "to", "node"].edge_index = edge_index_tensor

        return hetero_data

    def _encode_graph(self, hetero_data: HeteroData) -> HeteroData:
        """使用MetaHGT编码图数据"""
        if self.graph_tokenizer is None:
            return hetero_data

        try:
            graph_copy = hetero_data.clone()
            encoded_graph = self.graph_tokenizer.tokenize(graph_copy, use_graphgpt=False)
            # 确保图数据在CPU上,避免DataLoader pin_memory错误
            encoded_graph = encoded_graph.cpu()
            return encoded_graph
        except Exception as e:
            logger.warning(f"Graph encoding failed: {e}")
            return hetero_data

    def convert_sample(self, sample: Dict, task_type: str) -> Optional[Dict]:
        """转换单个样本为GraphAgent格式"""
        sample_id = sample.get('id', 'unknown')
        instruction = sample.get('instruction', '')
        output = sample.get('output', '')

        # 解析图结构
        graph_info = GraphTextParser.parse(instruction)

        if not graph_info['node_list']:
            self.stats['no_graph'] += 1
            return None

        num_graph_tokens = len(graph_info['node_list'])

        # 构建对话
        human_message = f"<graph>\n\n{instruction}"
        gpt_message = output

        conversations = [
            {'from': 'human', 'value': human_message},
            {'from': 'gpt', 'value': gpt_message}
        ]

        # 预处理对话,替换<graph> token
        processed_conversations = []
        for conv in conversations:
            processed_conv = copy.deepcopy(conv)
            processed_conv['value'] = self._preprocess_graph_token(
                processed_conv['value'], num_graph_tokens
            )
            processed_conversations.append(processed_conv)

        # 应用对话模板并tokenize
        try:
            prompt_conversations, conv = self._apply_prompt_template([processed_conversations])
            input_ids, labels = self._tokenize_and_mask_qwen(prompt_conversations, conv)
        except Exception as e:
            logger.debug(f"Failed to process sample {sample_id}: {e}")
            return None

        # 检查序列长度
        if input_ids.shape[1] >= self.max_length:
            self.stats['too_long'] += 1
            return None

        # 检查labels是否全为-100
        if torch.all(labels == IGNORE_TOKEN_ID):
            self.stats['empty_labels'] += 1
            return None

        # 创建图数据
        graph_data = self._create_hetero_data(graph_info)

        # 编码图数据
        if self.use_graph_encoding:
            graph_data = self._encode_graph(graph_data)

        result = {
            'id': sample_id,
            'input_ids': input_ids[0],
            'labels': labels[0],
            'graph_data': graph_data,
            'hetero_key_order': ['node'],
            'task_type': task_type,
        }

        return result

    def sample_and_convert(
        self,
        test_data_dir: str,
        output_path: str,
        samples_per_task: int = 10,
        seed: int = 42
    ):
        """从测试数据中采样并转换"""
        random.seed(seed)

        all_samples = []

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
            task_samples = []
            for sample in tqdm(sampled_data, desc=f"Converting {task_type}"):
                self.stats['total'] += 1
                result = self.convert_sample(sample, task_type)
                if result is not None:
                    task_samples.append(result)
                    self.stats['success'] += 1
                else:
                    self.stats['failed'] += 1

            all_samples.extend(task_samples)
            logger.info(f"  {task_type}: {len(task_samples)}/{len(sampled_data)} samples converted")

        # 保存
        logger.info(f"Saving {len(all_samples)} samples to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(all_samples, output_path)

        self._print_stats()

        return all_samples

    def _print_stats(self):
        """打印统计信息"""
        logger.info(f"\n{'='*60}")
        logger.info("Processing Statistics:")
        logger.info(f"  Total processed: {self.stats['total']}")
        logger.info(f"  Success: {self.stats['success']}")
        logger.info(f"  Failed: {self.stats['failed']}")
        logger.info(f"    - Too long: {self.stats['too_long']}")
        logger.info(f"    - No graph token: {self.stats['no_graph']}")
        logger.info(f"    - Empty labels: {self.stats['empty_labels']}")
        logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Prepare GraphInstruct evaluation data')
    parser.add_argument('--test-data-dir', type=str,
                        default='/nvme0/work/workspaces-zy/GraphInstruct/data/test',
                        help='Directory containing test JSON files')
    parser.add_argument('--output-path', type=str,
                        default='/nvme0/work/workspaces-zy/GraphInstruct/data/eval/graphinstruct_eval_19tasks_10samples.pt',
                        help='Output path for evaluation pt file')
    parser.add_argument('--model-path', type=str,
                        default='/nvme0/work/workspaces-zy/model/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507',
                        help='Path to Qwen model for tokenizer')
    parser.add_argument('--samples-per-task', type=int, default=10,
                        help='Number of samples per task type')
    parser.add_argument('--max-length', type=int, default=4096,
                        help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    parser.add_argument('--no-graph-encoding', action='store_true',
                        help='Disable graph encoding (use random features)')

    args = parser.parse_args()

    preparer = EvaluationDataPreparer(
        model_path=args.model_path,
        max_length=args.max_length,
        use_graph_encoding=not args.no_graph_encoding
    )

    preparer.sample_and_convert(
        test_data_dir=args.test_data_dir,
        output_path=args.output_path,
        samples_per_task=args.samples_per_task,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
