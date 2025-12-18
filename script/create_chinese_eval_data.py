#!/usr/bin/env python3
"""
中文评测数据生成脚本

生成中文版的GraphAgent评测数据集，支持两种格式：
1. .pt 格式 - 用于GraphAgent图模型评测
2. .json 格式 - 用于纯文本LLM基线评测

Usage:
    # 生成图模型评测数据 (.pt)
    python create_chinese_eval_data.py --format pt --samples-per-task 10

    # 生成纯文本评测数据 (.json)
    python create_chinese_eval_data.py --format text --samples-per-task 10

    # 同时生成两种格式
    python create_chinese_eval_data.py --format both --samples-per-task 10
"""

import json
import copy
import os
import sys
import random
import logging
import argparse
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import torch
import transformers
from torch_geometric.data import HeteroData

# 添加路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
GRAPHAGENT_TRAINING_PATH = "/nvme0/work/workspaces-zy/GraphAgent-zy/GraphAGent-training"
sys.path.insert(0, GRAPHAGENT_TRAINING_PATH)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

# 导入多语言模块
from GTG.utils.language import LANG_ZH, LANG_EN

# 导入conversation模块
try:
    from model.graph_action_agent import conversation as conversation_lib
except ImportError:
    conversation_lib = None
    logging.warning("Could not import conversation_lib from GraphAgent training")

# 导入图编码器
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

# 19个任务及其多语言生成器
ALL_TASKS = [
    'BFS', 'DFS', 'bipartite', 'clustering_coefficient', 'common_neighbor',
    'connected_component', 'connectivity', 'cycle', 'degree', 'diameter',
    'edge', 'jaccard', 'maximum_flow', 'MST', 'neighbor',
    'page_rank', 'predecessor', 'shortest_path', 'topological_sort'
]


def get_multilang_generator(task_name: str):
    """获取任务的多语言生成器"""
    try:
        module = __import__(f'GTG.tasks.{task_name}.{task_name}_multilang', fromlist=['generate_a_sample_multilang'])
        return module.generate_a_sample_multilang
    except ImportError as e:
        logger.warning(f"Could not import multilang generator for {task_name}: {e}")
        return None


class ChineseEvalDataGenerator:
    """中文评测数据生成器"""

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
            'empty_labels': 0,
            'generation_failed': 0
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

    def _parse_graph_adj_str(self, graph_adj_str: str) -> Dict[int, List[int]]:
        """解析邻接表字符串为字典"""
        import re
        adj_dict = {}

        pattern = r'<(\d+)>:\s*\[([^\]]*)\]'
        matches = re.findall(pattern, graph_adj_str)

        for node_str, neighbors_str in matches:
            node_id = int(node_str)
            neighbor_pattern = r'<(\d+)>'
            neighbors = [int(n) for n in re.findall(neighbor_pattern, neighbors_str)]
            adj_dict[node_id] = neighbors

        return adj_dict

    def _adj_to_edge_index(self, adj_dict: Dict[int, List[int]], directed: bool = False) -> List[List[int]]:
        """将邻接表转换为edge_index格式"""
        src_nodes = []
        dst_nodes = []
        seen_edges = set()

        for src, neighbors in adj_dict.items():
            for dst in neighbors:
                if directed:
                    src_nodes.append(src)
                    dst_nodes.append(dst)
                else:
                    edge = tuple(sorted([src, dst]))
                    if edge not in seen_edges:
                        seen_edges.add(edge)
                        src_nodes.append(src)
                        dst_nodes.append(dst)

        return [src_nodes, dst_nodes]

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
                conv_text = "<|im_start|>system\n你是GraphAgent，一个图推理任务的助手。<|im_end|>\n"
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

    def _create_hetero_data(self, node_list: List[int], edge_index: List[List[int]]) -> HeteroData:
        """创建HeteroData图数据"""
        num_nodes = len(node_list) if node_list else 1

        hetero_data = HeteroData()
        hetero_data["node"].x = torch.randn(num_nodes, self.node_feature_dim)
        hetero_data["node"].description = [f"节点 {i}" for i in range(num_nodes)]

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
            encoded_graph = encoded_graph.cpu()
            return encoded_graph
        except Exception as e:
            logger.warning(f"Graph encoding failed: {e}")
            return hetero_data

    def generate_sample(self, task_name: str, config: Dict, sample_idx: int) -> Optional[Dict]:
        """生成单个中文样本"""
        generator = get_multilang_generator(task_name)
        if generator is None:
            return None

        try:
            # 生成中文样本
            sample = generator(config, lang=LANG_ZH)
            if sample is None:
                return None

            return sample
        except Exception as e:
            logger.debug(f"Failed to generate sample for {task_name}: {e}")
            return None

    def convert_to_pt_format(self, sample: Dict, task_name: str, sample_idx: int) -> Optional[Dict]:
        """将样本转换为.pt格式（用于GraphAgent评测）"""
        try:
            # 解析图结构
            graph_adj_str = sample.get('graph_adj', '')
            adj_dict = self._parse_graph_adj_str(graph_adj_str)

            if not adj_dict:
                self.stats['no_graph'] += 1
                return None

            # 获取节点列表和边
            all_nodes = set(adj_dict.keys())
            for neighbors in adj_dict.values():
                all_nodes.update(neighbors)
            node_list = sorted(list(all_nodes))

            directed = sample.get('directed', False)
            edge_index = self._adj_to_edge_index(adj_dict, directed)

            num_graph_tokens = len(node_list)

            # 构建对话
            graph_nl = sample.get('graph_nl', '')
            question = sample.get('question', '')
            steps = sample.get('steps', '')
            answer = sample.get('answer', '')

            human_message = f"<graph>\n{graph_nl}\n\n{question}"
            gpt_message = f"{steps}<<<{answer}>>>"

            conversations = [
                {'from': 'human', 'value': human_message},
                {'from': 'gpt', 'value': gpt_message}
            ]

            # 预处理对话
            processed_conversations = []
            for conv in conversations:
                processed_conv = copy.deepcopy(conv)
                processed_conv['value'] = self._preprocess_graph_token(
                    processed_conv['value'], num_graph_tokens
                )
                processed_conversations.append(processed_conv)

            # 应用对话模板并tokenize
            prompt_conversations, conv = self._apply_prompt_template([processed_conversations])
            input_ids, labels = self._tokenize_and_mask_qwen(prompt_conversations, conv)

            # 检查序列长度
            if input_ids.shape[1] >= self.max_length:
                self.stats['too_long'] += 1
                return None

            # 检查labels是否全为-100
            if torch.all(labels == IGNORE_TOKEN_ID):
                self.stats['empty_labels'] += 1
                return None

            # 创建图数据
            graph_data = self._create_hetero_data(node_list, edge_index)

            # 编码图数据
            if self.use_graph_encoding:
                graph_data = self._encode_graph(graph_data)

            sample_id = f"{task_name}_zh_{sample_idx}"

            result = {
                'id': sample_id,
                'input_ids': input_ids[0],
                'labels': labels[0],
                'graph_data': graph_data,
                'hetero_key_order': ['node'],
                'task_type': task_name,
                'ground_truth': answer,
                'language': 'zh'
            }

            return result

        except Exception as e:
            logger.debug(f"Error converting sample to pt format: {e}")
            return None

    def convert_to_text_format(self, sample: Dict, task_name: str, sample_idx: int) -> Optional[Dict]:
        """将样本转换为纯文本JSON格式（用于LLM基线评测）"""
        try:
            graph_nl = sample.get('graph_nl', '')
            question = sample.get('question', '')
            steps = sample.get('steps', '')
            answer = sample.get('answer', '')

            sample_id = f"{task_name}_zh_{sample_idx}"

            # 构建instruction (图描述 + 问题)
            instruction = f"{graph_nl}\n\n{question}"

            # 构建output (推理步骤 + 答案)
            output = f"{steps}<<<{answer}>>>"

            result = {
                'id': sample_id,
                'task_type': task_name,
                'instruction': instruction,
                'output': output,
                'ground_truth': answer,
                'language': 'zh'
            }

            return result

        except Exception as e:
            logger.debug(f"Error converting sample to text format: {e}")
            return None

    def generate_dataset(
        self,
        output_dir: str,
        samples_per_task: int = 10,
        output_format: str = 'both',
        seed: int = 42
    ):
        """生成完整的中文评测数据集"""
        random.seed(seed)

        config = {'num_nodes_range': '[8, 15]'}

        pt_samples = []
        text_samples = []

        for task_name in ALL_TASKS:
            logger.info(f"Processing {task_name}...")

            task_pt_samples = []
            task_text_samples = []
            attempts = 0
            max_attempts = samples_per_task * 3  # 最多尝试3倍的次数

            while len(task_pt_samples) < samples_per_task and attempts < max_attempts:
                attempts += 1
                self.stats['total'] += 1

                # 生成样本
                sample = self.generate_sample(task_name, config, len(task_pt_samples))
                if sample is None:
                    self.stats['generation_failed'] += 1
                    continue

                # 转换为PT格式
                if output_format in ['pt', 'both']:
                    pt_result = self.convert_to_pt_format(sample, task_name, len(task_pt_samples))
                    if pt_result is not None:
                        task_pt_samples.append(pt_result)
                        self.stats['success'] += 1
                    else:
                        self.stats['failed'] += 1
                        continue

                # 转换为文本格式
                if output_format in ['text', 'both']:
                    text_result = self.convert_to_text_format(sample, task_name, len(task_text_samples))
                    if text_result is not None:
                        task_text_samples.append(text_result)

            pt_samples.extend(task_pt_samples)
            text_samples.extend(task_text_samples)

            logger.info(f"  {task_name}: {len(task_pt_samples)}/{samples_per_task} samples generated")

        # 保存数据
        os.makedirs(output_dir, exist_ok=True)

        if output_format in ['pt', 'both'] and pt_samples:
            pt_output_path = os.path.join(output_dir, f'graphinstruct_zh_{len(ALL_TASKS)}tasks_{samples_per_task}samples.pt')
            logger.info(f"Saving {len(pt_samples)} PT samples to {pt_output_path}...")
            torch.save(pt_samples, pt_output_path)

        if output_format in ['text', 'both'] and text_samples:
            text_output_path = os.path.join(output_dir, f'graphinstruct_zh_{len(ALL_TASKS)}tasks_{samples_per_task}samples_text.json')
            logger.info(f"Saving {len(text_samples)} text samples to {text_output_path}...")
            # 转换numpy类型为原生Python类型
            def convert_to_native(obj):
                import numpy as np
                if isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            text_samples = convert_to_native(text_samples)
            with open(text_output_path, 'w', encoding='utf-8') as f:
                json.dump(text_samples, f, indent=2, ensure_ascii=False)

        self._print_stats()

        return pt_samples, text_samples

    def _print_stats(self):
        """打印统计信息"""
        logger.info(f"\n{'='*60}")
        logger.info("中文评测数据生成统计:")
        logger.info(f"  总尝试次数: {self.stats['total']}")
        logger.info(f"  成功: {self.stats['success']}")
        logger.info(f"  失败: {self.stats['failed']}")
        logger.info(f"    - 生成失败: {self.stats['generation_failed']}")
        logger.info(f"    - 序列过长: {self.stats['too_long']}")
        logger.info(f"    - 无图结构: {self.stats['no_graph']}")
        logger.info(f"    - 空标签: {self.stats['empty_labels']}")
        logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='生成中文评测数据集')
    parser.add_argument('--output-dir', type=str,
                        default='/nvme0/work/workspaces-zy/GraphInstruct/data/eval',
                        help='输出目录')
    parser.add_argument('--model-path', type=str,
                        default='/nvme0/work/workspaces-zy/model/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507',
                        help='Qwen模型路径')
    parser.add_argument('--samples-per-task', type=int, default=10,
                        help='每个任务的样本数')
    parser.add_argument('--format', type=str, choices=['pt', 'text', 'both'], default='both',
                        help='输出格式: pt(图模型), text(纯文本), both(两者)')
    parser.add_argument('--max-length', type=int, default=4096,
                        help='最大序列长度')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--no-graph-encoding', action='store_true',
                        help='禁用图编码(使用随机特征)')

    args = parser.parse_args()

    generator = ChineseEvalDataGenerator(
        model_path=args.model_path,
        max_length=args.max_length,
        use_graph_encoding=not args.no_graph_encoding
    )

    generator.generate_dataset(
        output_dir=args.output_dir,
        samples_per_task=args.samples_per_task,
        output_format=args.format,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
