#!/usr/bin/env python3
"""
GraphInstruct 小样本训练数据准备脚本 - 采样版本
从每个任务中采样指定数量的样本，生成小规模训练数据

用法:
    python prepare_sampled_training_data.py \
        --input-dir /nvme0/work/workspaces-zy/GraphInstruct/data/converted_fixed \
        --output-dir /mnt/yrfs/GraphAgent_model/zy/data \
        --samples-per-task 1000
"""

import json
import copy
import os
import sys
import logging
import random
from typing import Dict, List, Optional
from tqdm import tqdm
import argparse

import torch
import transformers
from torch_geometric.data import HeteroData

# 添加 GraphAgent 训练代码路径
GRAPHAGENT_TRAINING_PATH = "/nvme0/work/workspaces-zy/GraphAgent-zy/GraphAGent-training"
sys.path.insert(0, GRAPHAGENT_TRAINING_PATH)

try:
    from model.graph_action_agent import conversation as conversation_lib
except ImportError:
    conversation_lib = None
    logging.warning("Could not import conversation_lib from GraphAgent training")

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


class SampledDataPreparer:
    """采样训练数据准备器 - 从每个任务采样指定数量"""

    def __init__(
        self,
        model_path: str,
        max_length: int = 4096,
        node_feature_dim: int = 768,
        use_graph_start_end: bool = True,
        seed: int = 42
    ):
        self.model_path = model_path
        self.max_length = max_length
        self.node_feature_dim = node_feature_dim
        self.use_graph_start_end = use_graph_start_end
        self.seed = seed

        random.seed(seed)

        # 初始化 tokenizer
        self.tokenizer = self._setup_tokenizer()

        # 统计信息
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'too_long': 0,
            'no_graph': 0,
            'empty_labels': 0,
            'by_task': {}
        }

    def _setup_tokenizer(self) -> transformers.PreTrainedTokenizer:
        """设置 Qwen tokenizer"""
        logger.info(f"Loading Qwen tokenizer from {self.model_path}...")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=self.max_length,
            padding_side="right",
        )

        eos_token = "<|im_end|>"
        eos_id = tokenizer.convert_tokens_to_ids(eos_token)
        if eos_id is None or eos_id == tokenizer.unk_token_id:
            eos_token = "<|endoftext|>"
            eos_id = tokenizer.convert_tokens_to_ids(eos_token)

        tokenizer.pad_token = eos_token
        tokenizer.pad_token_id = eos_id
        logger.info(f"Using pad token: {eos_token} (id={eos_id})")

        if conversation_lib is not None:
            conversation_lib.default_conversation = conversation_lib.conv_templates["qwen"]

        tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
        tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)

        logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

        return tokenizer

    def _preprocess_graph_token(self, text: str, num_graph_tokens: int) -> str:
        """将 <graph> 替换为具体的 patch token 序列"""
        if DEFAULT_GRAPH_TOKEN not in text:
            return text

        replace_token = DEFAULT_GRAPH_PATCH_TOKEN * num_graph_tokens
        if self.use_graph_start_end:
            replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN

        return text.replace(DEFAULT_GRAPH_TOKEN, replace_token)

    def _apply_prompt_template(self, sources: List[List[Dict]]):
        """应用 Qwen 对话模板"""
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
        """Tokenize 并 mask instruction 部分"""
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
        """从 graph_info 创建 HeteroData 图数据"""
        node_list = graph_info.get('node_list', [])
        edge_index = graph_info.get('edge_index', [[], []])

        num_nodes = len(node_list) if node_list else 1

        hetero_data = HeteroData()
        hetero_data["node"].x = torch.randn(num_nodes, self.node_feature_dim)

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

    def process_sample(self, sample: Dict) -> Optional[Dict]:
        """处理单个样本"""
        sample_id = sample.get('id', 'unknown')
        graph_info = sample.get('graph', {})
        conversations = sample.get('conversations', [])

        total_graph_tokens = sum(
            conv['value'].count(DEFAULT_GRAPH_TOKEN)
            for conv in conversations
        )

        if total_graph_tokens == 0:
            self.stats['no_graph'] += 1
            return None

        if total_graph_tokens > 1:
            self.stats['failed'] += 1
            return None

        node_list = graph_info.get('node_list', [])
        num_graph_tokens = len(node_list) if node_list else 1

        processed_conversations = []
        for conv in conversations:
            processed_conv = copy.deepcopy(conv)
            processed_conv['value'] = self._preprocess_graph_token(
                processed_conv['value'], num_graph_tokens
            )
            processed_conversations.append(processed_conv)

        try:
            prompt_conversations, conv = self._apply_prompt_template([processed_conversations])
            input_ids, labels = self._tokenize_and_mask_qwen(prompt_conversations, conv)
        except Exception as e:
            logger.debug(f"Failed to process sample {sample_id}: {e}")
            return None

        if input_ids.shape[1] >= self.max_length:
            self.stats['too_long'] += 1
            return None

        if torch.all(labels == IGNORE_TOKEN_ID):
            self.stats['empty_labels'] += 1
            return None

        graph_data = self._create_hetero_data(graph_info)

        result = {
            'id': sample_id,
            'input_ids': input_ids[0],
            'labels': labels[0],
            'graph_data': graph_data,
            'hetero_key_order': ['node'],
            'task_type': sample.get('task_type', 'unknown'),
        }

        return result

    def process_sampled(self, input_dir: str, output_path: str, samples_per_task: int = 1000):
        """从每个任务采样并处理"""
        # 查找所有 *_converted.json 文件
        task_files = []
        for f in sorted(os.listdir(input_dir)):
            if f.endswith('_converted.json') and not f.startswith('merged'):
                task_files.append(f)

        logger.info(f"Found {len(task_files)} task files")

        all_processed = []

        for filename in task_files:
            task_name = filename.replace('_converted.json', '')
            input_path = os.path.join(input_dir, filename)

            logger.info(f"\n{'='*60}")
            logger.info(f"Processing task: {task_name}")

            # 加载数据
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            total_available = len(data)
            logger.info(f"Total samples available: {total_available}")

            # 随机采样
            if total_available <= samples_per_task:
                sampled_data = data
                logger.info(f"Using all {total_available} samples (less than {samples_per_task})")
            else:
                sampled_data = random.sample(data, samples_per_task)
                logger.info(f"Sampled {samples_per_task} from {total_available} samples")

            # 处理采样数据
            task_processed = []
            for sample in tqdm(sampled_data, desc=f"Processing {task_name}"):
                self.stats['total'] += 1
                result = self.process_sample(sample)
                if result is not None:
                    task_processed.append(result)
                    self.stats['success'] += 1
                else:
                    self.stats['failed'] += 1

            self.stats['by_task'][task_name] = len(task_processed)
            all_processed.extend(task_processed)
            logger.info(f"Task {task_name}: {len(task_processed)} samples processed successfully")

        # 打乱顺序
        random.shuffle(all_processed)

        # 保存
        logger.info(f"\n{'='*60}")
        logger.info(f"Saving {len(all_processed)} total samples to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(all_processed, output_path)

        self._print_stats()

        return all_processed

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
        logger.info(f"\nSamples by task:")
        for task, count in sorted(self.stats['by_task'].items()):
            logger.info(f"  {task}: {count}")
        logger.info(f"\nTotal tasks: {len(self.stats['by_task'])}")


def main():
    parser = argparse.ArgumentParser(description='Prepare sampled GraphInstruct training data')
    parser.add_argument('--input-dir', type=str,
                        default='/nvme0/work/workspaces-zy/GraphInstruct/data/converted_fixed',
                        help='Directory containing converted JSON files')
    parser.add_argument('--output-dir', type=str,
                        default='/mnt/yrfs/GraphAgent_model/zy/data',
                        help='Output directory for training pt files')
    parser.add_argument('--model-path', type=str,
                        default='/nvme0/work/workspaces-zy/model/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507',
                        help='Path to Qwen model for tokenizer')
    parser.add_argument('--max-length', type=int, default=4096,
                        help='Maximum sequence length')
    parser.add_argument('--samples-per-task', type=int, default=1000,
                        help='Number of samples per task')
    parser.add_argument('--node-dim', type=int, default=768,
                        help='Node feature dimension')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output-name', type=str, default='graphinstruct_train_qwen_sampled_19k.pt',
                        help='Output filename')

    args = parser.parse_args()

    # 创建数据准备器
    preparer = SampledDataPreparer(
        model_path=args.model_path,
        max_length=args.max_length,
        node_feature_dim=args.node_dim,
        seed=args.seed
    )

    # 处理采样数据
    output_path = os.path.join(args.output_dir, args.output_name)
    preparer.process_sampled(
        args.input_dir,
        output_path,
        args.samples_per_task
    )

    logger.info(f"\n完成! 输出文件: {output_path}")


if __name__ == '__main__':
    main()
