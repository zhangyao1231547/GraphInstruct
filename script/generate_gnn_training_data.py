#!/usr/bin/env python3
"""
GraphAgent GNN Training Data Generator

This script generates training data for GNN-enabled GraphAgent model:
- Stage 1: Graph-Text Alignment (2万条)
- Stage 2: Graph Task Instruction Tuning (19种任务 × 1000条 = 19000条)

Both stages include edge_index for GNN encoder training.
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
from pathlib import Path

import torch
import numpy as np
import transformers
from torch_geometric.data import HeteroData

# Add GraphAgent training path
GRAPHAGENT_TRAINING_PATH = "/nvme0/work/workspaces-zy/GraphAgent-zy/GraphAGent-training"
sys.path.insert(0, GRAPHAGENT_TRAINING_PATH)

from sample_graph_tokenizer import SampleGraphTokenizer

try:
    from model.graph_action_agent import conversation as conversation_lib
except ImportError:
    conversation_lib = None
    logging.warning("Could not import conversation_lib from GraphAgent training")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
IGNORE_TOKEN_ID = -100
DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"

# 19 task types
TASK_TYPES = [
    'bfs', 'bipartite', 'clustering_coefficient', 'common_neighbor',
    'connected_component', 'connectivity', 'cycle', 'degree', 'dfs',
    'diameter', 'edge', 'jaccard', 'maximum_flow', 'mst', 'neighbor',
    'page_rank', 'predecessor', 'shortest_path', 'topological_sort'
]

# Stage 1 alignment task templates (graph-text alignment)
STAGE1_TEMPLATES = [
    {
        "instruction": "请描述这个图的基本结构。\n<graph>",
        "response": "这是一个包含{num_nodes}个节点和{num_edges}条边的{graph_type}。"
    },
    {
        "instruction": "分析以下图结构。\n<graph>",
        "response": "该图有{num_nodes}个节点,{num_edges}条边,平均度为{avg_degree:.2f}。"
    },
    {
        "instruction": "请识别图中的节点。\n<graph>",
        "response": "图中包含节点: {node_list}。"
    },
    {
        "instruction": "What is the structure of this graph?\n<graph>",
        "response": "This is a {graph_type} with {num_nodes} nodes and {num_edges} edges."
    },
    {
        "instruction": "Describe the graph connectivity.\n<graph>",
        "response": "The graph has {num_nodes} nodes connected by {num_edges} edges, with average degree {avg_degree:.2f}."
    },
]


class GNNTrainingDataGenerator:
    """Generator for GNN-enabled GraphAgent training data."""

    def __init__(
        self,
        model_path: str,
        max_length: int = 8192,
        node_feature_dim: int = 768,
        use_graph_start_end: bool = True,
        use_graph_tokenizer: bool = True,
    ):
        self.model_path = model_path
        self.max_length = max_length
        self.node_feature_dim = node_feature_dim
        self.use_graph_start_end = use_graph_start_end

        # Initialize tokenizer
        self.tokenizer = self._setup_tokenizer()

        # Initialize graph tokenizer
        self.graph_tokenizer = None
        if use_graph_tokenizer:
            try:
                self.graph_tokenizer = SampleGraphTokenizer(
                    device='cuda:0' if torch.cuda.is_available() else 'cpu',
                    sentence_transformer_path='/nvme0/work/workspaces-zy/model/GraphAgent/all-mpnet-base-v2',
                    pretrained_gnn_path='/nvme0/work/workspaces-zy/model/GraphAgent/GraphTokenizer'
                )
                logger.info("Graph tokenizer initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize graph tokenizer: {e}")

        # Statistics
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'too_long': 0,
        }

    def _setup_tokenizer(self) -> transformers.PreTrainedTokenizer:
        """Setup Qwen tokenizer."""
        logger.info(f"Loading Qwen tokenizer from {self.model_path}...")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=self.max_length,
            padding_side="right",
        )

        # Qwen uses <|im_end|> as EOS
        eos_token = "<|im_end|>"
        eos_id = tokenizer.convert_tokens_to_ids(eos_token)
        if eos_id is None or eos_id == tokenizer.unk_token_id:
            eos_token = "<|endoftext|>"
            eos_id = tokenizer.convert_tokens_to_ids(eos_token)

        tokenizer.pad_token = eos_token
        tokenizer.pad_token_id = eos_id

        # Set conversation template
        if conversation_lib is not None:
            conversation_lib.default_conversation = conversation_lib.conv_templates["qwen"]

        # Add special tokens
        tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
        tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)

        logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
        return tokenizer

    def _preprocess_graph_token(self, text: str, num_graph_tokens: int) -> str:
        """Replace <graph> with patch tokens."""
        if DEFAULT_GRAPH_TOKEN not in text:
            return text

        replace_token = DEFAULT_GRAPH_PATCH_TOKEN * num_graph_tokens
        if self.use_graph_start_end:
            replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN

        return text.replace(DEFAULT_GRAPH_TOKEN, replace_token)

    def _create_hetero_data(self, graph_info: Dict) -> HeteroData:
        """Create HeteroData with edge_index from graph info."""
        node_list = graph_info.get('node_list', [])
        edge_index = graph_info.get('edge_index', [[], []])

        num_nodes = len(node_list) if node_list else 1

        hetero_data = HeteroData()

        # Node features (will be encoded by graph tokenizer)
        hetero_data["node"].x = torch.randn(num_nodes, self.node_feature_dim)

        # Create edge_index tensor if edges exist
        if edge_index and len(edge_index) == 2 and len(edge_index[0]) > 0:
            # Map node IDs to indices
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

    def _encode_graph(self, hetero_data: HeteroData) -> Optional[HeteroData]:
        """Encode graph using MetaHGT tokenizer."""
        if self.graph_tokenizer is None:
            return hetero_data

        try:
            graph_copy = hetero_data.clone()
            encoded_graph = self.graph_tokenizer.tokenize(graph_copy, use_graphgpt=False)
            return encoded_graph
        except Exception as e:
            logger.debug(f"Graph encoding failed: {e}")
            return None

    def _apply_prompt_template(self, sources: List[List[Dict]]):
        """Apply Qwen conversation template."""
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

            conversations.append(conv.get_prompt())

        return conversations, conv

    def _tokenize_and_mask(self, conversations: List[str], conv):
        """Tokenize and mask instruction parts for training."""
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

    def generate_stage1_data(
        self,
        num_samples: int = 20000,
        output_path: str = None
    ) -> List[Dict]:
        """
        Generate Stage 1 graph-text alignment data.

        Args:
            num_samples: Number of samples to generate (default: 20000)
            output_path: Path to save output PT file

        Returns:
            List of processed training samples
        """
        logger.info(f"Generating Stage 1 alignment data: {num_samples} samples")

        processed_data = []

        for i in tqdm(range(num_samples), desc="Stage 1"):
            # Generate random graph
            num_nodes = random.randint(5, 30)
            edge_prob = random.uniform(0.1, 0.4)

            # Generate edges
            edges_src = []
            edges_dst = []
            for src in range(num_nodes):
                for dst in range(src + 1, num_nodes):
                    if random.random() < edge_prob:
                        edges_src.append(src)
                        edges_dst.append(dst)
                        edges_src.append(dst)
                        edges_dst.append(src)

            num_edges = len(edges_src) // 2
            avg_degree = len(edges_src) / num_nodes if num_nodes > 0 else 0
            graph_type = "无向图" if random.random() > 0.5 else "undirected graph"

            # Select template
            template = random.choice(STAGE1_TEMPLATES)

            # Fill in template
            instruction = template["instruction"]
            response = template["response"].format(
                num_nodes=num_nodes,
                num_edges=num_edges,
                avg_degree=avg_degree,
                graph_type=graph_type,
                node_list=", ".join(str(n) for n in range(min(10, num_nodes))) + ("..." if num_nodes > 10 else "")
            )

            # Create graph info
            graph_info = {
                'node_list': list(range(num_nodes)),
                'edge_index': [edges_src, edges_dst]
            }

            # Create HeteroData with edge_index
            hetero_data = self._create_hetero_data(graph_info)

            # Encode with graph tokenizer
            encoded_data = self._encode_graph(hetero_data)
            if encoded_data is None:
                continue

            # Process conversations
            num_graph_tokens = num_nodes
            processed_instruction = self._preprocess_graph_token(instruction, num_graph_tokens)

            conversations = [
                {"from": "human", "value": processed_instruction},
                {"from": "gpt", "value": response}
            ]

            try:
                prompt_conversations, conv = self._apply_prompt_template([conversations])
                input_ids, labels = self._tokenize_and_mask(prompt_conversations, conv)
            except Exception as e:
                logger.debug(f"Failed to process sample: {e}")
                continue

            if input_ids.shape[1] >= self.max_length:
                continue

            if torch.all(labels == IGNORE_TOKEN_ID):
                continue

            result = {
                'id': f'gnn_stage1_{i}',
                'input_ids': input_ids[0],
                'labels': labels[0],
                'graph_data': encoded_data,
                'hetero_key_order': ['node'],
                'task_type': 'alignment',
            }

            processed_data.append(result)
            self.stats['success'] += 1

        self.stats['total'] = num_samples

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(processed_data, output_path)
            logger.info(f"Saved {len(processed_data)} Stage 1 samples to {output_path}")

        return processed_data

    def generate_stage2_data(
        self,
        converted_dir: str,
        samples_per_task: int = 1000,
        output_path: str = None
    ) -> List[Dict]:
        """
        Generate Stage 2 instruction tuning data.

        Args:
            converted_dir: Directory containing converted JSON files
            samples_per_task: Number of samples per task (default: 1000)
            output_path: Path to save output PT file

        Returns:
            List of processed training samples
        """
        logger.info(f"Generating Stage 2 data: {len(TASK_TYPES)} tasks × {samples_per_task} samples")

        processed_data = []

        for task_type in TASK_TYPES:
            task_file = os.path.join(converted_dir, f"{task_type}_converted.json")

            if not os.path.exists(task_file):
                logger.warning(f"Task file not found: {task_file}")
                continue

            logger.info(f"Processing task: {task_type}")

            with open(task_file, 'r', encoding='utf-8') as f:
                task_data = json.load(f)

            # Sample from task data
            if len(task_data) > samples_per_task:
                task_data = random.sample(task_data, samples_per_task)

            for idx, sample in enumerate(tqdm(task_data, desc=f"  {task_type}")):
                self.stats['total'] += 1

                result = self._process_sample(sample, f"{task_type}_{idx}")
                if result is not None:
                    result['task_type'] = task_type
                    processed_data.append(result)
                    self.stats['success'] += 1

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(processed_data, output_path)
            logger.info(f"Saved {len(processed_data)} Stage 2 samples to {output_path}")

        return processed_data

    def _process_sample(self, sample: Dict, sample_id: str) -> Optional[Dict]:
        """Process a single sample from converted data."""
        graph_info = sample.get('graph', {})
        conversations = sample.get('conversations', [])

        # Count graph tokens
        total_graph_tokens = sum(
            conv['value'].count(DEFAULT_GRAPH_TOKEN)
            for conv in conversations
        )

        if total_graph_tokens == 0:
            return None

        if total_graph_tokens > 1:
            return None

        # Get number of nodes
        node_list = graph_info.get('node_list', [])
        num_graph_tokens = len(node_list) if node_list else 1

        # Create HeteroData with edge_index
        hetero_data = self._create_hetero_data(graph_info)

        # Encode with graph tokenizer
        encoded_data = self._encode_graph(hetero_data)
        if encoded_data is None:
            return None

        # Process conversations
        processed_conversations = []
        for conv in conversations:
            processed_conv = copy.deepcopy(conv)
            processed_conv['value'] = self._preprocess_graph_token(
                processed_conv['value'], num_graph_tokens
            )
            processed_conversations.append(processed_conv)

        try:
            prompt_conversations, conv = self._apply_prompt_template([processed_conversations])
            input_ids, labels = self._tokenize_and_mask(prompt_conversations, conv)
        except Exception as e:
            return None

        if input_ids.shape[1] >= self.max_length:
            self.stats['too_long'] += 1
            return None

        if torch.all(labels == IGNORE_TOKEN_ID):
            return None

        return {
            'id': sample_id,
            'input_ids': input_ids[0],
            'labels': labels[0],
            'graph_data': encoded_data,
            'hetero_key_order': ['node'],
        }

    def print_stats(self):
        """Print generation statistics."""
        logger.info(f"\n{'='*60}")
        logger.info("Generation Statistics:")
        logger.info(f"  Total processed: {self.stats['total']}")
        logger.info(f"  Success: {self.stats['success']}")
        logger.info(f"  Failed: {self.stats['total'] - self.stats['success']}")
        logger.info(f"    - Too long: {self.stats['too_long']}")


def main():
    parser = argparse.ArgumentParser(description='Generate GNN training data for GraphAgent')
    parser.add_argument('--stage', type=str, choices=['1', '2', 'both'], default='both',
                        help='Stage to generate: 1, 2, or both')
    parser.add_argument('--converted-dir', type=str,
                        default='/nvme0/work/workspaces-zy/GraphInstruct/data/converted',
                        help='Directory containing converted JSON files')
    parser.add_argument('--output-dir', type=str,
                        default='/nvme0/work/workspaces-zy/GraphInstruct/data/training/gnn',
                        help='Output directory for training PT files')
    parser.add_argument('--model-path', type=str,
                        default='/nvme0/work/workspaces-zy/model/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507',
                        help='Path to Qwen model for tokenizer')
    parser.add_argument('--max-length', type=int, default=8192,
                        help='Maximum sequence length')
    parser.add_argument('--stage1-samples', type=int, default=20000,
                        help='Number of Stage 1 samples')
    parser.add_argument('--stage2-samples-per-task', type=int, default=1000,
                        help='Number of Stage 2 samples per task')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize generator
    generator = GNNTrainingDataGenerator(
        model_path=args.model_path,
        max_length=args.max_length,
    )

    # Generate Stage 1 data
    if args.stage in ['1', 'both']:
        stage1_output = os.path.join(args.output_dir, 'gnn_stage1_alignment.pt')
        generator.generate_stage1_data(
            num_samples=args.stage1_samples,
            output_path=stage1_output
        )

    # Generate Stage 2 data
    if args.stage in ['2', 'both']:
        stage2_output = os.path.join(args.output_dir, 'gnn_stage2_19tasks.pt')
        generator.generate_stage2_data(
            converted_dir=args.converted_dir,
            samples_per_task=args.stage2_samples_per_task,
            output_path=stage2_output
        )

    generator.print_stats()

    logger.info(f"\n{'='*60}")
    logger.info("Data generation complete!")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
