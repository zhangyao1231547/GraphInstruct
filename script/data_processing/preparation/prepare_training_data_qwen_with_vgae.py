#!/usr/bin/env python3
"""
GraphInstruct 训练数据准备脚本 - Qwen版本
将转换后的 JSON 数据处理为 GraphAgent 训练所需的 pt 格式

Qwen 对话格式:
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_message}<|im_end|>
"""

import json
import copy
import os
import sys
import logging
from typing import Dict, List, Optional
from tqdm import tqdm
import math
import argparse
from sklearn.decomposition import PCA
import torch
import transformers
from torch_geometric.data import HeteroData
from sample_graph_tokenizer import SampleGraphTokenizer
import torch
import numpy as np
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
# 添加 GraphAgent 训练代码路径
GRAPHAGENT_TRAINING_PATH = "/nvme0/work/workspaces-zy/GraphAgent-zy/GraphAGent-training"

VGAE_MODEL_PATH = "/home/test/workspaces-zy/GraphInstruct/script/graph_autoencoder/checkpoints_sin/final_model.pth"
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


class StableNodeEncoder(torch.nn.Module):
    def __init__(self, in_feat_dim, hidden_dim, emb_dim):
        super().__init__()
        self.conv1 = GCNConv(in_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, emb_dim)
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = torch.tanh(x)

        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        return x


class StableGraphAutoencoder(torch.nn.Module):
    def __init__(self, in_feat_dim, hidden_dim, emb_dim):
        super().__init__()
        self.encoder = StableNodeEncoder(in_feat_dim, hidden_dim, emb_dim)

    def forward(self, x, edge_index):
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)

        node_emb = self.encoder(x, edge_index)
        return node_emb


# ===================== 特征提取工具类 =====================
class GraphFeatureExtractor:
    """使用训练好的图自编码器提取节点特征"""

    def __init__(self, model_path='checkpoints/best_model.pth', device=None):
        """
        初始化特征提取器

        Args:
            model_path: 训练好的模型路径
            device: 使用的设备
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_config = {}

        # 加载模型
        self._load_model(model_path)

    def _load_model(self, model_path):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # 获取模型配置
        self.model_config = {
            'emb_dim': checkpoint.get('emb_dim', 768),
            'hidden_dim': checkpoint.get('hidden_dim', 32),
            'in_feat_dim': checkpoint.get('in_feat_dim', 16)
        }

        # 初始化模型
        self.model = StableGraphAutoencoder(
            in_feat_dim=self.model_config['in_feat_dim'],
            hidden_dim=self.model_config['hidden_dim'],
            emb_dim=self.model_config['emb_dim']
        ).to(self.device)

        # 加载权重
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        encoder_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key.replace('encoder.', '')
                encoder_state_dict[new_key] = value
            elif key.startswith('module.encoder.'):
                new_key = key.replace('module.encoder.', '')
                encoder_state_dict[new_key] = value

        # 加载权重
        if encoder_state_dict:
            self.model.load_state_dict(encoder_state_dict, strict=False)

        self.model.eval()
        print(f"已加载模型: {model_path}")
        print(f"编码维度: {self.model_config['emb_dim']}")


    def sinusoidal_pe(self, adj_matrix, feature_dim=16):
        """正弦位置编码（适用于节点位置）"""
        num_nodes = adj_matrix.shape[0]

        # 直接在GPU上创建所有张量
        position = torch.arange(num_nodes, dtype=torch.float32, device=self.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, feature_dim, 2, dtype=torch.float32, device=self.device)
            * -(math.log(10000.0) / feature_dim)
        )

        pe = torch.zeros(num_nodes, feature_dim, device=self.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) if feature_dim % 2 == 0 else torch.cos(position * div_term[:-1])

        return pe
    # def compute_structural_features(self, adj_matrix, feature_dim=16):
    #     """计算结构化特征（与训练时一致）"""
    #     num_nodes = adj_matrix.shape[0]
    #     adj_np = adj_matrix.cpu().numpy()
    #
    #     features = []
    #
    #     # 1. 度特征
    #     degree = adj_np.sum(axis=1)
    #     features.append(degree)
    #
    #     # 2. 度的平方根
    #     features.append(degree ** 0.5)
    #
    #     # 3. 邻居平均度
    #     neighbor_avg_degree = np.zeros(num_nodes)
    #     for i in range(num_nodes):
    #         neighbors = np.where(adj_np[i] > 0)[0]
    #         if len(neighbors) > 0:
    #             neighbor_avg_degree[i] = degree[neighbors].mean()
    #     features.append(neighbor_avg_degree)
    #
    #     # 4. 简化聚类系数
    #     clustering = np.zeros(num_nodes)
    #     for i in range(num_nodes):
    #         neighbors = np.where(adj_np[i] > 0)[0]
    #         if len(neighbors) >= 2:
    #             triangle_count = 0
    #             for j in range(len(neighbors)):
    #                 for k in range(j + 1, len(neighbors)):
    #                     if adj_np[neighbors[j], neighbors[k]] > 0:
    #                         triangle_count += 1
    #             clustering[i] = 2 * triangle_count / (len(neighbors) * (len(neighbors) - 1))
    #     features.append(clustering)
    #
    #     # 拼接特征
    #     features = np.stack(features, axis=1)
    #
    #     # 扩展维度
    #     while features.shape[1] < feature_dim:
    #         new_col = features[:, :4].mean(axis=1, keepdims=True)
    #         features = np.hstack([features, new_col])
    #
    #     features = features[:, :feature_dim]
    #
    #     # 安全归一化
    #     features = np.nan_to_num(features, nan=0.0)
    #     mean = features.mean(axis=0)
    #     std = features.std(axis=0)
    #     # print("--------------adj_matrix-----------")
    #     # print(adj_matrix)
    #     # print("--------------std-----------")
    #     # print(mean)
    #     # print(features)
    #     # print(std)
    #     std[std < 1e-6] = 1.0
    #     features = (features - mean) / std
    #
    #     return torch.FloatTensor(features).to(self.device)
    def compute_structural_features(self, adj_matrix, feature_dim=32):
        """正弦位置编码（适用于节点位置）"""
        num_nodes = adj_matrix.shape[0]

        # 直接在GPU上创建所有张量
        position = torch.arange(num_nodes, dtype=torch.float32, device=adj_matrix.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, feature_dim, 2, dtype=torch.float32, device=adj_matrix.device)
            * -(math.log(10000.0) / feature_dim)
        )

        pe = torch.zeros(num_nodes, feature_dim, device=adj_matrix.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) if feature_dim % 2 == 0 else torch.cos(position * div_term[:-1])

        return pe
    def encode_graph(self, edge_index, num_nodes):
        """
        编码图结构

        Args:
            edge_index: 边索引，形状为 [2, num_edges]
            num_nodes: 节点数量

        Returns:
            node_embeddings: 节点编码，形状为 [num_nodes, emb_dim]
        """
        # 创建邻接矩阵
        edge_index = edge_index.to(device=self.device)
        adj_matrix = torch.zeros((num_nodes, num_nodes), device=self.device)
        adj_matrix[edge_index[0], edge_index[1]] = 1.0

        # 计算结构化特征
        node_feat = self.compute_structural_features(adj_matrix, feature_dim=self.model_config['in_feat_dim'])
        node_feat = self.compute_structural_features(adj_matrix, feature_dim=768)

        # 使用模型编码
        # with torch.no_grad():
            # node_emb = self.model(node_feat, edge_index)

        # return node_emb.cpu()
        return node_feat.cpu()

    def encode_from_adj_matrix(self, adj_matrix):
        """
        直接从邻接矩阵编码

        Args:
            adj_matrix: 邻接矩阵，形状为 [num_nodes, num_nodes]

        Returns:
            node_embeddings: 节点编码
        """
        adj_matrix = adj_matrix.to(self.device)

        # 转换为edge_index
        edge_index = torch.nonzero(adj_matrix, as_tuple=False).t()

        # 计算结构化特征
        node_feat = self.compute_structural_features(adj_matrix, feature_dim=self.model_config['in_feat_dim'])

        # 使用模型编码
        with torch.no_grad():
            node_emb = self.model(node_feat, edge_index)

        return node_emb.cpu()


vgae_feature_extractor = GraphFeatureExtractor(VGAE_MODEL_PATH)

class QwenGraphInstructDataPreparer:
    """GraphInstruct 训练数据准备器 - Qwen版本"""

    def __init__(
        self,
        model_path: str,
        max_length: int = 4096,
        node_feature_dim: int = 768,
        use_graph_start_end: bool = True
    ):
        """
        Args:
            model_path: Qwen模型路径
            max_length: 最大序列长度
            node_feature_dim: 节点特征维度
            use_graph_start_end: 是否使用 <g_start> 和 <g_end> token
        """
        self.model_path = model_path
        self.max_length = max_length
        self.node_feature_dim = node_feature_dim
        self.use_graph_start_end = use_graph_start_end

        # 初始化 tokenizer
        self.tokenizer = self._setup_tokenizer()

        # 统计信息
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'too_long': 0,
            'no_graph': 0,
            'empty_labels': 0
        }

    def _setup_tokenizer(self) -> transformers.PreTrainedTokenizer:
        """设置 Qwen tokenizer"""
        logger.info(f"Loading Qwen tokenizer from {self.model_path}...")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=self.max_length,
            padding_side="right",
        )

        # Qwen 使用 <|im_end|> 或 <|endoftext|> 作为 EOS 和 pad token
        eos_token = "<|im_end|>"
        eos_id = tokenizer.convert_tokens_to_ids(eos_token)
        if eos_id is None or eos_id == tokenizer.unk_token_id:
            eos_token = "<|endoftext|>"
            eos_id = tokenizer.convert_tokens_to_ids(eos_token)

        tokenizer.pad_token = eos_token
        tokenizer.pad_token_id = eos_id
        logger.info(f"Using pad token: {eos_token} (id={eos_id})")

        # 设置 Qwen 对话模板
        if conversation_lib is not None:
            conversation_lib.default_conversation = conversation_lib.conv_templates["qwen"]

        # 添加特殊 token
        tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
        tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)

        logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
        logger.info(f"Special tokens: {DEFAULT_GRAPH_PATCH_TOKEN}={tokenizer.convert_tokens_to_ids(DEFAULT_GRAPH_PATCH_TOKEN)}, "
                   f"{DEFAULT_G_START_TOKEN}={tokenizer.convert_tokens_to_ids(DEFAULT_G_START_TOKEN)}, "
                   f"{DEFAULT_G_END_TOKEN}={tokenizer.convert_tokens_to_ids(DEFAULT_G_END_TOKEN)}")

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
            # Fallback: 手动构建 Qwen 格式
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
        """Tokenize 并 mask instruction 部分 - Qwen版本

        Qwen 的对话格式:
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        {assistant_message}<|im_end|>
        """
        input_ids = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False
        ).input_ids

        targets = input_ids.clone()

        # Qwen 的分隔符
        im_start = "<|im_start|>"
        im_end = "<|im_end|>"

        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(self.tokenizer.pad_token_id).sum())
            cur_len = 0

            # 分割对话
            parts = conversation.split(im_start)

            for i, part in enumerate(parts):
                if not part:
                    continue

                # 重新添加 <|im_start|> 前缀
                if i > 0:
                    part = im_start + part

                part_ids = self.tokenizer(part, add_special_tokens=False).input_ids
                part_len = len(part_ids)

                # 判断是否是 assistant 的回复
                if part.startswith(f"{im_start}assistant"):
                    # assistant 部分: mask 掉 "<|im_start|>assistant\n", 保留回答内容
                    header = f"{im_start}assistant\n"
                    header_len = len(self.tokenizer(header, add_special_tokens=False).input_ids)
                    # mask header 部分
                    target[cur_len:cur_len + header_len] = IGNORE_TOKEN_ID
                else:
                    # 非 assistant 部分全部 mask
                    target[cur_len:cur_len + part_len] = IGNORE_TOKEN_ID

                cur_len += part_len

            # mask 掉 padding 部分
            target[cur_len:] = IGNORE_TOKEN_ID

        return input_ids, targets


    def _create_hetero_data(self, graph_info: Dict) -> HeteroData:
        """从 graph_info 创建 HeteroData 图数据"""
        node_list = graph_info.get('node_list', [])
        edge_index = graph_info.get('edge_index', [[], []])

        # 创建 HeteroData 对象
        hetero_data = HeteroData()
        num_nodes = len(node_list) if node_list else 1
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        node_emb = vgae_feature_extractor.encode_graph(edge_index_tensor, num_nodes)

        hetero_data["node"].x = node_emb

        if edge_index and len(edge_index) == 2 and len(edge_index[0]) > 0:
            # 节点 ID 映射: 原始节点 ID -> 连续索引
            node_to_idx = {node: idx for idx, node in enumerate(node_list)}

            # 转换边索引
            mapped_src = []
            mapped_dst = []
            for src, dst in zip(edge_index[0], edge_index[1]):
                if src in node_to_idx and dst in node_to_idx:
                    mapped_src.append(node_to_idx[src])
                    mapped_dst.append(node_to_idx[dst])

            if mapped_src:
                edge_index_tensor = torch.tensor([mapped_src, mapped_dst], dtype=torch.long)
                hetero_data["node", "to", "node"].edge_index = edge_index_tensor
        # hetero_data.is_directed(graph_info.get('is_directed', True))
        return hetero_data



    def simple_torch_ascii_plot(self, points, width=60, height=20):
        """
        最简单的torch高维点ASCII可视化
        """
        # 转换和降维
        if torch.is_tensor(points):
            points_np = points.numpy()
        else:
            points_np = np.array(points)

        # 使用PCA降维
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(points_np)

        x, y = points_2d[:, 0], points_2d[:, 1]
        n_points = len(x)

        # 创建画布
        canvas = [['·' for _ in range(width)] for _ in range(height)]

        # 归一化
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        x_norm = (x - x_min) / (x_max - x_min) * 0.9 * width + 0.05 * width
        y_norm = (y - y_min) / (y_max - y_min) * 0.9 * height + 0.05 * height

        # 放置点
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i in range(n_points):
            xi, yi = int(x_norm[i]), int(y_norm[i])
            char = chars[i % len(chars)]
            if 0 <= xi < width and 0 <= yi < height:
                canvas[height - 1 - yi][xi] = char

        # 生成输出
        output = [f"Torch Points ({n_points} points)"]
        output.append(f"X range: [{x_min:.2f}, {x_max:.2f}]  Y range: [{y_min:.2f}, {y_max:.2f}]")
        output.append("=" * width)

        for row in canvas:
            output.append("".join(row))

        output.append("=" * width)
        output.append("Legend: " + " ".join([f"{chars[i]}:{i}" for i in range(min(n_points, 10))]))

        return "\n".join(output)
    def process_sample(self, sample: Dict) -> Optional[Dict]:
        """处理单个样本"""
        sample_id = sample.get('id', 'unknown')
        graph_info = sample.get('graph', {})
        conversations = sample.get('conversations', [])

        # 计算对话中 <graph> token 的数量
        total_graph_tokens = sum(
            conv['value'].count(DEFAULT_GRAPH_TOKEN)
            for conv in conversations
        )

        # 过滤没有 <graph> 的样本
        if total_graph_tokens == 0:
            self.stats['no_graph'] += 1
            return None

        # 过滤多个 <graph> 的样本
        if total_graph_tokens > 1:
            logger.debug(f"Sample {sample_id} has {total_graph_tokens} <graph> tokens, skipping")
            self.stats['failed'] += 1
            return None

        # 获取图节点数量
        node_list = graph_info.get('node_list', [])
        num_graph_tokens = len(node_list) if node_list else 1

        # 预处理对话，替换 <graph> token
        processed_conversations = []
        for conv in conversations:
            processed_conv = copy.deepcopy(conv)
            processed_conv['value'] = self._preprocess_graph_token(
                processed_conv['value'], num_graph_tokens
            )
            processed_conversations.append(processed_conv)

        # 应用对话模板并 tokenize
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

        # 检查 labels 是否全为 -100
        if torch.all(labels == IGNORE_TOKEN_ID):
            self.stats['empty_labels'] += 1
            return None

        # 创建图数据
        graph_data = self._create_hetero_data(graph_info)

        # # graph_data_a = self.metahgt_encoding(graph_data)
        # edge_index = graph_info.get('edge_index', [[], []])
        # ascii_art = self.simple_torch_ascii_plot(graph_data.x_dict['node'])
        # normalized_emb = F.normalize(graph_data.x_dict['node'], p=2, dim=1)
        # similarity = torch.mm(normalized_emb, normalized_emb.t())
        # print(edge_index)
        # print(ascii_art)
        # print(similarity)
        # 构建返回字典
        result = {
            'id': sample_id,
            'input_ids': input_ids[0],
            'labels': labels[0],
            'graph_data': graph_data,
            'hetero_key_order': ['node'],
            'task_type': sample.get('task_type', 'unknown'),
        }

        return result

    def process_file(self, input_path: str, output_path: str, max_samples: Optional[int] = None):
        """处理单个 JSON 文件"""
        logger.info(f"Processing {input_path}...")

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if max_samples:
            data = data[:max_samples]

        logger.info(f"Total samples to process: {len(data)}")

        processed_data = []
        for sample in tqdm(data, desc="Processing samples"):
            self.stats['total'] += 1
            result = self.process_sample(sample)
            if result is not None:
                processed_data.append(result)
                self.stats['success'] += 1
            else:
                self.stats['failed'] += 1

        # 保存
        logger.info(f"Saving {len(processed_data)} samples to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(processed_data, output_path)

        return processed_data

    def process_all(self, converted_dir: str, output_dir: str, max_samples_per_task: Optional[int] = None):
        """处理所有转换后的文件"""
        # # 优先查找合并文件
        # merged_path = os.path.join(converted_dir, 'merged_graphinstruct.json')
        # if os.path.exists(merged_path):
        #     output_path = os.path.join(output_dir, 'graphinstruct_train_qwen.pt')
        #     self.process_file(merged_path, output_path, max_samples_per_task)
        #     self._print_stats()
        #     return

        # 查找所有 *_converted.json 文件
        converted_files = ["common_neighbor_converted.json"]
        # for f in os.listdir(converted_dir):
        #     if f.endswith('_converted.json'):
        #         converted_files.append(f)

        all_processed = []
        for filename in converted_files:
            input_path = os.path.join(converted_dir, filename)
            task_name = filename.replace('_converted.json', '')
            output_path = os.path.join(output_dir, f'{task_name}_train_qwen.pt')

            processed = self.process_file(input_path, output_path, max_samples_per_task)
            all_processed.extend(processed)

        # 保存合并后的数据
        merged_output = os.path.join(output_dir, 'graphinstruct_train_qwen_merged.pt')
        logger.info(f"Saving merged data ({len(all_processed)} samples) to {merged_output}...")
        torch.save(all_processed, merged_output)

        self._print_stats()

    def _print_stats(self):
        """打印统计信息"""
        logger.info(f"\n{'='*60}")
        logger.info("Processing Statistics (Qwen):")
        logger.info(f"  Total processed: {self.stats['total']}")
        logger.info(f"  Success: {self.stats['success']}")
        logger.info(f"  Failed: {self.stats['failed']}")
        logger.info(f"    - Too long: {self.stats['too_long']}")
        logger.info(f"    - No graph token: {self.stats['no_graph']}")
        logger.info(f"    - Empty labels: {self.stats['empty_labels']}")


def verify_qwen_data(output_path: str, tokenizer, num_samples: int = 3):
    """验证生成的 Qwen 数据"""
    logger.info(f"\n===== 验证数据: {output_path} =====")
    data = torch.load(output_path, weights_only=False)
    logger.info(f"总样本数: {len(data)}")

    for i in range(min(num_samples, len(data))):
        sample = data[i]
        input_ids = sample['input_ids']
        labels = sample['labels']

        logger.info(f"\n--- 样本 {i} ---")
        total_tokens = labels.shape[0]
        masked_tokens = (labels == -100).sum().item()
        valid_tokens = total_tokens - masked_tokens

        logger.info(f"总token数: {total_tokens}")
        logger.info(f"被mask: {masked_tokens} ({100*masked_tokens/total_tokens:.1f}%)")
        logger.info(f"有效训练: {valid_tokens} ({100*valid_tokens/total_tokens:.1f}%)")

        # 解码验证
        text = tokenizer.decode(input_ids[:100].tolist(), skip_special_tokens=False)
        logger.info(f"前100 token解码:\n{text[:300]}...")


def main():
    parser = argparse.ArgumentParser(description='Prepare GraphInstruct training data for GraphAgent (Qwen version)')
    parser.add_argument('--input-dir', type=str,
                        default='/nvme0/work/workspaces-zy/GraphInstruct/data/converted',
                        help='Directory containing converted JSON files')
    parser.add_argument('--output-dir', type=str,
                        default='/nvme0/work/workspaces-zy/GraphInstruct/data/training',
                        help='Output directory for training pt files')
    parser.add_argument('--model-path', type=str,
                        default='/nvme0/work/workspaces-zy/model/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507',
                        help='Path to Qwen model for tokenizer')
    parser.add_argument('--max-length', type=int, default=4096,
                        help='Maximum sequence length')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples per task (for testing)')
    parser.add_argument('--node-dim', type=int, default=768,
                        help='Node feature dimension')
    parser.add_argument('--verify', action='store_true',
                        help='Verify generated data after processing')

    args = parser.parse_args()

    # 创建数据准备器
    preparer = QwenGraphInstructDataPreparer(
        model_path=args.model_path,
        max_length=args.max_length,
        node_feature_dim=args.node_dim
    )

    # 处理所有文件
    preparer.process_all(
        args.input_dir,
        args.output_dir,
        args.max_samples
    )

    # 验证数据
    if args.verify:
        output_path = os.path.join(args.output_dir, 'graphinstruct_train_qwen.pt')
        if os.path.exists(output_path):
            verify_qwen_data(output_path, preparer.tokenizer)


if __name__ == '__main__':
    main()
