import os
import torch
from torch_geometric.data import HeteroData
from typing import Dict, Optional, Any
import glob
import json
import sys
GRAPHAGENT_TRAINING_PATH = "/nvme0/work/workspaces-zy/GraphAgent-zy/GraphAGent-training"
sys.path.insert(0, GRAPHAGENT_TRAINING_PATH)
GRAPHAGENT_INFERENCE_PATH = "/nvme0/work/workspaces-zy/GraphAgent-zy/GraphAgent-inference"
sys.path.insert(0, GRAPHAGENT_INFERENCE_PATH)
try:
    from model.graph_action_agent import conversation as conversation_lib
except ImportError:
    conversation_lib = None


class SampleGraphTokenizer:
    """图编码器工具类，封装异构图编码功能"""

    def __init__(self,
                 device: str = None,
                 sentence_transformer_path: str = None,
                 pretrained_gnn_path: str = None):
        """
        初始化图编码器

        Args:
            device: 设备 ('cuda' 或 'cpu')
            sentence_transformer_path: SentenceTransformer 模型路径
            pretrained_gnn_path: 预训练图编码器路径
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sentence_transformer = None
        self.metahgt_model = None
        self._initialize_models(sentence_transformer_path, pretrained_gnn_path)

        # 描述性属性映射
        self.describe_keys = {
            "User": ["userName"],
            "OperationLog": ["bizAction"],
            "Organization": ["name"],
            "Ticket": ["requirementDescription", "solution"],
            "Project": ["customer", "name"],
            "Domain": ["name"],
            "TeamworkType": ["name"],
            "ProductLine": ["name"],
            "Module": ["name"],
            "File": ["uri"],
            "Version": ["versionName"],
            "Region": ["name"]
        }

    def _initialize_models(self, sentence_transformer_path: str, pretrained_gnn_path: str):
        """初始化模型"""
        # 设置默认路径
        if sentence_transformer_path is None:
            sentence_transformer_path = os.environ.get(
                'SENTENCE_TRANSFORMER_MODEL_PATH',
                '/nvme0/work/workspaces-zy/model/GraphAgent/all-mpnet-base-v2'
            )

        if pretrained_gnn_path is None:
            pretrained_gnn_path = os.environ.get(
                'GRAPH_TOKENIZE_MODEL_PATH',
                '/nvme0/work/workspaces-zy/model/GraphAgent/GraphTokenizer'
            )

        # 初始化 SentenceTransformer
        try:
            from sentence_transformers import SentenceTransformer
            print(f"加载 SentenceTransformer: {sentence_transformer_path}")
            self.sentence_transformer = SentenceTransformer(sentence_transformer_path)
            print("✓ SentenceTransformer 加载成功")
        except Exception as e:
            print(f"✗ SentenceTransformer 加载失败: {e}")
            self.sentence_transformer = None

        # 初始化 MetaHGT 模型
        try:
            print(f"加载预训练图编码器: {pretrained_gnn_path}")
            self.metahgt_model = self._load_graph_tokenizer_pretrained(
                self.MetaHGTConv, pretrained_gnn_path
            )
            self.metahgt_model = self.metahgt_model.to(self.device)
            self.metahgt_model.eval()  # 设置为评估模式
            print("✓ 图编码器加载成功")
        except Exception as e:
            print(f"✗ 图编码器加载失败: {e}")
            self.metahgt_model = None

    @property
    def MetaHGTConv(self):
        """动态导入 MetaHGTConv 类"""
        # 根据您的实际导入路径调整
        try:
            from graph_action_agent.graphllm.meta_hgt import MetaHGTConv
            return MetaHGTConv
        except ImportError:
            # 备用导入路径
            try:
                from model.graph_action_agent.graphllm.meta_hgt import MetaHGTConv
                return MetaHGTConv
            except ImportError as e:
                raise ImportError(f"无法导入 MetaHGTConv: {e}")

    @property
    def MetaHGTConvCfg(self):
        """动态导入 MetaHGTConvCfg 类"""
        try:
            from graph_action_agent.graphllm.meta_hgt import MetaHGTConvCfg
            return MetaHGTConvCfg
        except ImportError:
            try:
                from model.graph_action_agent.graphllm.meta_hgt import MetaHGTConvCfg
                return MetaHGTConvCfg
            except ImportError as e:
                raise ImportError(f"无法导入 MetaHGTConvCfg: {e}")

    @property
    def CLIPTextCfg(self):
        """动态导入 CLIPTextCfg 类"""
        try:
            from graph_action_agent.graphllm.heteclip_models import CLIPTextCfg
            return CLIPTextCfg
        except ImportError:
            try:
                from model.graph_action_agent.graphllm.heteclip_models import CLIPTextCfg
                return CLIPTextCfg
            except ImportError as e:
                raise ImportError(f"无法导入 CLIPTextCfg: {e}")

    def _load_graph_tokenizer_pretrained(self, model_name, pretrain_model_path):
        """加载预训练图编码器"""
        import os.path as osp

        # 检查配置文件
        assert osp.exists(osp.join(pretrain_model_path, 'graph_config.json')), 'graph_config.json missing'
        with open(osp.join(pretrain_model_path, 'graph_config.json'), 'r') as f:
            graph_config_dict = json.load(f)
        graph_cfg = self.MetaHGTConvCfg(**graph_config_dict)

        assert osp.exists(osp.join(pretrain_model_path, 'text_config.json')), 'text_config.json missing'
        with open(osp.join(pretrain_model_path, 'text_config.json'), 'r') as f:
            text_config_dict = json.load(f)
        text_cfg = self.CLIPTextCfg(**text_config_dict)

        # 创建模型
        model = model_name(
            in_channels=graph_cfg.in_channels,
            out_channels=graph_cfg.out_channels,
            heads=graph_cfg.heads,
            dynamic=graph_cfg.dynamic,
            text_cfg=text_cfg,
        )

        # 加载权重
        pkl_files = glob.glob(osp.join(pretrain_model_path, '*.ckpt'))
        if not pkl_files:
            raise FileNotFoundError(f"未找到 .ckpt 文件在 {pretrain_model_path}")

        state_dict = torch.load(pkl_files[0], map_location='cpu')['state_dict']
        print('加载图编码器权重...')

        gnn_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.graph_encoder'):
                new_key = key.split('model.graph_encoder.')[1]
                gnn_state_dict[new_key] = value

        model.load_state_dict(gnn_state_dict, strict=False)
        return model

    def build_meta_type_emb_dict(self, pyg_graph: HeteroData) -> Dict[str, Any]:
        """构建元类型嵌入字典"""
        if self.sentence_transformer is None:
            raise RuntimeError("SentenceTransformer 未初始化")

        node_type_emb_dict = {}
        for node_type in pyg_graph.node_types:
            node_type_emb_dict[node_type] = self.sentence_transformer.encode(
                [node_type], convert_to_tensor=True, show_progress_bar=False
            )[0]

        edge_type_emb_dict = {}
        for edge_type in pyg_graph.edge_types:
            edge_type_emb_dict[edge_type] = self.sentence_transformer.encode(
                [edge_type], convert_to_tensor=True, show_progress_bar=False
            )[0]

        return {
            "node_type_emb_dict": node_type_emb_dict,
            "edge_type_emb_dict": edge_type_emb_dict
        }

    def encode_node_text(self, pyg_graph: HeteroData) -> HeteroData:
        """编码节点文本"""
        if self.sentence_transformer is None:
            raise RuntimeError("SentenceTransformer 未初始化")

        x_dict = {}
        for node_type in pyg_graph.node_types:
            node_set = pyg_graph[node_type]

            if 'description' not in node_set:
                print(f"警告: {node_type} 节点没有 description 属性")
                continue

            descriptions = node_set['description']
            x_dict_type_i = torch.zeros(len(descriptions), 768)

            for i, node_text in enumerate(descriptions):
                node_text_emb = self.sentence_transformer.encode(
                    [node_text], convert_to_tensor=True, show_progress_bar=False
                )[0]
                x_dict_type_i[i] = node_text_emb

            # 更新节点特征
            node_set['x'] = x_dict_type_i
            x_dict[node_type] = x_dict_type_i

        pyg_graph.x_dict = x_dict
        return pyg_graph

    def tokenize(self,
                 pyg_graph: HeteroData,
                 use_graphgpt: bool = False,
                 encoder_path: Optional[str] = None) -> HeteroData:
        """
        对异构图进行编码

        Args:
            pyg_graph: PyG 异构图
            use_graphgpt: 是否使用 GraphGPT 编码器（目前仅支持 MetaHGT）
            encoder_path: GraphGPT 编码器路径

        Returns:
            编码后的图（在 CPU 上）
        """
        if self.metahgt_model is None:
            raise RuntimeError("图编码器未初始化")

        # 构建类型嵌入字典
        node_edge_type_emb_dict = self.build_meta_type_emb_dict(pyg_graph)

        # 编码节点文本
        pyg_graph = self.encode_node_text(pyg_graph)

        # 准备数据并移动到设备
        node_feas_dict = node_edge_type_emb_dict["node_type_emb_dict"]
        edge_feas_dict = node_edge_type_emb_dict["edge_type_emb_dict"]

        for k, v in node_feas_dict.items():
            node_feas_dict[k] = v.to(self.device)

        for k, v in edge_feas_dict.items():
            edge_feas_dict[k] = v.to(self.device)

        # 将图移动到设备
        pyg_graph = pyg_graph.to(self.device)

        # 转换边索引类型
        for k in pyg_graph.edge_index_dict:
            pyg_graph[k].edge_index = pyg_graph[k].edge_index.to(torch.int64)

        # 准备节点和边特征字典
        node_type_feat_dict_i = {}
        for k in pyg_graph.x_dict:
            node_type_feat_dict_i[k] = node_feas_dict[k]

        edge_type_feat_dict_i = {}
        for k in pyg_graph.edge_index_dict:
            edge_type_feat_dict_i[k] = edge_feas_dict[k]

        # 编码图
        with torch.no_grad():
            res = self.metahgt_model(
                x_dict=pyg_graph.x_dict,
                edge_index_dict=pyg_graph.edge_index_dict,
                node_type_feas_dict=node_type_feat_dict_i,
                edge_type_feas_dict=edge_type_feat_dict_i
            )

        # 确保结果在 CPU 上（重要！）
        cpu_res = {}
        for key, tensor in res.items():
            cpu_res[key] = tensor.cpu().detach()

        # 更新图数据
        pyg_graph.x_dict = cpu_res

        # 将整个图移回 CPU
        pyg_graph = self.move_to_cpu(pyg_graph)

        return pyg_graph

    def move_to_cpu(self, pyg_graph: HeteroData) -> HeteroData:
        """将 HeteroData 中的所有张量移动到 CPU"""
        # 处理节点特征
        for node_type in pyg_graph.x_dict:
            if isinstance(pyg_graph.x_dict[node_type], torch.Tensor):
                pyg_graph.x_dict[node_type] = pyg_graph.x_dict[node_type].cpu()

        # 处理边索引
        for edge_type in pyg_graph.edge_index_dict:
            if isinstance(pyg_graph.edge_index_dict[edge_type], torch.Tensor):
                pyg_graph.edge_index_dict[edge_type] = pyg_graph.edge_index_dict[edge_type].cpu()

        return pyg_graph



    def test_encode(self, test_graph: HeteroData = None) -> tuple:
        """测试编码功能"""
        if test_graph is None:
            test_graph = self._create_test_graph()

        print("测试编码...")
        print("编码前:")
        print(test_graph)

        try:
            encoded_graph = self.tokenize(test_graph, use_graphgpt=False)
            print("编码后:")
            print(encoded_graph)

            # 验证编码结果
            for node_type in encoded_graph.x_dict:
                if isinstance(encoded_graph.x_dict[node_type], torch.Tensor):
                    print(f"{node_type} 特征形状: {encoded_graph.x_dict[node_type].shape}")
                    print(f"{node_type} 特征设备: {encoded_graph.x_dict[node_type].device}")

            return encoded_graph, True

        except Exception as e:
            print(f"编码测试失败: {e}")
            import traceback
            traceback.print_exc()
            return None, False

    def _create_test_graph(self) -> HeteroData:
        """创建测试图"""
        graph = HeteroData()

        # 添加节点
        graph['paper'].x = torch.randn(5, 768)
        graph['paper'].description = [
            "Graph neural networks for node classification",
            "Deep learning on graphs",
            "Message passing neural networks",
            "Attention mechanisms in GNNs",
            "Heterogeneous graph transformers"
        ]

        graph['author'].x = torch.randn(3, 768)
        graph['author'].description = [
            "Researcher in machine learning",
            "Expert in graph theory",
            "Professor of computer science"
        ]

        # 添加边
        graph['paper', 'cites', 'paper'].edge_index = torch.tensor([
            [0, 1, 2, 3],
            [1, 2, 3, 4]
        ])

        graph['author', 'writes', 'paper'].edge_index = torch.tensor([
            [0, 1, 2, 0, 1],
            [0, 1, 2, 3, 4]
        ])

        return graph


# def main():
#     # 创建图编码器实例
#     graph_tokenizer = SampleGraphTokenizer(
#         device='cuda:0' if torch.cuda.is_available() else 'cpu',
#         sentence_transformer_path='/nvme0/work/workspaces-zy/model/GraphAgent/all-mpnet-base-v2',
#         pretrained_gnn_path='/nvme0/work/workspaces-zy/model/GraphAgent/GraphTokenizer'
#     )
#
#     # 测试编码
#     encoded_graph, success = graph_tokenizer.test_encode()
#
#     if success:
#         print("✓ 图编码器测试成功")
#
# if __name__ == "__main__":
#     main()