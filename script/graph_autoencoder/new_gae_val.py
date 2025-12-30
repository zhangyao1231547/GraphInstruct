import torch
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import warnings
import math
import json
warnings.filterwarnings('ignore')

VAL_PATH = "/home/test/workspaces-zy/GraphInstruct/data/converted/common_neighbor_converted.json"
# ===================== 简化的特征生成（避免复杂计算） =====================
def sinusoidal_pe(adj_matrix, feature_dim=32):
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
import random
# ===================== 简化的图生成 =====================
def generate_simple_community_graph(min_nodes=5, max_nodes=20, edge_prob=0.2):
    """简化图生成"""
    num_nodes = random.randint(min_nodes, max_nodes)

    # 生成邻接矩阵（对称，无自环）
    adj_matrix = torch.zeros((num_nodes, num_nodes))

    # adj_matrix.fill_diagonal_(1.0)
    # 为每个可能的无向边生成随机连接
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # j > i 避免自环和重复
            if random.random() < edge_prob:
                # 无向边：同时设置i->j和j->i
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0

    # 确保至少有一些边
    while adj_matrix.sum() == 0:
        # 如果没有边，随机添加一条
        i, j = random.sample(range(num_nodes), 2)
        adj_matrix[i, j] = 1.0
        adj_matrix[j, i] = 1.0


    adj_matrix = torch.FloatTensor(adj_matrix)
    node_feat = sinusoidal_pe(adj_matrix, feature_dim=32)
    edge_index = torch.nonzero(adj_matrix, as_tuple=False).t()

    return edge_index, adj_matrix, node_feat,num_nodes


# ===================== 修复的模型（避免NaN） =====================
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


class MatrixAdjDecoder(torch.nn.Module):
    """基于矩阵运算的高效解码器"""

    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

        # 使用双线性变换
        self.bilinear = torch.nn.Bilinear(emb_dim, emb_dim, 1)
        torch.nn.init.normal_(self.bilinear.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.bilinear.bias)

        self.sigmoid = torch.nn.Sigmoid()



    def forward(self, node_emb):
        n = node_emb.shape[0]

        # 方法1：双线性注意力（高效）
        # 扩展维度用于批量双线性运算
        node_i = node_emb.unsqueeze(1).expand(n, n, self.emb_dim)  # [n, n, emb_dim]
        node_j = node_emb.unsqueeze(0).expand(n, n, self.emb_dim)  # [n, n, emb_dim]

        # 重塑为2D用于批量处理
        node_i_flat = node_i.reshape(-1, self.emb_dim)  # [n*n, emb_dim]
        node_j_flat = node_j.reshape(-1, self.emb_dim)  # [n*n, emb_dim]

        # 批量双线性计算
        logits = self.bilinear(node_i_flat, node_j_flat)  # [n*n, 1]
        logits = logits.reshape(n, n)

        # 温度缩放 - 稳定训练

        adj_recon = self.sigmoid(logits)  # [n, n]

        # 确保对称性
        adj_recon = (adj_recon + adj_recon.t()) / 2

        return adj_recon


class StableGraphAutoencoder(torch.nn.Module):
    def __init__(self, in_feat_dim, hidden_dim, emb_dim):
        super().__init__()
        self.encoder = StableNodeEncoder(in_feat_dim, hidden_dim, emb_dim)
        self.decoder = MatrixAdjDecoder(emb_dim)


    def forward(self, x, edge_index):
        # 检查输入
        if torch.isnan(x).any():
            print("警告: 模型输入包含NaN!")
            x = torch.nan_to_num(x, nan=0.0)


        # 编码
        node_emb = self.encoder(x, edge_index)

        # 解码
        adj_recon = self.decoder(node_emb)

        return node_emb, adj_recon


def process_sample(sample, feature_dim=32):
    """
    从.pt文件加载图数据

    Args:
        pt_path: .pt文件路径
        sample_idx: 样本索引（如果文件包含多个样本）
        feature_dim: 节点特征维度

    Returns:
        edge_index: [2, num_edges]
        adj_matrix: [num_nodes, num_nodes]
        node_feat: [num_nodes, feature_dim]
        num_nodes: 节点数量
    """



    # 获取图信息
    graph_info = sample.get('graph', {})

    # 获取边索引
    edge_list = graph_info.get('edge_index', [[], []])

    if not edge_list or len(edge_list[0]) == 0:
        # 如果没有边，创建空图
        print("警告: 没有找到边信息，创建空图")
        num_nodes = graph_info.get('num_nodes', 1)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    else:
        # 转换边列表为tensor
        if isinstance(edge_list, list):
            edge_index = torch.tensor(edge_list, dtype=torch.long)
        elif isinstance(edge_list, torch.Tensor):
            edge_index = edge_list.long()
        else:
            raise ValueError(f"无法识别的edge_index类型: {type(edge_list)}")

        # 确保edge_index是[2, num_edges]格式
        if edge_index.dim() == 2 and edge_index.shape[0] != 2:
            if edge_index.shape[1] == 2:
                edge_index = edge_index.t()  # 转置为[2, num_edges]
            else:
                raise ValueError(f"edge_index形状异常: {edge_index.shape}")

        # 计算节点数量
        if 'num_nodes' in graph_info:
            num_nodes = graph_info['num_nodes']
        else:
            num_nodes = edge_index.max().item() + 1

        # 创建邻接矩阵
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        adj_matrix[edge_index[0], edge_index[1]] = 1.0

        # 确保无向图的对称性（如果适用）
        if graph_info.get('undirected', True):
            adj_matrix = adj_matrix + adj_matrix.t()
            adj_matrix = (adj_matrix > 0).float()

    # 生成节点特征（使用正弦位置编码）
    node_feat = sinusoidal_pe(adj_matrix, feature_dim=feature_dim)

    return edge_index, adj_matrix, node_feat, num_nodes
def load_validata(max_samples=1000):
    val_graphs = []
    with open(VAL_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    for sample in tqdm(data, desc="Processing samples"):
        edge_index, adj_matrix, node_feat, num_nodes = process_sample(sample)
        val_graphs.append(Data(x=node_feat, edge_index=edge_index, adj=adj_matrix, num_nodes=num_nodes))
    return val_graphs

# ===================== 修复的训练函数 =====================
def stable_train_adj_recon_gae():
    parser = argparse.ArgumentParser(description='稳定的图自编码器训练')
    parser.add_argument('--min_nodes', type=int, default=3, help='最小节点数')
    parser.add_argument('--max_nodes', type=int, default=20, help='最大节点数')
    parser.add_argument('--edge_prob', type=float, default=0.3, help='边生成概率')
    parser.add_argument('--emb_dim', type=int, default=768, help='节点编码维度')
    parser.add_argument('--hidden_dim', type=int, default=32, help='编码器隐藏层维度')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--visualize', action='store_true', default=True, help='可视化结果')
    parser.add_argument('--save_dir', type=str, default='checkpoints_sin_val', help='模型保存目录')
    args = parser.parse_args()

    print("=" * 60)
    print("稳定的图自编码器训练")
    print("=" * 60)

    # 创建保存目录
    import os
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 生成训练数据
    print("\n生成训练数据...")
    # graphs = []
    # for idx in tqdm(range(10000), desc="生成图数据", ncols=100):
    #
    #     edge_index, adj_matrix, node_feat, num_nodes = generate_simple_community_graph(
    #         args.min_nodes, args.max_nodes, args.edge_prob
    #     )
    #
    #     # 数值检查
    #     if torch.isnan(node_feat).any():
    #         continue
    #
    #     graphs.append(Data(x=node_feat, edge_index=edge_index, adj=adj_matrix, num_nodes=num_nodes))

    # print(f"生成 {len(graphs)} 个有效的训练图")

    # 2. 初始化模型
    in_feat_dim = 32
    model = StableGraphAutoencoder(
        in_feat_dim=in_feat_dim,
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim
    )

    print(f"\n模型信息:")
    print(f"  输入特征维度: {in_feat_dim}")
    print(f"  隐藏层维度: {args.hidden_dim}")
    print(f"  嵌入维度: {args.emb_dim}")
    print(f"  总参数: {sum(p.numel() for p in model.parameters()):,}")

    vali_graphs = load_validata(max_samples=10000)
    # 3. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # 4. 训练循环
    train_history = {'loss': [], 'auc': [], 'acc': []}
    best_loss = float('inf')
    best_auc = 0.0
    best_model_state = None

    print(f"\n开始训练...")
    model.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_auc = 0.0
        epoch_acc = 0.0
        num_batches = 0
        graphs = vali_graphs
        # graphs = []
        # for idx in tqdm(range(10000), desc="生成图数据", ncols=100):
        #
        #     edge_index, adj_matrix, node_feat, num_nodes = generate_simple_community_graph(
        #         args.min_nodes, args.max_nodes, args.edge_prob
        #     )
        #
        #     # 数值检查
        #     if torch.isnan(node_feat).any():
        #         continue
        #
        #     graphs.append(Data(x=node_feat, edge_index=edge_index, adj=adj_matrix, num_nodes=num_nodes))

        np.random.shuffle(graphs)
        progress_bar = tqdm(graphs,
                            total=len(graphs),
                            desc=f"Epoch {epoch + 1:03d}/{args.epochs}",
                            ncols=100)

        for graph in progress_bar:
            optimizer.zero_grad()

            # 前向传播
            node_emb, adj_recon = model(graph.x, graph.edge_index)

            # 调试输出（仅第一个epoch的第一个batch）
            if num_batches == 0:
                print(f"\n第一个图的调试信息:")
                print(f"  节点特征范围: [{graph.x.min().item():.4f}, {graph.x.max().item():.4f}]")
                print(f"  编码输出范围: [{node_emb.min().item():.4f}, {node_emb.max().item():.4f}]")
                print(f"  重建输出范围: [{adj_recon.min().item():.4f}, {adj_recon.max().item():.4f}]")

            # 计算损失
            loss = adj_reconstruction_loss(graph.adj, adj_recon)

            # 检查损失是否有效
            if torch.isnan(loss):
                continue

            # 反向传播
            loss.backward()

            # 梯度检查
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    param.grad = torch.nan_to_num(param.grad, nan=0.0)

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参数
            optimizer.step()

            # print(graph.edge_index)
            # 评估
            with torch.no_grad():
                metrics = eval_adj_recon(graph.adj, adj_recon)

            epoch_loss += loss.item()
            epoch_auc += metrics['auc']
            epoch_acc += metrics['acc']
            num_batches += 1

        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            avg_auc = epoch_auc / num_batches
            avg_acc = epoch_acc / num_batches

            train_history['loss'].append(avg_loss)
            train_history['auc'].append(avg_auc)
            train_history['acc'].append(avg_acc)

            # 更新学习率
            scheduler.step()

            # 每2个epoch评估并保存模型
            if (epoch + 1) % 1 == 0:
                # 评估当前模型在验证集上的表现
                val_loss, val_auc, val_acc = evaluate_model(model, vali_graphs)  # 使用前100个图作为验证集

                print(f"\nEpoch {epoch + 1:03d}/{args.epochs}")
                print(f"  训练 - Loss: {avg_loss:.4f} | AUC: {avg_auc:.4f} | ACC: {avg_acc:.4f}")
                print(f"  验证 - Loss: {val_loss:.4f} | AUC: {val_auc:.4f} | ACC: {val_acc:.4f}")
                print(f"  学习率: {scheduler.get_last_lr()[0]:.6f}")

                # 保存最佳模型（基于验证损失）
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_auc = val_auc
                    best_model_state = model.state_dict().copy()

                    # 保存最佳模型
                    checkpoint_path = os.path.join(args.save_dir, f'best_model.pth')
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': best_model_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': best_loss,
                        'auc': best_auc,
                        'emb_dim': args.emb_dim,
                        'hidden_dim': args.hidden_dim,
                        'in_feat_dim': in_feat_dim,
                        'train_history': train_history
                    }, checkpoint_path)
                    print(f"  ✓ 保存最佳模型到: {checkpoint_path}")

                # 每2个epoch保存一次检查点
                checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1:03d}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'auc': avg_auc,
                    'acc': avg_acc,
                    'emb_dim': args.emb_dim,
                    'hidden_dim': args.hidden_dim,
                    'in_feat_dim': in_feat_dim,
                    'train_history': train_history
                }, checkpoint_path)
                print(f"  ✓ 保存检查点到: {checkpoint_path}")

    # 5. 加载最佳模型进行测试
    print("\n" + "=" * 60)
    print("测试阶段 - 使用最佳模型")
    print("=" * 60)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    test_sizes = [8, 12, 16]

    for num_nodes in test_sizes:
        edge_index, adj_matrix, node_feat, num_nodes = generate_simple_community_graph(
            args.min_nodes, args.max_nodes, args.edge_prob
        )

        with torch.no_grad():
            node_emb, adj_recon = model(node_feat, edge_index)

        metrics = eval_adj_recon(adj_matrix, adj_recon)
        print(f"测试图({num_nodes}节点): AUC={metrics['auc']:.4f}, ACC={metrics['acc']:.4f}")

    # 6. 可视化
    if args.visualize and train_history['loss']:
        print("\n" + "=" * 60)
        print("可视化训练结果")
        print("=" * 60)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(train_history['loss'], 'b-', linewidth=2, label='训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练损失曲线')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=best_loss, color='r', linestyle='--', alpha=0.5,
                    label=f'最佳验证损失: {best_loss:.4f}')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(train_history['auc'], 'g-', linewidth=2, label='训练AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('AUC曲线')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=best_auc, color='r', linestyle='--', alpha=0.5,
                    label=f'最佳验证AUC: {best_auc:.4f}')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(train_history['acc'], 'orange', linewidth=2, label='训练准确率')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('准确率曲线')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.show()

        # 可视化一个测试图的重建结果
        print("\n可视化测试图的重建结果...")
        edge_index, adj_matrix, node_feat, num_nodes = generate_simple_community_graph(
            args.min_nodes, args.max_nodes, args.edge_prob
        )

        with torch.no_grad():
            node_emb, adj_recon = model(node_feat, edge_index)

        # 绘制原始和重建的邻接矩阵
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im1 = axes[0].imshow(adj_matrix.numpy(), cmap='Blues', vmin=0, vmax=1)
        axes[0].set_title('原始邻接矩阵')
        axes[0].set_xlabel('节点')
        axes[0].set_ylabel('节点')
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(adj_recon.numpy(), cmap='Reds', vmin=0, vmax=1)
        axes[1].set_title('重建邻接矩阵')
        axes[1].set_xlabel('节点')
        axes[1].set_ylabel('节点')
        plt.colorbar(im2, ax=axes[1])

        plt.suptitle('最佳模型重建结果', fontsize=14)
        plt.tight_layout()
        plt.show()

    # 7. 保存最终模型
    final_model_path = os.path.join(args.save_dir, 'final_model.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': train_history['loss'][-1] if train_history['loss'] else 0,
        'auc': train_history['auc'][-1] if train_history['auc'] else 0,
        'acc': train_history['acc'][-1] if train_history['acc'] else 0,
        'emb_dim': args.emb_dim,
        'hidden_dim': args.hidden_dim,
        'in_feat_dim': in_feat_dim,
        'train_history': train_history,
        'config': {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'epochs': args.epochs
        }
    }, final_model_path)

    print(f"\n" + "=" * 60)
    print("训练完成!")
    print(f"最终训练损失: {train_history['loss'][-1]:.4f}" if train_history['loss'] else "无训练损失")
    print(f"最终训练AUC: {train_history['auc'][-1]:.4f}" if train_history['auc'] else "无训练AUC")
    print(f"最佳验证损失: {best_loss:.4f}")
    print(f"最佳验证AUC: {best_auc:.4f}")
    print(f"最终模型已保存: {final_model_path}")
    print("=" * 60)


def evaluate_model(model, graphs):
    """评估模型在验证集上的表现"""
    model.eval()
    val_loss = 0.0
    val_auc = 0.0
    val_acc = 0.0
    num_graphs = 0

    with torch.no_grad():
        for graph in graphs[:100]:  # 只评估前100个图
            node_emb, adj_recon = model(graph.x, graph.edge_index)

            # 计算损失
            loss = adj_reconstruction_loss(graph.adj, adj_recon)
            if not torch.isnan(loss):
                val_loss += loss.item()

            # 计算评估指标
            metrics = eval_adj_recon(graph.adj, adj_recon)
            val_auc += metrics['auc']
            val_acc += metrics['acc']
            num_graphs += 1

    model.train()

    if num_graphs > 0:
        return val_loss / num_graphs, val_auc / num_graphs, val_acc / num_graphs
    else:
        return 0.0, 0.0, 0.0


# ===================== 辅助函数 =====================
def adj_reconstruction_loss(adj_gt, adj_recon):
    """稳定的损失函数"""
    # 数值检查
    if torch.isnan(adj_recon).any():
        print("警告: 重建矩阵包含NaN，使用0.5代替")
        adj_recon = torch.nan_to_num(adj_recon, nan=0.5)

    # 限制范围
    adj_recon = torch.clamp(adj_recon, min=1e-8, max=1.0 - 1e-8)

    # 忽略自环
    mask = 1 - torch.eye(adj_gt.shape[0], device=adj_gt.device)
    gt_flat = adj_gt[mask.bool()]
    recon_flat = adj_recon[mask.bool()]

    return F.binary_cross_entropy(recon_flat, gt_flat)


def eval_adj_recon(adj_gt, adj_recon):
    """评估函数"""
    adj_recon = adj_recon.detach()
    adj_gt = adj_gt.detach()
    #print(adj_gt, adj_recon)
    # 数值检查
    adj_recon = torch.nan_to_num(adj_recon, nan=0.5)
    adj_recon = torch.clamp(adj_recon, 0, 1)

    mask = 1 - torch.eye(adj_gt.shape[0], device=adj_gt.device)
    gt = adj_gt[mask.bool()].cpu().numpy()
    recon = adj_recon[mask.bool()].cpu().numpy()
    recon_bin = (recon > 0.5).astype(int)

    if len(np.unique(gt)) <= 1:
        return {'auc': 0.5, 'acc': 1.0 if len(np.unique(gt)) == 1 else 0.5}

    try:
        auc = roc_auc_score(gt, recon)
    except:
        auc = 0.5

    acc = accuracy_score(gt, recon_bin)
    return {'auc': auc, 'acc': acc}


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    stable_train_adj_recon_gae()