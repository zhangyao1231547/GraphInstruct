import torch
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.data import Data
import json
import os
import sys
import math

# 导入你的模型定义
sys.path.append('.')  # 确保可以导入当前目录的模块


# 定义相同的模型结构（必须与训练时相同）
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


class StableAdjDecoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1))
        self.bias = torch.nn.Parameter(torch.tensor(-1.0))

    def forward(self, node_emb):
        if torch.isnan(node_emb).any():
            node_emb = torch.nan_to_num(node_emb, nan=0.0)

        norms = torch.norm(node_emb, dim=1, keepdim=True)
        safe_norms = torch.clamp(norms, min=1e-8)
        normalized_emb = node_emb / safe_norms
        similarity = torch.mm(normalized_emb, normalized_emb.t())
        adj_recon = self.scale * similarity + self.bias
        adj_recon = torch.sigmoid(adj_recon)

        if torch.isnan(adj_recon).any():
            adj_recon = torch.nan_to_num(adj_recon, nan=0.5)
        return adj_recon


class StableGraphAutoencoder(torch.nn.Module):
    def __init__(self, in_feat_dim, hidden_dim, emb_dim):
        super().__init__()
        self.encoder = StableNodeEncoder(in_feat_dim, hidden_dim, emb_dim)
        self.decoder = StableAdjDecoder(emb_dim)

    def forward(self, x, edge_index):
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        node_emb = self.encoder(x, edge_index)
        adj_recon = self.decoder(node_emb)
        return node_emb, adj_recon


# 导入GCNConv
from torch_geometric.nn import GCNConv


# ===================== 辅助函数 =====================
def sinusoidal_pe(adj_matrix, feature_dim=32):
    """正弦位置编码（与训练时相同）"""
    num_nodes = adj_matrix.shape[0]
    position = torch.arange(num_nodes, dtype=torch.float32, device=adj_matrix.device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, feature_dim, 2, dtype=torch.float32, device=adj_matrix.device)
        * -(math.log(10000.0) / feature_dim)
    )

    pe = torch.zeros(num_nodes, feature_dim, device=adj_matrix.device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term) if feature_dim % 2 == 0 else torch.cos(position * div_term[:-1])
    return pe


def process_sample(sample, feature_dim=32, device='cpu'):
    """
    从样本数据加载图数据
    """
    graph_info = sample.get('graph', {})
    edge_list = graph_info.get('edge_index', [[], []])

    if not edge_list or len(edge_list[0]) == 0:
        num_nodes = graph_info.get('num_nodes', 1)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    else:
        if isinstance(edge_list, list):
            edge_index = torch.tensor(edge_list, dtype=torch.long)
        elif isinstance(edge_list, torch.Tensor):
            edge_index = edge_list.long()
        else:
            raise ValueError(f"无法识别的edge_index类型: {type(edge_list)}")

        if edge_index.dim() == 2 and edge_index.shape[0] != 2:
            if edge_index.shape[1] == 2:
                edge_index = edge_index.t()
            else:
                raise ValueError(f"edge_index形状异常: {edge_index.shape}")

        if 'num_nodes' in graph_info:
            num_nodes = graph_info['num_nodes']
        else:
            num_nodes = edge_index.max().item() + 1

        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        adj_matrix[edge_index[0], edge_index[1]] = 1.0

        if graph_info.get('undirected', True):
            adj_matrix = adj_matrix + adj_matrix.t()
            adj_matrix = (adj_matrix > 0).float()

    node_feat = sinusoidal_pe(adj_matrix, feature_dim=feature_dim)

    return Data(
        x=node_feat.to(device),
        edge_index=edge_index.to(device),
        adj=adj_matrix.to(device),
        num_nodes=num_nodes
    )


def load_validation_data(val_path, max_samples=1000, device='cpu'):
    """加载验证集数据"""
    val_graphs = []
    with open(val_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    for sample in tqdm(data, desc="加载验证集"):
        try:
            graph_data = process_sample(sample, feature_dim=32, device=device)
            val_graphs.append(graph_data)
        except Exception as e:
            print(f"处理样本时出错: {e}")
            continue

    print(f"成功加载 {len(val_graphs)} 个验证图")
    return val_graphs


def adj_reconstruction_loss(adj_gt, adj_recon):
    """稳定的损失函数"""
    if torch.isnan(adj_recon).any():
        adj_recon = torch.nan_to_num(adj_recon, nan=0.5)
    adj_recon = torch.clamp(adj_recon, min=1e-8, max=1.0 - 1e-8)
    mask = 1 - torch.eye(adj_gt.shape[0], device=adj_gt.device)
    gt_flat = adj_gt[mask.bool()]
    recon_flat = adj_recon[mask.bool()]
    return F.binary_cross_entropy(recon_flat, gt_flat)


def eval_adj_recon(adj_gt, adj_recon):
    """评估函数"""
    adj_recon = adj_recon.detach()
    adj_gt = adj_gt.detach()
    adj_recon = torch.nan_to_num(adj_recon, nan=0.5)
    adj_recon = torch.clamp(adj_recon, 0, 1)

    mask = 1 - torch.eye(adj_gt.shape[0], device=adj_gt.device)
    gt = adj_gt[mask.bool()].cpu().numpy()
    recon = adj_recon[mask.bool()].cpu().numpy()
    recon_bin = (recon > 0.5).astype(int)

    if len(np.unique(gt)) <= 1:
        return {
            'auc': 0.5,
            'acc': 1.0 if len(np.unique(gt)) == 1 else 0.5,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

    try:
        auc = roc_auc_score(gt, recon)
    except:
        auc = 0.5

    acc = accuracy_score(gt, recon_bin)

    try:
        precision = precision_score(gt, recon_bin, zero_division=0)
        recall = recall_score(gt, recon_bin, zero_division=0)
        f1 = f1_score(gt, recon_bin, zero_division=0)
    except:
        precision = recall = f1 = 0.0

    return {
        'auc': auc,
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ===================== 模型加载和验证函数 =====================
def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """从检查点加载模型"""
    print(f"正在加载模型: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 安全地打印检查点信息
    print(f"检查点信息:")

    # 获取各个值，如果是字符串则不格式化
    epoch = checkpoint.get('epoch', '未知')
    loss_val = checkpoint.get('loss', '未知')
    auc_val = checkpoint.get('auc', '未知')
    acc_val = checkpoint.get('acc', '未知')

    # 安全的格式化输出
    if isinstance(epoch, (int, float)):
        print(f"  训练轮数: {epoch}")
    else:
        print(f"  训练轮数: {epoch}")

    if isinstance(loss_val, (int, float)):
        print(f"  损失: {loss_val:.4f}")
    else:
        print(f"  损失: {loss_val}")

    if isinstance(auc_val, (int, float)):
        print(f"  AUC: {auc_val:.4f}")
    else:
        print(f"  AUC: {auc_val}")

    if isinstance(acc_val, (int, float)):
        print(f"  准确率: {acc_val:.4f}")
    else:
        print(f"  准确率: {acc_val}")

    # 其他信息
    print(f"  嵌入维度: {checkpoint.get('emb_dim', '未知')}")
    print(f"  隐藏维度: {checkpoint.get('hidden_dim', '未知')}")
    print(f"  输入特征维度: {checkpoint.get('in_feat_dim', '未知')}")

    # 获取模型配置
    emb_dim = checkpoint.get('emb_dim', 768)
    hidden_dim = checkpoint.get('hidden_dim', 32)
    in_feat_dim = checkpoint.get('in_feat_dim', 32)

    # 创建模型实例
    model = StableGraphAutoencoder(
        in_feat_dim=in_feat_dim,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim
    ).to(device)

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"模型加载成功!")
    print(f"  总参数: {sum(p.numel() for p in model.parameters()):,}")

    return model, checkpoint


def validate_model(model, val_graphs, device='cpu', batch_size=1):
    """在验证集上评估模型"""
    print(f"\n开始验证，验证集大小: {len(val_graphs)}")

    model.eval()

    all_metrics = {
        'loss': [],
        'auc': [],
        'acc': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    # 按节点数分组统计
    node_size_stats = {}

    with torch.no_grad():
        for i, graph in enumerate(tqdm(val_graphs, desc="验证进度")):
            # 确保数据在正确的设备上
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            adj_gt = graph.adj.to(device)

            # 前向传播
            node_emb, adj_recon = model(x, edge_index)

            # 计算损失
            loss = adj_reconstruction_loss(adj_gt, adj_recon)
            print(edge_index)
            print("--------adj_gt---------")
            print(adj_gt)
            print("--------adj_recon---------")
            print(torch.round(adj_recon * 100) / 100)
            # 计算评估指标
            metrics = eval_adj_recon(adj_gt, adj_recon)

            # 收集所有指标
            all_metrics['loss'].append(loss.item())
            all_metrics['auc'].append(metrics['auc'])
            all_metrics['acc'].append(metrics['acc'])
            all_metrics['precision'].append(metrics['precision'])
            all_metrics['recall'].append(metrics['recall'])
            all_metrics['f1'].append(metrics['f1'])

            # 按节点数统计
            num_nodes = graph.num_nodes
            if num_nodes not in node_size_stats:
                node_size_stats[num_nodes] = {
                    'count': 0,
                    'loss': [],
                    'auc': [],
                    'acc': []
                }

            node_size_stats[num_nodes]['count'] += 1
            node_size_stats[num_nodes]['loss'].append(loss.item())
            node_size_stats[num_nodes]['auc'].append(metrics['auc'])
            node_size_stats[num_nodes]['acc'].append(metrics['acc'])

            # 每100个样本打印一次进度
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1}/{len(val_graphs)} 个样本")

    # 计算平均指标
    avg_metrics = {}
    for key in all_metrics:
        if all_metrics[key]:
            avg_metrics[key] = np.mean(all_metrics[key])
        else:
            avg_metrics[key] = 0.0

    return avg_metrics, all_metrics, node_size_stats


def visualize_results(avg_metrics, node_size_stats, checkpoint_path):
    """可视化验证结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 总体指标条形图
    metrics_names = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']
    metrics_values = [
        avg_metrics['auc'],
        avg_metrics['acc'],
        avg_metrics['precision'],
        avg_metrics['recall'],
        avg_metrics['f1']
    ]

    bars = axes[0, 0].bar(metrics_names, metrics_values, color=['blue', 'green', 'orange', 'red', 'purple'])
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].set_title('总体评估指标')
    axes[0, 0].set_ylabel('分数')
    axes[0, 0].grid(True, alpha=0.3)

    # 在条形上添加数值标签
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                        f'{value:.3f}', ha='center', va='bottom')

    # 2. 损失分布直方图
    axes[0, 1].hist(avg_metrics.get('all_losses', [0]), bins=20, alpha=0.7, color='red')
    axes[0, 1].axvline(x=avg_metrics['loss'], color='black', linestyle='--',
                       label=f'平均损失: {avg_metrics["loss"]:.4f}')
    axes[0, 1].set_xlabel('损失值')
    axes[0, 1].set_ylabel('频数')
    axes[0, 1].set_title('损失值分布')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. AUC分布直方图
    axes[0, 2].hist(avg_metrics.get('all_auc', [0]), bins=20, alpha=0.7, color='blue')
    axes[0, 2].axvline(x=avg_metrics['auc'], color='black', linestyle='--', label=f'平均AUC: {avg_metrics["auc"]:.4f}')
    axes[0, 2].set_xlabel('AUC值')
    axes[0, 2].set_ylabel('频数')
    axes[0, 2].set_title('AUC分布')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. 按节点数统计的AUC
    if node_size_stats:
        node_sizes = sorted(node_size_stats.keys())
        auc_by_size = [np.mean(node_size_stats[size]['auc']) for size in node_sizes]
        count_by_size = [node_size_stats[size]['count'] for size in node_sizes]

        axes[1, 0].bar(node_sizes, auc_by_size, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('节点数')
        axes[1, 0].set_ylabel('平均AUC')
        axes[1, 0].set_title('不同节点数的AUC表现')
        axes[1, 0].grid(True, alpha=0.3)

        # 在柱子上方添加样本数
        for i, (size, count) in enumerate(zip(node_sizes, count_by_size)):
            axes[1, 0].text(size, auc_by_size[i] + 0.02, f'n={count}', ha='center', va='bottom', fontsize=8)

    # 5. 按节点数统计的准确率
    if node_size_stats:
        acc_by_size = [np.mean(node_size_stats[size]['acc']) for size in node_sizes]

        axes[1, 1].bar(node_sizes, acc_by_size, alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('节点数')
        axes[1, 1].set_ylabel('平均准确率')
        axes[1, 1].set_title('不同节点数的准确率表现')
        axes[1, 1].grid(True, alpha=0.3)

        for i, (size, count) in enumerate(zip(node_sizes, count_by_size)):
            axes[1, 1].text(size, acc_by_size[i] + 0.02, f'n={count}', ha='center', va='bottom', fontsize=8)

    # 6. 模型信息
    checkpoint_name = os.path.basename(checkpoint_path)
    model_info = f"""
    模型检查点: {checkpoint_name}
    验证样本数: {sum(node_size_stats[size]['count'] for size in node_size_stats)}
    平均损失: {avg_metrics['loss']:.4f}
    平均AUC: {avg_metrics['auc']:.4f}
    平均准确率: {avg_metrics['acc']:.4f}
    平均F1分数: {avg_metrics['f1']:.4f}
    """

    axes[1, 2].text(0.1, 0.5, model_info, fontsize=10, verticalalignment='center')
    axes[1, 2].set_title('模型验证摘要')
    axes[1, 2].axis('off')

    plt.suptitle('图自编码器验证结果', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def save_validation_results(avg_metrics, node_size_stats, checkpoint_path, save_dir='validation_results'):
    """保存验证结果到文件"""
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_name = os.path.basename(checkpoint_path).replace('.pth', '')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(save_dir, f'validation_{checkpoint_name}_{timestamp}.json')

    # 准备结果数据
    results = {
        'checkpoint': checkpoint_path,
        'timestamp': timestamp,
        'average_metrics': avg_metrics,
        'node_size_statistics': node_size_stats,
        'summary': {
            'total_samples': sum(node_size_stats[size]['count'] for size in node_size_stats),
            'unique_node_sizes': len(node_size_stats),
            'best_metric': max(avg_metrics['auc'], avg_metrics['acc'], avg_metrics['f1'])
        }
    }

    # 保存为JSON
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n验证结果已保存到: {result_file}")

    # 同时保存为txt格式便于阅读
    txt_file = os.path.join(save_dir, f'validation_{checkpoint_name}_{timestamp}.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("图自编码器验证结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"检查点文件: {checkpoint_path}\n")
        f.write(f"验证时间: {timestamp}\n")
        f.write(f"验证样本总数: {results['summary']['total_samples']}\n\n")

        f.write("总体评估指标:\n")
        f.write("-" * 40 + "\n")
        for key, value in avg_metrics.items():
            f.write(f"{key:15s}: {value:.4f}\n")

        f.write("\n按节点数统计:\n")
        f.write("-" * 40 + "\n")
        for size in sorted(node_size_stats.keys()):
            stats = node_size_stats[size]
            f.write(f"节点数 {size:3d}: {stats['count']:3d} 个样本 | "
                    f"损失: {np.mean(stats['loss']):.4f} | "
                    f"AUC: {np.mean(stats['auc']):.4f} | "
                    f"准确率: {np.mean(stats['acc']):.4f}\n")

    print(f"文本报告已保存到: {txt_file}")

    return result_file, txt_file


# ===================== 主函数 =====================
def main():
    import datetime

    parser = argparse.ArgumentParser(description='图自编码器验证')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--val_path', type=str,
                        default='/home/test/workspaces-zy/GraphInstruct/data/converted/common_neighbor_converted.json',
                        help='验证集路径')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='最大验证样本数')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批次大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='计算设备')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='是否可视化结果')
    parser.add_argument('--save_results', action='store_true', default=True,
                        help='是否保存验证结果')
    parser.add_argument('--output_dir', type=str, default='validation_results',
                        help='结果保存目录')

    args = parser.parse_args()

    print("=" * 60)
    print("图自编码器验证脚本")
    print("=" * 60)
    print(f"设备: {args.device}")
    print(f"检查点: {args.checkpoint}")
    print(f"验证集: {args.val_path}")
    print(f"最大样本数: {args.max_samples}")
    print("=" * 60)

    # 1. 加载验证数据
    print("\n[1/3] 加载验证数据...")
    val_graphs = load_validation_data(args.val_path, args.max_samples, args.device)

    if not val_graphs:
        print("错误: 没有加载到验证数据!")
        return

    # 2. 加载模型
    print("\n[2/3] 加载模型...")
    try:
        model, checkpoint_info = load_model_from_checkpoint(args.checkpoint, args.device)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 3. 验证模型
    print("\n[3/3] 验证模型...")
    avg_metrics, all_metrics, node_size_stats = validate_model(
        model, val_graphs, args.device, args.batch_size
    )

    # 保存所有指标用于可视化
    avg_metrics['all_losses'] = all_metrics['loss']
    avg_metrics['all_auc'] = all_metrics['auc']
    avg_metrics['all_acc'] = all_metrics['acc']

    # 4. 打印结果
    print("\n" + "=" * 60)
    print("验证结果摘要")
    print("=" * 60)
    print(f"验证样本总数: {len(val_graphs)}")
    print(f"平均损失: {avg_metrics['loss']:.4f}")
    print(f"平均AUC: {avg_metrics['auc']:.4f}")
    print(f"平均准确率: {avg_metrics['acc']:.4f}")
    print(f"平均精确率: {avg_metrics['precision']:.4f}")
    print(f"平均召回率: {avg_metrics['recall']:.4f}")
    print(f"平均F1分数: {avg_metrics['f1']:.4f}")

    print("\n按节点数统计:")
    print("-" * 40)
    for size in sorted(node_size_stats.keys()):
        stats = node_size_stats[size]
        print(f"节点数 {size:3d}: {stats['count']:3d} 个样本 | "
              f"损失: {np.mean(stats['loss']):.4f} | "
              f"AUC: {np.mean(stats['auc']):.4f} | "
              f"准确率: {np.mean(stats['acc']):.4f}")

    # # 5. 可视化
    # if args.visualize:
    #     print("\n生成可视化图表...")
    #     visualize_results(avg_metrics, node_size_stats, args.checkpoint)
    #
    # # 6. 保存结果
    # if args.save_results:
    #     print("\n保存验证结果...")
    #     save_validation_results(avg_metrics, node_size_stats, args.checkpoint, args.output_dir)
    #
    # # 7. 示例可视化（展示几个具体的重建结果）
    # print("\n示例重建可视化...")
    # visualize_examples(model, val_graphs[:3], args.device)

    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)


def visualize_examples(model, example_graphs, device='cpu'):
    """可视化几个具体的重建示例"""
    model.eval()

    fig, axes = plt.subplots(len(example_graphs), 2, figsize=(12, 4 * len(example_graphs)))

    if len(example_graphs) == 1:
        axes = [axes]

    with torch.no_grad():
        for idx, graph in enumerate(example_graphs):
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            adj_gt = graph.adj.to(device)

            node_emb, adj_recon = model(x, edge_index)
            metrics = eval_adj_recon(adj_gt, adj_recon)

            # 原始邻接矩阵
            im1 = axes[idx][0].imshow(adj_gt.cpu().numpy(), cmap='Blues', vmin=0, vmax=1)
            axes[idx][0].set_title(f'示例 {idx + 1}: 原始邻接矩阵\n节点数: {graph.num_nodes}')
            axes[idx][0].set_xlabel('节点')
            axes[idx][0].set_ylabel('节点')

            # 重建邻接矩阵
            im2 = axes[idx][1].imshow(adj_recon.cpu().numpy(), cmap='Reds', vmin=0, vmax=1)
            title = f'重建邻接矩阵\nAUC: {metrics["auc"]:.3f}, ACC: {metrics["acc"]:.3f}'
            axes[idx][1].set_title(title)
            axes[idx][1].set_xlabel('节点')
            axes[idx][1].set_ylabel('节点')

            # 添加颜色条
            plt.colorbar(im1, ax=axes[idx][0], fraction=0.046, pad=0.04)
            plt.colorbar(im2, ax=axes[idx][1], fraction=0.046, pad=0.04)

    plt.suptitle('邻接矩阵重建示例', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()