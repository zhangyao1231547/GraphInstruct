import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import os
import torch.nn.functional as F

from model import  GraphAutoencoder

def dense_adj_from_edge_index(edge_index, num_nodes):
    """将稀疏边索引转换为稠密邻接矩阵"""
    adj = torch.zeros((num_nodes, num_nodes))
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj


def reconstruction_loss(adj_original, adj_recon, edge_index=None):
    """计算重建损失（二元交叉熵）"""

    loss = F.binary_cross_entropy(adj_recon, adj_original)
    return loss


def kl_loss(mu, logvar):
    """计算KL散度损失"""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.numel()


def evaluate_reconstruction(adj_original, adj_recon):
    """评估重建质量"""
    adj_original_np = adj_original.cpu().numpy().flatten()
    adj_recon_np = adj_recon.detach().cpu().numpy().flatten()

    # 计算AUC-ROC和AP
    auc = roc_auc_score(adj_original_np, adj_recon_np)
    ap = average_precision_score(adj_original_np, adj_recon_np)

    # 计算准确率（阈值0.5）
    adj_pred = (adj_recon_np > 0.5).astype(float)
    accuracy = np.mean(adj_pred == adj_original_np)

    return {
        'auc': auc,
        'ap': ap,
        'accuracy': accuracy
    }


def save_model(model, path, epoch, optimizer=None):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_type': model.__class__.__name__,
        'embedding_dim': model.embedding_dim,
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, path)
    print(f"模型已保存到 {path}")


def load_model(path, model_class=None, **model_kwargs):
    """加载模型检查点"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")

    checkpoint = torch.load(path, map_location='cpu')

    # 确定模型类型
    if model_class is None:
        model_type = checkpoint.get('model_type', 'GraphAutoencoder')
        if 'VGAE' in model_type or 'Variational' in model_type:
            model_class = GraphAutoencoder

    # 创建模型实例
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, checkpoint