import torch
import numpy as np
import json
import pickle
from model import GraphAutoencoder, VariationalGraphAutoencoder
from utils import load_model
import argparse


def encode_and_save():
    """编码图数据并保存嵌入"""
    parser = argparse.ArgumentParser(description='编码图数据')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--data_path', type=str, help='数据路径（PyG格式）')
    parser.add_argument('--output_dir', type=str, default='embeddings', help='输出目录')
    parser.add_argument('--use_cpu', action='store_true', help='使用CPU')
    args = parser.parse_args()

    # 1. 设置设备
    device = torch.device('cpu' if args.use_cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"使用设备: {device}")

    # 2. 加载模型
    print(f"加载模型: {args.model_path}")

    # 模型参数需要根据保存时的配置设置
    # 这里假设已知参数，实际使用时可以从配置文件中读取
    model_kwargs = {
        'in_channels': 1433,  # Cora数据集特征维度
        'hidden_channels': 128,
        'embedding_dim': 64
    }

    try:
        model, checkpoint = load_model(args.model_path, **model_kwargs)
        model = model.to(device)
        model.eval()
        print(f"模型加载成功，训练轮数: {checkpoint.get('epoch', '未知')}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 3. 加载数据
    if args.data_path:
        # 从文件加载数据
        data = torch.load(args.data_path)
    else:
        # 示例：使用Cora数据集
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root='data/Cora', name='Cora')
        data = dataset[0]

    x = data.x if hasattr(data, 'x') else torch.eye(data.num_nodes)
    edge_index = data.edge_index

    # 移动到设备
    x = x.to(device)
    edge_index = edge_index.to(device)

    # 4. 编码图数据
    print(f"编码 {x.shape[0]} 个节点...")
    with torch.no_grad():
        if hasattr(model, 'encode'):
            embeddings = model.encode(x, edge_index)
        else:
            embeddings, _ = model(x, edge_index)

    # 5. 保存嵌入
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存为numpy格式
    embeddings_np = embeddings.cpu().numpy()
    np.save(f'{args.output_dir}/embeddings.npy', embeddings_np)

    # 保存为文本格式（可读）
    np.savetxt(f'{args.output_dir}/embeddings.txt', embeddings_np, fmt='%.6f')

    # 保存元数据
    metadata = {
        'num_nodes': embeddings.shape[0],
        'embedding_dim': embeddings.shape[1],
        'model_type': model.__class__.__name__,
        'model_path': args.model_path
    }

    with open(f'{args.output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"嵌入已保存到 {args.output_dir}/")
    print(f"嵌入形状: {embeddings.shape}")

    return embeddings


def load_embeddings(embedding_dir):
    """加载保存的嵌入"""
    import os

    # 检查文件是否存在
    npy_path = f'{embedding_dir}/embeddings.npy'
    metadata_path = f'{embedding_dir}/metadata.json'

    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"嵌入文件不存在: {npy_path}")

    # 加载嵌入
    embeddings = np.load(npy_path)

    # 加载元数据
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    print(f"加载嵌入: 形状={embeddings.shape}")
    print(f"元数据: {metadata}")

    return embeddings, metadata


def reconstruct_from_embeddings(model_path, embeddings, threshold=0.5):
    """从嵌入重建图结构"""
    # 加载模型（仅解码器部分）
    model_kwargs = {
        'in_channels': 1433,
        'hidden_channels': 128,
        'embedding_dim': embeddings.shape[1]
    }

    model, _ = load_model(model_path, **model_kwargs)
    model.eval()

    # 使用解码器重建邻接矩阵
    with torch.no_grad():
        embeddings_tensor = torch.FloatTensor(embeddings)
        adj_recon = model.decoder(embeddings_tensor)

    # 应用阈值得到二值邻接矩阵
    adj_binary = (adj_recon > threshold).float()

    return adj_recon.numpy(), adj_binary.numpy()


if __name__ == '__main__':
    # 示例用法
    import sys

    if len(sys.argv) > 1:
        # 编码模式
        encode_and_save()
    else:
        # 交互式菜单
        print("图自编码器工具")
        print("1. 编码图数据")
        print("2. 加载嵌入")

        choice = input("请选择操作 [1/2]: ")

        if choice == '1':
            model_path = input("模型路径 (默认: gae_model.pth): ") or "gae_model.pth"
            output_dir = input("输出目录 (默认: embeddings): ") or "embeddings"

            encode_and_save()
        elif choice == '2':
            embedding_dir = input("嵌入目录 (默认: embeddings): ") or "embeddings"
            embeddings, metadata = load_embeddings(embedding_dir)

            # 示例：显示前5个节点的嵌入
            print("\n前5个节点的嵌入:")
            for i in range(min(5, embeddings.shape[0])):
                print(f"节点 {i}: {embeddings[i, :5]}...")
        else:
            print("无效选择")