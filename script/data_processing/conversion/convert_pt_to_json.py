#!/usr/bin/env python3
"""
将 .pt 格式的评测数据转换为 JSON 格式
"""
import torch
import json
from pathlib import Path


def convert_pt_to_json(pt_file_path, json_file_path=None):
    """
    将 .pt 文件转换为 JSON 文件

    Args:
        pt_file_path: .pt 文件路径
        json_file_path: 输出的 JSON 文件路径，默认为同名的 .json 文件
    """
    # 加载 .pt 文件
    print(f"正在加载文件: {pt_file_path}")
    data = torch.load(pt_file_path, map_location='cpu')

    # 如果未指定输出路径，使用同名的 .json 文件
    if json_file_path is None:
        json_file_path = str(pt_file_path).replace('.pt', '.json')

    # 转换数据为可 JSON 序列化的格式
    def convert_to_serializable(obj):
        """递归转换对象为可 JSON 序列化的格式"""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # 对于其他类型，尝试转换为字符串
            return str(obj)

    print("正在转换数据格式...")
    serializable_data = convert_to_serializable(data)

    # 保存为 JSON 文件
    print(f"正在保存到: {json_file_path}")
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)

    print(f"✓ 转换完成！")
    print(f"  输入文件: {pt_file_path}")
    print(f"  输出文件: {json_file_path}")

    # 打印一些统计信息
    if isinstance(serializable_data, list):
        print(f"  数据条数: {len(serializable_data)}")
    elif isinstance(serializable_data, dict):
        print(f"  字典键数: {len(serializable_data)}")
        if serializable_data:
            print(f"  顶层键名: {list(serializable_data.keys())}")


if __name__ == '__main__':
    pt_file = '/nvme0/work/workspaces-zy/GraphInstruct/data/eval/graphinstruct_eval_19tasks_10samples_v3_text.pt'
    convert_pt_to_json(pt_file)
