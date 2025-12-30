#!/usr/bin/env python3
"""
修复评测数据中的错误答案

问题：
1. DEGREE: 3/10 样本答案与图结构不符
2. NEIGHBOR: 5/10 样本邻居列表不完整
3. CLUSTERING_COEFFICIENT: 2/10 样本系数计算错误

修复方法：根据实际图结构重新计算正确答案
"""

import torch
import re
import ast
from collections import defaultdict
from transformers import AutoTokenizer
import copy
import sys
import os

# 添加 GraphAgent 训练代码路径
sys.path.insert(0, "/nvme0/work/workspaces-zy/GraphAgent-zy/GraphAGent-training")

try:
    from model.graph_action_agent import conversation as conversation_lib
except ImportError:
    conversation_lib = None

IGNORE_TOKEN_ID = -100


def get_adjacency_from_graph_data(graph_data):
    """从graph_data构建邻接表"""
    adj = defaultdict(set)
    in_edges = defaultdict(set)  # 入边(用于有向图的predecessor)

    if hasattr(graph_data, 'edge_index_dict'):
        for key, edge_index in graph_data.edge_index_dict.items():
            for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                adj[src].add(dst)
                adj[dst].add(src)  # 无向图
                in_edges[dst].add(src)  # 有向图入边
    return adj, in_edges


def compute_correct_degree(graph_data, target_node):
    """计算正确的度数"""
    adj, _ = get_adjacency_from_graph_data(graph_data)
    return len(adj[target_node])


def compute_correct_neighbors(graph_data, target_node):
    """计算正确的邻居列表"""
    adj, _ = get_adjacency_from_graph_data(graph_data)
    return sorted(list(adj[target_node]))


def compute_correct_clustering_coefficient(graph_data, target_node):
    """计算正确的聚类系数"""
    adj, _ = get_adjacency_from_graph_data(graph_data)
    neighbors = list(adj[target_node])
    k = len(neighbors)

    if k < 2:
        return 0.0

    # Count edges between neighbors
    edge_count = 0
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            if neighbors[j] in adj[neighbors[i]]:
                edge_count += 1

    # For undirected: CC = 2*edges / (k*(k-1))
    cc = 2 * edge_count / (k * (k - 1))
    return round(cc, 4)


def extract_target_node_from_answer(answer_text, task_type):
    """从答案文本中提取目标节点"""
    if task_type == 'degree':
        # "the degree of node X is"
        match = re.search(r'degree of node (\d+)', answer_text)
        if match:
            return int(match.group(1))
    elif task_type == 'neighbor':
        # "neighbors of node X are"
        match = re.search(r'neighbors of node (\d+)', answer_text)
        if match:
            return int(match.group(1))
    elif task_type == 'clustering_coefficient':
        # "Node X's neighbors"
        match = re.search(r"[Nn]ode (\d+)'s neighbors", answer_text)
        if match:
            return int(match.group(1))
    return None


def fix_answer_text(original_answer, task_type, correct_answer):
    """修复答案文本中<<<>>>内的答案"""
    if task_type == 'degree':
        # 替换 <<<数字>>>
        return re.sub(r'<<<\d+>>>', f'<<<{correct_answer}>>>', original_answer)
    elif task_type == 'neighbor':
        # 替换 <<<[列表]>>>
        return re.sub(r'<<<\[[\d,\s]+\]>>>', f'<<<{correct_answer}>>>', original_answer)
    elif task_type == 'clustering_coefficient':
        # 替换 <<<小数>>>
        return re.sub(r'<<<[\d.]+>>>', f'<<<{correct_answer}>>>', original_answer)
    return original_answer


def retokenize_sample(tokenizer, sample, new_answer_text):
    """重新tokenize样本"""
    # 解码原始input_ids获取完整对话
    original_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)

    # 找到assistant回复的位置并替换
    # Qwen格式: <|im_start|>assistant\n{answer}<|im_end|>
    pattern = r'(<\|im_start\|>assistant\n)(.*?)(<\|im_end\|>)'

    def replace_answer(match):
        return match.group(1) + new_answer_text + match.group(3)

    new_text = re.sub(pattern, replace_answer, original_text, flags=re.DOTALL)

    # 重新tokenize
    new_input_ids = tokenizer(
        new_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=4096
    ).input_ids[0]

    # 重新计算labels (mask掉instruction部分)
    new_labels = new_input_ids.clone()

    # 找到assistant回复开始的位置
    im_start_assistant = "<|im_start|>assistant\n"
    im_start_assistant_ids = tokenizer(im_start_assistant, add_special_tokens=False).input_ids

    # 遍历找到assistant开始位置
    assistant_start = -1
    for i in range(len(new_input_ids) - len(im_start_assistant_ids) + 1):
        if new_input_ids[i:i+len(im_start_assistant_ids)].tolist() == im_start_assistant_ids:
            assistant_start = i + len(im_start_assistant_ids)
            break

    # mask instruction部分
    if assistant_start > 0:
        new_labels[:assistant_start] = IGNORE_TOKEN_ID

    return new_input_ids, new_labels


def fix_evaluation_data(input_path, output_path, tokenizer_path):
    """修复评测数据"""
    print(f"Loading evaluation data from {input_path}...")
    data = torch.load(input_path, weights_only=False)
    print(f"Loaded {len(data)} samples")

    # 加载tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    eos_token = "<|im_end|>"
    eos_id = tokenizer.convert_tokens_to_ids(eos_token)
    if eos_id is None or eos_id == tokenizer.unk_token_id:
        eos_token = "<|endoftext|>"
    tokenizer.pad_token = eos_token

    fixed_count = {'degree': 0, 'neighbor': 0, 'clustering_coefficient': 0}
    error_details = []

    for i, sample in enumerate(data):
        task_type = sample.get('task_type', '')

        if task_type not in ['degree', 'neighbor', 'clustering_coefficient']:
            continue

        # 解码当前答案
        labels = sample['labels']
        answer_tokens = [t.item() for t in labels if t != -100]
        original_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

        # 提取目标节点
        target_node = extract_target_node_from_answer(original_answer, task_type)
        if target_node is None:
            print(f"  Warning: Cannot extract target node from sample {i} ({task_type})")
            continue

        # 计算正确答案
        graph_data = sample['graph_data']

        if task_type == 'degree':
            correct_value = compute_correct_degree(graph_data, target_node)
            # 提取原始答案
            match = re.search(r'<<<(\d+)>>>', original_answer)
            if match:
                original_value = int(match.group(1))
                if original_value != correct_value:
                    print(f"  Fixing DEGREE sample {i}: node {target_node}: {original_value} -> {correct_value}")
                    new_answer = fix_answer_text(original_answer, task_type, correct_value)
                    new_input_ids, new_labels = retokenize_sample(tokenizer, sample, new_answer)
                    data[i]['input_ids'] = new_input_ids
                    data[i]['labels'] = new_labels
                    fixed_count['degree'] += 1
                    error_details.append({
                        'sample_idx': i,
                        'task_type': task_type,
                        'target_node': target_node,
                        'original': original_value,
                        'corrected': correct_value
                    })

        elif task_type == 'neighbor':
            correct_neighbors = compute_correct_neighbors(graph_data, target_node)
            match = re.search(r'<<<(\[[\d,\s]+\])>>>', original_answer)
            if match:
                try:
                    original_neighbors = sorted(ast.literal_eval(match.group(1)))
                    if original_neighbors != correct_neighbors:
                        print(f"  Fixing NEIGHBOR sample {i}: node {target_node}: {original_neighbors} -> {correct_neighbors}")
                        new_answer = fix_answer_text(original_answer, task_type, str(correct_neighbors))
                        new_input_ids, new_labels = retokenize_sample(tokenizer, sample, new_answer)
                        data[i]['input_ids'] = new_input_ids
                        data[i]['labels'] = new_labels
                        fixed_count['neighbor'] += 1
                        error_details.append({
                            'sample_idx': i,
                            'task_type': task_type,
                            'target_node': target_node,
                            'original': original_neighbors,
                            'corrected': correct_neighbors
                        })
                except:
                    pass

        elif task_type == 'clustering_coefficient':
            correct_cc = compute_correct_clustering_coefficient(graph_data, target_node)
            match = re.search(r'<<<([\d.]+)>>>', original_answer)
            if match:
                original_cc = float(match.group(1))
                if abs(original_cc - correct_cc) > 0.01:
                    print(f"  Fixing CLUSTERING_COEFFICIENT sample {i}: node {target_node}: {original_cc} -> {correct_cc}")
                    new_answer = fix_answer_text(original_answer, task_type, correct_cc)
                    new_input_ids, new_labels = retokenize_sample(tokenizer, sample, new_answer)
                    data[i]['input_ids'] = new_input_ids
                    data[i]['labels'] = new_labels
                    fixed_count['clustering_coefficient'] += 1
                    error_details.append({
                        'sample_idx': i,
                        'task_type': task_type,
                        'target_node': target_node,
                        'original': original_cc,
                        'corrected': correct_cc
                    })

    # 保存修复后的数据
    print(f"\nSaving fixed data to {output_path}...")
    torch.save(data, output_path)

    # 打印统计
    print(f"\n{'='*60}")
    print("Fix Statistics:")
    print(f"  DEGREE: {fixed_count['degree']} samples fixed")
    print(f"  NEIGHBOR: {fixed_count['neighbor']} samples fixed")
    print(f"  CLUSTERING_COEFFICIENT: {fixed_count['clustering_coefficient']} samples fixed")
    print(f"  Total: {sum(fixed_count.values())} samples fixed")
    print(f"{'='*60}")

    return data, error_details


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='/nvme0/work/workspaces-zy/GraphInstruct/data/eval/graphinstruct_eval_19tasks_10samples.pt')
    parser.add_argument('--output', type=str,
                        default='/nvme0/work/workspaces-zy/GraphInstruct/data/eval/graphinstruct_eval_19tasks_10samples_fixed.pt')
    parser.add_argument('--tokenizer', type=str,
                        default='/nvme0/work/workspaces-zy/model/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507')
    args = parser.parse_args()

    fix_evaluation_data(args.input, args.output, args.tokenizer)
