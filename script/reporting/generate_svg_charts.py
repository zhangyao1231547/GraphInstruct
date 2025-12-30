#!/usr/bin/env python3
"""
生成GraphInstruct 6模型评测的SVG图表
1. 总体准确率对比图
2. 详细任务性能热力图
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
BASE_DIR = '/nvme0/work/workspaces-zy/GraphInstruct/data/eval'
OUTPUT_DIR = os.path.join(BASE_DIR, 'comprehensive_report_v2')

# 加载数据
with open(os.path.join(OUTPUT_DIR, 'all_models_metrics.json'), 'r') as f:
    all_metrics = json.load(f)

# 模型配置
MODELS = {
    'LLM-LoRA': {'color': '#2ecc71', 'type': 'LoRA微调'},
    'Graph-Stage2': {'color': '#3498db', 'type': '图微调'},
    'Graph-Stage3': {'color': '#9b59b6', 'type': '文+图微调'},
    'Qwen-Max': {'color': '#e74c3c', 'type': 'API云端'},
    'DeepSeek-V3': {'color': '#f39c12', 'type': 'API云端'},
    'Doubao-1.8': {'color': '#1abc9c', 'type': 'API云端'}
}

# 任务列表
ALL_TASKS = [
    'BFS', 'DFS', 'bipartite', 'clustering_coefficient', 'common_neighbor',
    'connected_component', 'connectivity', 'cycle', 'degree', 'diameter',
    'edge', 'jaccard', 'maximum_flow', 'MST', 'neighbor',
    'page_rank', 'predecessor', 'shortest_path', 'topological_sort'
]

def generate_accuracy_svg():
    """生成总体准确率对比SVG图"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # 按准确率排序
    sorted_models = sorted(all_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    models = [m[0] for m in sorted_models]
    accuracies = [m[1]['accuracy'] * 100 for m in sorted_models]
    colors = [MODELS[m]['color'] for m in models]

    # 创建柱状图
    bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5, width=0.7)

    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

    # 添加模型类型标签
    for i, model in enumerate(models):
        ax.annotate(MODELS[model]['type'],
                    xy=(i, -3),
                    ha='center', va='top', fontsize=11, color='gray')

    # 设置样式
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('GraphInstruct 6-Model Overall Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # 设置x轴标签
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')

    # 添加水平参考线
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=60, color='orange', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()

    # 保存为SVG
    svg_path = os.path.join(OUTPUT_DIR, 'accuracy_comparison.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved: {svg_path}")


def generate_task_heatmap_svg():
    """生成详细任务性能热力图SVG"""
    # 按准确率排序模型
    sorted_models = sorted(all_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    models = [m[0] for m in sorted_models]

    # 构建数据矩阵
    data = []
    for task in ALL_TASKS:
        row = []
        for model in models:
            task_data = all_metrics[model].get('by_task', {}).get(task, {})
            acc = task_data.get('accuracy', 0) * 100
            row.append(acc)
        data.append(row)

    data = np.array(data)

    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 14))

    # 创建热力图
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # 设置刻度
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(ALL_TASKS)))
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax.set_yticklabels(ALL_TASKS, fontsize=11)

    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # 添加数值标签
    for i in range(len(ALL_TASKS)):
        for j in range(len(models)):
            value = data[i, j]
            color = 'white' if value < 50 else 'black'
            fontweight = 'bold' if value >= 90 or value == 0 else 'normal'
            ax.text(j, i, f'{value:.0f}', ha="center", va="center",
                   color=color, fontsize=10, fontweight=fontweight)

    # 设置标题
    ax.set_title('Task-wise Accuracy Heatmap (%)\nGraphInstruct 19 Tasks × 6 Models',
                 fontsize=16, fontweight='bold', pad=20)

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va="bottom", fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # 添加网格线
    ax.set_xticks(np.arange(-.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ALL_TASKS), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

    # 添加模型类型标注
    for i, model in enumerate(models):
        ax.annotate(MODELS[model]['type'],
                    xy=(i, len(ALL_TASKS) + 0.5),
                    ha='center', va='top', fontsize=9, color='gray')

    plt.tight_layout()

    # 保存为SVG
    svg_path = os.path.join(OUTPUT_DIR, 'task_heatmap.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved: {svg_path}")


def main():
    print("Generating SVG charts...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # 显示模型数据概览
    print("Model accuracies:")
    for model, metrics in sorted(all_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"  {model}: {metrics['accuracy']*100:.2f}%")
    print()

    # 生成图表
    generate_accuracy_svg()
    generate_task_heatmap_svg()

    print("\nDone! SVG charts generated successfully.")


if __name__ == '__main__':
    main()
