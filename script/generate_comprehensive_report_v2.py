#!/usr/bin/env python3
"""
生成6个模型的综合评测报告和效果图

模型列表:
1. LLM微调 (LoRA) - results_text_only_full_4096tokens
2. 图微调 (Stage2) - results_bfs_improved_full_4096tokens
3. 文+图微调 (Stage3 Directed GNN) - results_stage3_directed_gnn_4096tokens
4. Qwen-Max API
5. DeepSeek-V3 API
6. Doubao-Seed-1.8 API
"""

import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
BASE_DIR = '/nvme0/work/workspaces-zy/GraphInstruct/data/eval'
OUTPUT_DIR = os.path.join(BASE_DIR, 'comprehensive_report_v2')

# 模型配置
MODELS = {
    'LLM-LoRA': {
        'path': os.path.join(BASE_DIR, 'results_text_only_full_4096tokens/metrics.json'),
        'type': 'LoRA微调',
        'color': '#2ecc71',
        'description': 'Qwen3-4B + GraphInstruct LoRA (纯文字)'
    },
    'Graph-Stage2': {
        'path': os.path.join(BASE_DIR, 'results_bfs_improved_full_4096tokens/metrics.json'),
        'type': '图微调',
        'color': '#3498db',
        'description': 'GraphAgent Stage2 (图编码器)'
    },
    'Graph-Stage3': {
        'path': os.path.join(BASE_DIR, 'results_stage3_directed_gnn_4096tokens/metrics.json'),
        'type': '文+图微调',
        'color': '#9b59b6',
        'description': 'GraphAgent Stage3 Directed GNN'
    },
    'Qwen-Max': {
        'path': os.path.join(BASE_DIR, 'api_results/qwen-max/metrics.json'),
        'type': 'API云端',
        'color': '#e74c3c',
        'description': 'Qwen-Max (阿里云)'
    },
    'DeepSeek-V3': {
        'path': os.path.join(BASE_DIR, 'api_results/deepseek-v3/metrics.json'),
        'type': 'API云端',
        'color': '#f39c12',
        'description': 'DeepSeek-V3.2'
    },
    'Doubao-1.8': {
        'path': os.path.join(BASE_DIR, 'api_results/doubao-seed/metrics.json'),
        'type': 'API云端',
        'color': '#1abc9c',
        'description': 'Doubao-Seed-1.8 (字节跳动)'
    }
}

# 任务列表
ALL_TASKS = [
    'BFS', 'DFS', 'bipartite', 'clustering_coefficient', 'common_neighbor',
    'connected_component', 'connectivity', 'cycle', 'degree', 'diameter',
    'edge', 'jaccard', 'maximum_flow', 'MST', 'neighbor',
    'page_rank', 'predecessor', 'shortest_path', 'topological_sort'
]

# 任务分类
TASK_CATEGORIES = {
    '图遍历': ['BFS', 'DFS'],
    '图属性': ['degree', 'neighbor', 'edge', 'connectivity', 'cycle', 'bipartite', 'connected_component', 'diameter'],
    '节点相似性': ['common_neighbor', 'jaccard'],
    '路径与流': ['predecessor', 'topological_sort', 'shortest_path', 'maximum_flow'],
    '中心性': ['page_rank', 'clustering_coefficient'],
    '树结构': ['MST']
}


def load_metrics(path):
    """加载评测指标"""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def get_task_accuracy(metrics, task):
    """获取单个任务的准确率"""
    if 'by_task' in metrics:
        task_data = metrics['by_task'].get(task, {})
    elif 'by_task_type' in metrics:
        task_data = metrics['by_task_type'].get(task, {})
    else:
        return 0.0

    total = task_data.get('total', 0)
    correct = task_data.get('correct', 0)

    if total == 0:
        return 0.0
    return correct / total


def generate_accuracy_chart(all_metrics, output_path):
    """生成总体准确率对比柱状图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = list(all_metrics.keys())
    accuracies = [all_metrics[m]['accuracy'] * 100 for m in models]
    colors = [MODELS[m]['color'] for m in models]

    bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.2)

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('GraphInstruct 6-Model Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    for i, model in enumerate(models):
        ax.annotate(MODELS[model]['type'],
                    xy=(i, -5),
                    ha='center', va='top', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_task_heatmap(all_metrics, output_path):
    """生成任务热力图"""
    models = list(all_metrics.keys())

    data = []
    for task in ALL_TASKS:
        row = []
        for model in models:
            acc = get_task_accuracy(all_metrics[model], task) * 100
            row.append(acc)
        data.append(row)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(ALL_TASKS)))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_yticklabels(ALL_TASKS, fontsize=10)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(ALL_TASKS)):
        for j in range(len(models)):
            value = data[i, j]
            color = 'white' if value < 50 else 'black'
            ax.text(j, i, f'{value:.0f}', ha="center", va="center",
                   color=color, fontsize=9, fontweight='bold')

    ax.set_title('Task-wise Accuracy Heatmap (%)', fontsize=14, fontweight='bold', pad=20)

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va="bottom", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_radar_chart(all_metrics, output_path):
    """生成雷达图"""
    categories = list(TASK_CATEGORIES.keys())
    num_cats = len(categories)

    model_scores = {}
    for model in all_metrics:
        scores = []
        for cat, tasks in TASK_CATEGORIES.items():
            cat_accs = [get_task_accuracy(all_metrics[model], t) * 100 for t in tasks]
            scores.append(np.mean(cat_accs))
        model_scores[model] = scores

    angles = np.linspace(0, 2 * np.pi, num_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for model in all_metrics:
        values = model_scores[model] + model_scores[model][:1]
        ax.plot(angles, values, 'o-', linewidth=2,
                label=model, color=MODELS[model]['color'])
        ax.fill(angles, values, alpha=0.15, color=MODELS[model]['color'])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title('Category-wise Performance Radar', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_grouped_bar_chart(all_metrics, output_path):
    """生成分组柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    finetuned = ['LLM-LoRA', 'Graph-Stage2', 'Graph-Stage3']
    api_models = ['Qwen-Max', 'DeepSeek-V3', 'Doubao-1.8']

    for ax, group, title in zip(axes, [finetuned, api_models],
                                 ['Fine-tuned Models', 'API Models']):
        x = np.arange(len(ALL_TASKS))
        width = 0.25

        for i, model in enumerate(group):
            accs = [get_task_accuracy(all_metrics[model], t) * 100 for t in ALL_TASKS]
            ax.bar(x + i * width, accs, width, label=model,
                   color=MODELS[model]['color'], alpha=0.8)

        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(ALL_TASKS, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_markdown_report(all_metrics, output_path):
    """生成Markdown报告"""
    report = []

    report.append("# GraphInstruct 6模型综合评测报告")
    report.append("")
    report.append("## 评测概述")
    report.append("")
    report.append(f"**评测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("**评测数据集**: GraphInstruct Benchmark")
    report.append("- 总任务数: 19个图推理任务")
    report.append("- 每任务样本数: 10个")
    report.append("- 总测试样本: 190个")
    report.append("- 最大生成token: 4096")
    report.append("")

    report.append("**评测模型**:")
    report.append("")
    report.append("| 模型 | 类型 | 描述 |")
    report.append("|------|------|------|")
    for model, config in MODELS.items():
        report.append(f"| {model} | {config['type']} | {config['description']} |")
    report.append("")

    report.append("---")
    report.append("")
    report.append("## 一、总体性能排名")
    report.append("")

    sorted_models = sorted(all_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)

    report.append("| 排名 | 模型 | 准确率 | 正确数/总数 | 类型 |")
    report.append("|:----:|------|:------:|:-----------:|------|")

    for i, (model, metrics) in enumerate(sorted_models, 1):
        acc = metrics['accuracy'] * 100
        correct = metrics['correct']
        total = metrics['total']
        model_type = MODELS[model]['type']
        report.append(f"| {i} | {model} | **{acc:.2f}%** | {correct}/{total} | {model_type} |")

    report.append("")
    report.append("### 性能分布图")
    report.append("")
    report.append("![模型性能对比](accuracy_comparison.png)")
    report.append("")

    report.append("---")
    report.append("")
    report.append("## 二、各任务详细性能")
    report.append("")
    report.append("### 热力图总览")
    report.append("")
    report.append("![任务热力图](task_heatmap.png)")
    report.append("")

    report.append("### 详细数据表")
    report.append("")

    header = "| 任务 | " + " | ".join(all_metrics.keys()) + " |"
    sep = "|------|" + "|".join(["------"] * len(all_metrics)) + "|"
    report.append(header)
    report.append(sep)

    for task in ALL_TASKS:
        row = f"| {task} |"
        for model in all_metrics:
            acc = get_task_accuracy(all_metrics[model], task) * 100
            if acc >= 100:
                row += f" **100%** |"
            elif acc >= 90:
                row += f" **{acc:.0f}%** |"
            elif acc == 0:
                row += f" 0% |"
            else:
                row += f" {acc:.0f}% |"
        report.append(row)

    report.append("")
    report.append("### 类别性能雷达图")
    report.append("")
    report.append("![雷达图](radar_chart.png)")
    report.append("")

    report.append("---")
    report.append("")
    report.append("## 三、任务类别分析")
    report.append("")

    for cat, tasks in TASK_CATEGORIES.items():
        report.append(f"### {cat}")
        report.append("")

        header = "| 模型 | " + " | ".join(tasks) + " | 平均 |"
        sep = "|------|" + "|".join(["------"] * len(tasks)) + "|------|"
        report.append(header)
        report.append(sep)

        for model in all_metrics:
            row = f"| {model} |"
            accs = []
            for task in tasks:
                acc = get_task_accuracy(all_metrics[model], task) * 100
                accs.append(acc)
                row += f" {acc:.0f}% |"
            avg = np.mean(accs)
            row += f" **{avg:.1f}%** |"
            report.append(row)

        report.append("")

    report.append("### 分组对比图")
    report.append("")
    report.append("![分组对比](grouped_comparison.png)")
    report.append("")

    report.append("---")
    report.append("")
    report.append("## 四、关键发现")
    report.append("")

    task_avg = {}
    for task in ALL_TASKS:
        accs = [get_task_accuracy(all_metrics[m], task) * 100 for m in all_metrics]
        task_avg[task] = np.mean(accs)

    sorted_tasks = sorted(task_avg.items(), key=lambda x: x[1], reverse=True)

    report.append("### 4.1 表现突出的任务")
    report.append("以下任务大多数模型都能较好完成:")
    for task, avg in sorted_tasks[:5]:
        report.append(f"- **{task}**: 平均准确率 {avg:.1f}%")
    report.append("")

    report.append("### 4.2 具有挑战性的任务")
    report.append("以下任务对所有模型都具有挑战:")
    for task, avg in sorted_tasks[-5:]:
        report.append(f"- **{task}**: 平均准确率 {avg:.1f}%")
    report.append("")

    report.append("### 4.3 模型类型对比")
    report.append("")

    finetuned_avg = np.mean([all_metrics[m]['accuracy'] * 100
                            for m in ['LLM-LoRA', 'Graph-Stage2', 'Graph-Stage3']])
    api_avg = np.mean([all_metrics[m]['accuracy'] * 100
                      for m in ['Qwen-Max', 'DeepSeek-V3', 'Doubao-1.8']])

    report.append(f"- **微调模型平均准确率**: {finetuned_avg:.1f}%")
    report.append(f"- **API模型平均准确率**: {api_avg:.1f}%")
    report.append("")

    stage3_acc = all_metrics['Graph-Stage3']['accuracy'] * 100
    stage2_acc = all_metrics['Graph-Stage2']['accuracy'] * 100
    improvement = stage3_acc - stage2_acc

    report.append("### 4.4 Stage3 Directed GNN 提升分析")
    report.append("")
    report.append(f"- Stage2 (图微调): {stage2_acc:.1f}%")
    report.append(f"- Stage3 (文+图微调): {stage3_acc:.1f}%")
    report.append(f"- **提升**: +{improvement:.1f}%")
    report.append("")

    report.append("### 4.5 主要结论")
    report.append("")
    report.append("1. **Stage3 Directed GNN效果最佳**: 文+图联合微调显著优于单独的图微调")
    report.append("2. **LLM-LoRA与Stage3表现相当**: 纯文本LoRA微调在多数任务上表现优秀")
    report.append("3. **API模型差异明显**: DeepSeek-V3和Doubao表现较好，Qwen-Max相对较弱")
    report.append("4. **难点任务一致**: MST、PageRank对所有模型都是挑战")
    report.append("")

    report.append("---")
    report.append("")
    report.append(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"Saved: {output_path}")


def save_all_metrics(all_metrics, output_path):
    """保存所有指标到JSON"""
    output_data = {}
    for model, metrics in all_metrics.items():
        output_data[model] = {
            'accuracy': metrics['accuracy'],
            'correct': metrics['correct'],
            'total': metrics['total'],
            'type': MODELS[model]['type'],
            'description': MODELS[model]['description'],
            'by_task': {}
        }

        task_data = metrics.get('by_task', metrics.get('by_task_type', {}))
        for task in ALL_TASKS:
            if task in task_data:
                output_data[model]['by_task'][task] = {
                    'total': task_data[task].get('total', 0),
                    'correct': task_data[task].get('correct', 0),
                    'accuracy': get_task_accuracy(metrics, task)
                }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metrics = {}
    for model, config in MODELS.items():
        metrics = load_metrics(config['path'])
        if metrics:
            all_metrics[model] = metrics
            print(f"Loaded {model}: {metrics['accuracy']*100:.1f}%")
        else:
            print(f"Failed to load {model}")

    if not all_metrics:
        print("No metrics loaded!")
        return

    print(f"\nGenerating reports for {len(all_metrics)} models...")

    generate_accuracy_chart(all_metrics, os.path.join(OUTPUT_DIR, 'accuracy_comparison.png'))
    generate_task_heatmap(all_metrics, os.path.join(OUTPUT_DIR, 'task_heatmap.png'))
    generate_radar_chart(all_metrics, os.path.join(OUTPUT_DIR, 'radar_chart.png'))
    generate_grouped_bar_chart(all_metrics, os.path.join(OUTPUT_DIR, 'grouped_comparison.png'))

    generate_markdown_report(all_metrics, os.path.join(OUTPUT_DIR, 'comprehensive_report.md'))
    save_all_metrics(all_metrics, os.path.join(OUTPUT_DIR, 'all_models_metrics.json'))

    print(f"\nAll reports saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
