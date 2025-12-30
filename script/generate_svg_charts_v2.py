#!/usr/bin/env python3
"""
生成与comprehensive_report完全一致样式的SVG图表
1. 总体准确率对比图 - 横向条形图
2. 详细任务热力图
"""

import json
import os
from datetime import datetime

# 数据路径
BASE_DIR = '/nvme0/work/workspaces-zy/GraphInstruct/data/eval'
OUTPUT_DIR = os.path.join(BASE_DIR, 'comprehensive_report_v2')

# 加载数据
with open(os.path.join(OUTPUT_DIR, 'all_models_metrics.json'), 'r') as f:
    all_metrics = json.load(f)

# 模型配置 - 按准确率排序
MODELS_CONFIG = [
    ('DeepSeek-V3', '#f39c12', 'API云端'),
    ('LLM-LoRA', '#2ecc71', 'LoRA微调'),
    ('Graph-Stage3', '#9b59b6', '文+图微调'),
    ('Doubao-1.8', '#1abc9c', 'API云端'),
    ('Graph-Stage2', '#3498db', '图微调'),
    ('Qwen-Max', '#e74c3c', 'API云端'),
]

# 任务列表
ALL_TASKS = [
    'BFS', 'DFS', 'MST', 'bipartite', 'clustering_coefficient',
    'common_neighbor', 'connected_component', 'connectivity', 'cycle',
    'degree', 'diameter', 'edge', 'jaccard', 'maximum_flow',
    'neighbor', 'page_rank', 'predecessor', 'shortest_path', 'topological_sort'
]


def accuracy_to_color(acc):
    """根据准确率返回颜色 (红-黄-绿渐变)"""
    if acc <= 50:
        # 红到黄
        r = 244
        g = int(67 + (235 - 67) * (acc / 50))
        b = int(54 + (59 - 54) * (acc / 50))
    else:
        # 黄到绿
        ratio = (acc - 50) / 50
        r = int(255 - (255 - 76) * ratio)
        g = int(235 - (235 - 175) * ratio)
        b = int(59 + (80 - 59) * ratio)
    return f"rgb({r},{g},{b})"


def generate_accuracy_svg():
    """生成总体准确率对比SVG图 - 横向条形图样式"""

    # 按准确率排序
    sorted_models = sorted(all_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)

    width = 1200
    height = 800
    margin_left = 180
    margin_right = 50
    margin_top = 80
    margin_bottom = 100
    bar_height = 50
    bar_gap = 33.75

    chart_width = width - margin_left - margin_right

    svg = []
    svg.append('<?xml version="1.0" encoding="UTF-8"?>')
    svg.append(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">')

    # 定义
    svg.append('  <defs>')
    svg.append('    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="0%" y2="100%">')
    svg.append('      <stop offset="0%" style="stop-color:#f8f9fa;stop-opacity:1" />')
    svg.append('      <stop offset="100%" style="stop-color:#e9ecef;stop-opacity:1" />')
    svg.append('    </linearGradient>')
    svg.append('    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">')
    svg.append('      <feDropShadow dx="2" dy="2" stdDeviation="2" flood-opacity="0.3"/>')
    svg.append('    </filter>')
    svg.append('  </defs>')

    # 背景
    svg.append(f'  <rect width="{width}" height="{height}" fill="url(#bgGrad)"/>')

    # 标题
    svg.append(f'  <text x="{width/2}" y="40" text-anchor="middle" font-family="Source Han Sans SC, sans-serif" font-size="24" font-weight="bold" fill="#333">')
    svg.append('    GraphInstruct Benchmark - 6模型评测结果对比')
    svg.append('  </text>')
    svg.append(f'  <text x="{width/2}" y="65" text-anchor="middle" font-family="Source Han Sans SC, sans-serif" font-size="14" fill="#666">')
    svg.append(f'    19个图推理任务 × 10个样本 = 190个测试用例 | 评测时间: {datetime.now().strftime("%Y-%m-%d")}')
    svg.append('  </text>')

    # 网格线
    for i in range(11):
        x = margin_left + (chart_width * i / 10)
        svg.append(f'  <line x1="{x}" y1="{margin_top}" x2="{x}" y2="{height - margin_bottom}" stroke="#ddd" stroke-dasharray="3,3"/>')
        svg.append(f'  <text x="{x}" y="{height - margin_bottom + 20}" text-anchor="middle" font-family="Source Han Sans SC, sans-serif" font-size="12" fill="#666">{i*10}%</text>')

    # 模型颜色映射
    model_colors = {
        'DeepSeek-V3': '#f39c12',
        'LLM-LoRA': '#4CAF50',
        'Graph-Stage3': '#9C27B0',
        'Doubao-1.8': '#1abc9c',
        'Graph-Stage2': '#2196F3',
        'Qwen-Max': '#e74c3c',
    }

    # 绘制条形
    for i, (model, metrics) in enumerate(sorted_models):
        acc = metrics['accuracy'] * 100
        correct = metrics['correct']
        total = metrics['total']

        y = margin_top + i * (bar_height + bar_gap) + bar_gap / 2
        bar_width = chart_width * (acc / 100)
        color = model_colors.get(model, '#666')

        # 条形
        svg.append(f'  <rect x="{margin_left}" y="{y}" width="{bar_width}" height="{bar_height}"')
        svg.append(f'        fill="{color}" rx="4" ry="4" filter="url(#shadow)"/>')

        # 模型名称
        svg.append(f'  <text x="{margin_left - 10}" y="{y + bar_height/2 + 5}" text-anchor="end"')
        svg.append(f'        font-family="Source Han Sans SC, sans-serif" font-size="13" fill="#333">{model}</text>')

        # 准确率
        svg.append(f'  <text x="{margin_left + bar_width + 10}" y="{y + bar_height/2 + 5}" text-anchor="start"')
        svg.append(f'        font-family="Source Han Sans SC, sans-serif" font-size="14" font-weight="bold" fill="{color}">{acc:.2f}%</text>')

        # 正确数/总数
        svg.append(f'  <text x="{margin_left + bar_width + 70}" y="{y + bar_height/2 + 5}" text-anchor="start"')
        svg.append(f'        font-family="Source Han Sans SC, sans-serif" font-size="11" fill="#888">({correct}/{total})</text>')

    svg.append('</svg>')

    output_path = os.path.join(OUTPUT_DIR, 'comprehensive_accuracy_chart.svg')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Saved: {output_path}")


def generate_task_heatmap_svg():
    """生成详细任务热力图SVG"""

    # 按准确率排序模型
    sorted_models = sorted(all_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    models = [m[0] for m in sorted_models]

    width = 1400
    height = 600
    margin_left = 200
    margin_right = 50
    margin_top = 120
    margin_bottom = 50

    num_tasks = len(ALL_TASKS)
    num_models = len(models)

    cell_width = (width - margin_left - margin_right) / num_tasks
    cell_height = (height - margin_top - margin_bottom) / num_models

    svg = []
    svg.append('<?xml version="1.0" encoding="UTF-8"?>')
    svg.append(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">')

    # 定义渐变
    svg.append('  <defs>')
    svg.append('    <linearGradient id="heatGrad" x1="0%" y1="0%" x2="100%" y2="0%">')
    svg.append('      <stop offset="0%" style="stop-color:#f44336;stop-opacity:1" />')
    svg.append('      <stop offset="50%" style="stop-color:#ffeb3b;stop-opacity:1" />')
    svg.append('      <stop offset="100%" style="stop-color:#4caf50;stop-opacity:1" />')
    svg.append('    </linearGradient>')
    svg.append('  </defs>')

    # 背景
    svg.append(f'  <rect width="{width}" height="{height}" fill="#fafafa"/>')

    # 标题
    svg.append(f'  <text x="{width/2}" y="35" text-anchor="middle" font-family="Source Han Sans SC, sans-serif" font-size="22" font-weight="bold" fill="#333">')
    svg.append('    各任务准确率热力图 (6模型对比)')
    svg.append('  </text>')
    svg.append(f'  <text x="{width/2}" y="55" text-anchor="middle" font-family="Source Han Sans SC, sans-serif" font-size="12" fill="#666">')
    svg.append('    颜色深浅表示准确率: 红色(0%) → 黄色(50%) → 绿色(100%)')
    svg.append('  </text>')

    # 图例
    legend_x = width / 2 - 100
    svg.append(f'  <rect x="{legend_x}" y="70" width="200" height="15" fill="url(#heatGrad)"/>')
    svg.append(f'  <text x="{legend_x}" y="98" text-anchor="middle" font-family="Source Han Sans SC, sans-serif" font-size="10" fill="#666">0%</text>')
    svg.append(f'  <text x="{legend_x + 100}" y="98" text-anchor="middle" font-family="Source Han Sans SC, sans-serif" font-size="10" fill="#666">50%</text>')
    svg.append(f'  <text x="{legend_x + 200}" y="98" text-anchor="middle" font-family="Source Han Sans SC, sans-serif" font-size="10" fill="#666">100%</text>')

    # 任务标签（斜着显示）
    for j, task in enumerate(ALL_TASKS):
        x = margin_left + j * cell_width + cell_width / 2
        svg.append(f'  <text x="{x}" y="110" text-anchor="start" font-family="Source Han Sans SC, sans-serif" font-size="10" fill="#333" transform="rotate(-45 {x} 110)">{task}</text>')

    # 绘制热力图
    for i, model in enumerate(models):
        y = margin_top + i * cell_height

        # 模型名称
        svg.append(f'  <text x="{margin_left - 10}" y="{y + cell_height/2 + 4}" text-anchor="end" font-family="Source Han Sans SC, sans-serif" font-size="12" fill="#333">{model}</text>')

        for j, task in enumerate(ALL_TASKS):
            x = margin_left + j * cell_width

            # 获取准确率
            task_data = all_metrics[model].get('by_task', {}).get(task, {})
            acc = task_data.get('accuracy', 0) * 100

            color = accuracy_to_color(acc)

            # 单元格
            svg.append(f'  <rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" fill="{color}" stroke="#fff" stroke-width="1"/>')

            # 数值
            text_color = '#fff' if acc < 60 or acc > 80 else '#333'
            svg.append(f'  <text x="{x + cell_width/2}" y="{y + cell_height/2 + 4}" text-anchor="middle" font-family="Source Han Sans SC, sans-serif" font-size="9" fill="{text_color}" font-weight="bold">{int(acc)}</text>')

    svg.append('</svg>')

    output_path = os.path.join(OUTPUT_DIR, 'task_heatmap_v2.svg')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Saved: {output_path}")


def main():
    print("Generating SVG charts (same style as comprehensive_report)...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # 显示模型数据
    print("Model accuracies:")
    for model, metrics in sorted(all_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"  {model}: {metrics['accuracy']*100:.2f}% ({metrics['correct']}/{metrics['total']})")
    print()

    generate_accuracy_svg()
    generate_task_heatmap_svg()

    print("\nDone!")


if __name__ == '__main__':
    main()
