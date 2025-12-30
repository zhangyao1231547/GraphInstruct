#!/usr/bin/env python3
"""
综合评测报告生成脚本
生成包含所有模型评测结果的Markdown报告和SVG可视化图表
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

# 定义19个任务的分类
TASK_CATEGORIES = {
    "图遍历 (Graph Traversal)": ["BFS", "DFS"],
    "图属性 (Graph Properties)": ["degree", "neighbor", "edge", "connectivity", "cycle", "bipartite", "connected_component", "diameter"],
    "节点相似性 (Node Similarity)": ["common_neighbor", "jaccard"],
    "路径与流 (Path & Flow)": ["predecessor", "topological_sort", "shortest_path", "maximum_flow"],
    "中心性 (Centrality)": ["page_rank", "clustering_coefficient"],
    "树结构 (Tree Structure)": ["MST"]
}

# 所有任务列表
ALL_TASKS = [
    "BFS", "DFS", "MST", "bipartite", "clustering_coefficient",
    "common_neighbor", "connected_component", "connectivity", "cycle",
    "degree", "diameter", "edge", "jaccard", "maximum_flow",
    "neighbor", "page_rank", "predecessor", "shortest_path", "topological_sort"
]


def load_metrics(results_dir: str) -> Dict:
    """加载评测结果metrics.json"""
    metrics_path = os.path.join(results_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def generate_svg_chart(models_data: List[Dict], output_path: str):
    """生成SVG柱状图"""
    # 图表尺寸
    width = 1200
    height = 800
    margin_left = 180
    margin_right = 50
    margin_top = 80
    margin_bottom = 100
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    # 排序模型（按准确率降序）
    models_data = sorted(models_data, key=lambda x: x['accuracy'], reverse=True)

    # 颜色方案
    colors = [
        "#4CAF50",  # 绿色 - LoRA EN
        "#8BC34A",  # 浅绿 - LoRA ZH
        "#2196F3",  # 蓝色 - 豆包
        "#03A9F4",  # 浅蓝 - DeepSeek
        "#FF9800",  # 橙色 - Qwen-Max
        "#9C27B0",  # 紫色 - Graph EN
        "#E91E63",  # 粉色 - Graph ZH
        "#607D8B",  # 灰色 - Baseline
    ]

    bar_height = min(50, (chart_height - 20) / len(models_data) * 0.7)
    gap = (chart_height - bar_height * len(models_data)) / (len(models_data) + 1)

    # 字体配置
    font_family = "Source Han Sans SC, sans-serif"

    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#f8f9fa;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#e9ecef;stop-opacity:1" />
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="2" dy="2" stdDeviation="2" flood-opacity="0.3"/>
    </filter>
  </defs>

  <!-- 背景 -->
  <rect width="{width}" height="{height}" fill="url(#bgGrad)"/>

  <!-- 标题 -->
  <text x="{width/2}" y="40" text-anchor="middle" font-family="{font_family}" font-size="24" font-weight="bold" fill="#333">
    GraphInstruct Benchmark - 模型评测结果对比
  </text>
  <text x="{width/2}" y="65" text-anchor="middle" font-family="{font_family}" font-size="14" fill="#666">
    19个图推理任务 × 10个样本 = 190个测试用例 | 评测时间: {datetime.now().strftime('%Y-%m-%d')}
  </text>

  <!-- 网格线 -->
'''

    # 添加网格线
    for i in range(11):
        x = margin_left + (chart_width * i / 10)
        svg_content += f'  <line x1="{x}" y1="{margin_top}" x2="{x}" y2="{height - margin_bottom}" stroke="#ddd" stroke-dasharray="3,3"/>\n'
        svg_content += f'  <text x="{x}" y="{height - margin_bottom + 20}" text-anchor="middle" font-family="{font_family}" font-size="12" fill="#666">{i*10}%</text>\n'

    # 绘制柱状图
    for i, model in enumerate(models_data):
        y = margin_top + gap * (i + 1) + bar_height * i
        bar_width = (model['accuracy'] * chart_width)
        color = colors[i % len(colors)]

        # 柱状条
        svg_content += f'''
  <!-- {model['name']} -->
  <rect x="{margin_left}" y="{y}" width="{bar_width}" height="{bar_height}"
        fill="{color}" rx="4" ry="4" filter="url(#shadow)"/>
  <text x="{margin_left - 10}" y="{y + bar_height/2 + 5}" text-anchor="end"
        font-family="{font_family}" font-size="13" fill="#333">{model['name']}</text>
  <text x="{margin_left + bar_width + 10}" y="{y + bar_height/2 + 5}" text-anchor="start"
        font-family="{font_family}" font-size="14" font-weight="bold" fill="{color}">{model['accuracy']*100:.1f}%</text>
  <text x="{margin_left + bar_width + 60}" y="{y + bar_height/2 + 5}" text-anchor="start"
        font-family="{font_family}" font-size="11" fill="#888">({model['correct']}/{model['total']})</text>
'''

    svg_content += '</svg>'

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)


def generate_task_heatmap_svg(models_data: List[Dict], output_path: str):
    """生成任务性能热力图SVG"""
    width = 1400
    height = 900
    margin_left = 200
    margin_top = 120
    margin_right = 120
    margin_bottom = 50

    cell_width = (width - margin_left - margin_right) / len(ALL_TASKS)
    cell_height = (height - margin_top - margin_bottom) / len(models_data)

    # 字体配置
    font_family = "Source Han Sans SC, sans-serif"

    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="heatGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#f44336;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#ffeb3b;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#4caf50;stop-opacity:1" />
    </linearGradient>
  </defs>

  <rect width="{width}" height="{height}" fill="#fafafa"/>

  <!-- 标题 -->
  <text x="{width/2}" y="35" text-anchor="middle" font-family="{font_family}" font-size="22" font-weight="bold" fill="#333">
    各任务准确率热力图
  </text>
  <text x="{width/2}" y="55" text-anchor="middle" font-family="{font_family}" font-size="12" fill="#666">
    颜色深浅表示准确率: 红色(0%) → 黄色(50%) → 绿色(100%)
  </text>

  <!-- 图例 -->
  <rect x="{width/2-100}" y="70" width="200" height="15" fill="url(#heatGrad)"/>
  <text x="{width/2-100}" y="98" text-anchor="middle" font-family="{font_family}" font-size="10" fill="#666">0%</text>
  <text x="{width/2}" y="98" text-anchor="middle" font-family="{font_family}" font-size="10" fill="#666">50%</text>
  <text x="{width/2+100}" y="98" text-anchor="middle" font-family="{font_family}" font-size="10" fill="#666">100%</text>

  <!-- 任务标签（斜着显示）-->
'''

    # 任务标签
    for j, task in enumerate(ALL_TASKS):
        x = margin_left + j * cell_width + cell_width / 2
        svg_content += f'  <text x="{x}" y="{margin_top - 10}" text-anchor="start" font-family="{font_family}" font-size="10" fill="#333" transform="rotate(-45 {x} {margin_top - 10})">{task}</text>\n'

    # 绘制热力图单元格
    for i, model in enumerate(models_data):
        y = margin_top + i * cell_height

        # 模型名称
        svg_content += f'  <text x="{margin_left - 10}" y="{y + cell_height/2 + 5}" text-anchor="end" font-family="{font_family}" font-size="12" fill="#333">{model["name"]}</text>\n'

        by_task = model.get('by_task', {})
        for j, task in enumerate(ALL_TASKS):
            x = margin_left + j * cell_width
            task_data = by_task.get(task, {'total': 10, 'correct': 0})
            acc = task_data['correct'] / task_data['total'] if task_data['total'] > 0 else 0

            # 颜色计算 (红 -> 黄 -> 绿)
            if acc < 0.5:
                r = 244
                g = int(67 + acc * 2 * (235 - 67))
                b = int(54 + acc * 2 * (59 - 54))
            else:
                r = int(255 - (acc - 0.5) * 2 * (255 - 76))
                g = int(235 - (acc - 0.5) * 2 * (235 - 175))
                b = int(59 + (acc - 0.5) * 2 * (80 - 59))

            color = f"rgb({r},{g},{b})"

            svg_content += f'  <rect x="{x+1}" y="{y+1}" width="{cell_width-2}" height="{cell_height-2}" fill="{color}" stroke="#fff" stroke-width="1"/>\n'
            svg_content += f'  <text x="{x + cell_width/2}" y="{y + cell_height/2 + 4}" text-anchor="middle" font-family="{font_family}" font-size="9" fill="#fff" font-weight="bold">{int(acc*100)}</text>\n'

    svg_content += '</svg>'

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)


def generate_markdown_report(models_data: List[Dict], output_path: str):
    """生成详细的Markdown报告"""

    # 按准确率排序
    models_sorted = sorted(models_data, key=lambda x: x['accuracy'], reverse=True)

    report = f"""# GraphInstruct 综合评测报告

## 评测概述

**评测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**评测数据集**: GraphInstruct Benchmark
- 总任务数: 19个图推理任务
- 每任务样本数: 10个
- 总测试样本: 190个

**评测任务分类**:
"""

    for category, tasks in TASK_CATEGORIES.items():
        report += f"- **{category}**: {', '.join(tasks)}\n"

    report += """
---

## 一、总体性能排名

| 排名 | 模型 | 准确率 | 正确数/总数 | 类型 |
|:----:|------|:------:|:-----------:|------|
"""

    for i, model in enumerate(models_sorted, 1):
        acc_pct = f"{model['accuracy']*100:.2f}%"
        model_type = model.get('type', 'Unknown')
        report += f"| {i} | {model['name']} | **{acc_pct}** | {model['correct']}/{model['total']} | {model_type} |\n"

    report += """
### 性能分布图

![模型性能对比](comprehensive_accuracy_chart.svg)

---

## 二、各任务详细性能

### 热力图总览

![任务热力图](task_heatmap.svg)

### 详细数据表

| 任务 | """ + " | ".join([m['short_name'] for m in models_sorted]) + """ |
|------|""" + "|".join(["------" for _ in models_sorted]) + """|
"""

    for task in ALL_TASKS:
        row = f"| {task} |"
        for model in models_sorted:
            by_task = model.get('by_task', {})
            task_data = by_task.get(task, {'total': 10, 'correct': 0})
            acc = task_data['correct'] / task_data['total'] * 100 if task_data['total'] > 0 else 0
            if acc == 100:
                row += f" **100%** |"
            elif acc == 0:
                row += f" 0% |"
            else:
                row += f" {acc:.0f}% |"
        report += row + "\n"

    report += """
---

## 三、任务类别分析

"""

    for category, tasks in TASK_CATEGORIES.items():
        report += f"### {category}\n\n"
        report += "| 模型 | " + " | ".join(tasks) + " | 平均 |\n"
        report += "|------|" + "|".join(["------" for _ in tasks]) + "|------|\n"

        for model in models_sorted:
            by_task = model.get('by_task', {})
            row = f"| {model['short_name']} |"
            total_acc = 0
            for task in tasks:
                task_data = by_task.get(task, {'total': 10, 'correct': 0})
                acc = task_data['correct'] / task_data['total'] * 100 if task_data['total'] > 0 else 0
                total_acc += acc
                row += f" {acc:.0f}% |"
            avg_acc = total_acc / len(tasks)
            row += f" **{avg_acc:.1f}%** |"
            report += row + "\n"
        report += "\n"

    report += """---

## 四、模型对比分析

### 4.1 LoRA微调模型 vs 基础模型
"""

    # 找到LoRA和基础模型进行对比
    lora_en = next((m for m in models_data if 'LoRA' in m['name'] and 'EN' in m['name']), None)
    baseline = next((m for m in models_data if 'Baseline' in m['name'] or 'Text-only' in m['name']), None)

    if lora_en and baseline:
        improvement = (lora_en['accuracy'] - baseline['accuracy']) * 100
        report += f"""
LoRA微调在英文数据上相比基础模型提升了 **{improvement:.1f}%** 的准确率。

- 基础模型 (Qwen3-4B): {baseline['accuracy']*100:.2f}%
- LoRA微调后: {lora_en['accuracy']*100:.2f}%
"""

    report += """
### 4.2 英文 vs 中文性能

| 模型 | 英文准确率 | 中文准确率 | 差异 |
|------|:----------:|:----------:|:----:|
"""

    # 找到英文/中文对
    lora_zh = next((m for m in models_data if 'LoRA' in m['name'] and 'ZH' in m['name']), None)
    graph_en = next((m for m in models_data if 'Graph' in m['name'] and 'EN' in m['name']), None)
    graph_zh = next((m for m in models_data if 'Graph' in m['name'] and 'ZH' in m['name']), None)

    if lora_en and lora_zh:
        diff = (lora_en['accuracy'] - lora_zh['accuracy']) * 100
        report += f"| LoRA | {lora_en['accuracy']*100:.1f}% | {lora_zh['accuracy']*100:.1f}% | {diff:+.1f}% |\n"

    if graph_en and graph_zh:
        diff = (graph_en['accuracy'] - graph_zh['accuracy']) * 100
        report += f"| Graph Model | {graph_en['accuracy']*100:.1f}% | {graph_zh['accuracy']*100:.1f}% | {diff:+.1f}% |\n"

    report += """
### 4.3 云端API模型对比

| 模型 | 准确率 | 优势任务 | 弱势任务 |
|------|:------:|----------|----------|
"""

    api_models = [m for m in models_data if any(x in m['name'] for x in ['豆包', 'DeepSeek', 'Qwen-Max', 'Doubao'])]
    for model in sorted(api_models, key=lambda x: x['accuracy'], reverse=True):
        by_task = model.get('by_task', {})

        # 找优势和弱势任务
        task_accs = [(t, by_task.get(t, {}).get('correct', 0) / by_task.get(t, {}).get('total', 10)) for t in ALL_TASKS]
        task_accs.sort(key=lambda x: x[1], reverse=True)

        strong_tasks = [t[0] for t in task_accs[:3] if t[1] >= 0.8]
        weak_tasks = [t[0] for t in task_accs[-3:] if t[1] <= 0.2]

        report += f"| {model['name']} | {model['accuracy']*100:.1f}% | {', '.join(strong_tasks) or '-'} | {', '.join(weak_tasks) or '-'} |\n"

    report += """
---

## 五、关键发现

### 5.1 表现突出的任务
以下任务大多数模型都能较好完成:
"""

    # 统计各任务平均准确率
    task_avg = {}
    for task in ALL_TASKS:
        total = 0
        count = 0
        for model in models_data:
            by_task = model.get('by_task', {})
            if task in by_task:
                total += by_task[task]['correct'] / by_task[task]['total']
                count += 1
        task_avg[task] = total / count if count > 0 else 0

    sorted_tasks = sorted(task_avg.items(), key=lambda x: x[1], reverse=True)

    for task, avg in sorted_tasks[:5]:
        report += f"- **{task}**: 平均准确率 {avg*100:.1f}%\n"

    report += """
### 5.2 具有挑战性的任务
以下任务对所有模型都具有挑战:
"""

    for task, avg in sorted_tasks[-5:]:
        report += f"- **{task}**: 平均准确率 {avg*100:.1f}%\n"

    report += """
### 5.3 主要结论

1. **LoRA微调效果显著**: 针对图推理任务的LoRA微调大幅提升了模型在结构化推理任务上的表现
2. **中英文差距**: 所有模型在中文任务上的表现都低于英文，表明需要更多中文训练数据
3. **API模型优势**: 大参数量的云端API模型在某些复杂任务上表现更好
4. **Graph Encoder局限**: 当前图编码器的表现不如预期，需要进一步优化

---

## 六、改进建议

1. **增强中文训练数据**: 扩充中文图推理训练样本
2. **优化图编码器**: 改进图结构编码方式，提升图模型性能
3. **任务特定优化**: 针对MST、PageRank、Shortest Path等弱势任务进行专项训练
4. **Chain-of-Thought**: 考虑引入思维链提示策略改善复杂推理任务

---

*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)


def main():
    """主函数"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_dir = os.path.join(base_dir, 'data', 'eval')

    # 定义所有评测结果目录及其显示名称
    result_configs = [
        ('results_lora_en', 'LoRA (EN)', 'LoRA-EN', 'LoRA微调'),
        ('results_lora_zh', 'LoRA (ZH)', 'LoRA-ZH', 'LoRA微调'),
        ('results_api_doubao', '豆包 Doubao-1.8', 'Doubao', 'API云端'),
        ('results_api_deepseek', 'DeepSeek V3.2', 'DS-V3', 'API云端'),
        ('results_api_qwen_max', 'Qwen-Max', 'Qwen', 'API云端'),
        ('results_eval_19tasks_1024tokens', 'Graph Model (EN)', 'Graph-EN', '图模型'),
        ('results_zh_graph_1024tokens', 'Graph Model (ZH)', 'Graph-ZH', '图模型'),
        ('results_text_only_19tasks_8k', 'Qwen3-4B Baseline', 'Base', '基础模型'),
    ]

    models_data = []

    print("Loading evaluation results...")
    for dir_name, display_name, short_name, model_type in result_configs:
        results_path = os.path.join(eval_dir, dir_name)
        metrics = load_metrics(results_path)
        if metrics:
            models_data.append({
                'name': display_name,
                'short_name': short_name,
                'type': model_type,
                'accuracy': metrics.get('accuracy', 0),
                'correct': metrics.get('correct', 0),
                'total': metrics.get('total', 190),
                'by_task': metrics.get('by_task', {})
            })
            print(f"  Loaded: {display_name} - {metrics.get('accuracy', 0)*100:.2f}%")
        else:
            print(f"  [SKIP] {display_name} - No metrics found")

    if not models_data:
        print("No evaluation results found!")
        return

    print(f"\nLoaded {len(models_data)} models")

    # 生成输出
    output_dir = os.path.join(eval_dir, 'comprehensive_report')
    os.makedirs(output_dir, exist_ok=True)

    # 生成SVG图表
    print("\nGenerating charts...")
    svg_path = os.path.join(output_dir, 'comprehensive_accuracy_chart.svg')
    generate_svg_chart(models_data, svg_path)
    print(f"  Created: {svg_path}")

    heatmap_path = os.path.join(output_dir, 'task_heatmap.svg')
    generate_task_heatmap_svg(models_data, heatmap_path)
    print(f"  Created: {heatmap_path}")

    # 生成Markdown报告
    print("\nGenerating report...")
    report_path = os.path.join(output_dir, 'comprehensive_evaluation_report.md')
    generate_markdown_report(models_data, report_path)
    print(f"  Created: {report_path}")

    # 保存原始数据
    data_path = os.path.join(output_dir, 'all_models_metrics.json')
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(models_data, f, indent=2, ensure_ascii=False)
    print(f"  Created: {data_path}")

    print(f"\n" + "="*60)
    print("Comprehensive report generated successfully!")
    print(f"Output directory: {output_dir}")
    print("="*60)

    # 打印摘要
    print("\n模型性能排名:")
    for i, m in enumerate(sorted(models_data, key=lambda x: x['accuracy'], reverse=True), 1):
        print(f"  {i}. {m['name']}: {m['accuracy']*100:.2f}%")


if __name__ == '__main__':
    main()
