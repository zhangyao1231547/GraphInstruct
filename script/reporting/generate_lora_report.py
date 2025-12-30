#!/usr/bin/env python3
"""
LoRA模型评测报告生成脚本
根据评测结果生成Markdown格式的评测报告
"""

import json
import os
import argparse
from datetime import datetime
from typing import Dict, List, Tuple


def load_metrics(results_dir: str) -> Dict:
    """加载评测指标"""
    metrics_path = os.path.join(results_dir, 'metrics.json')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def categorize_tasks(by_task: Dict) -> Tuple[List, List, List, List]:
    """按准确率分类任务"""
    perfect = []      # 100%
    high = []         # 80-99%
    medium = []       # 40-79%
    low = []          # 0-39%

    for task, stats in by_task.items():
        total = stats['total']
        correct = stats['correct']
        accuracy = correct / total if total > 0 else 0

        item = {
            'task': task,
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        }

        if accuracy == 1.0:
            perfect.append(item)
        elif accuracy >= 0.8:
            high.append(item)
        elif accuracy >= 0.4:
            medium.append(item)
        else:
            low.append(item)

    # 按准确率排序
    for lst in [perfect, high, medium, low]:
        lst.sort(key=lambda x: (-x['accuracy'], x['task']))

    return perfect, high, medium, low


def generate_english_report(metrics: Dict, output_path: str):
    """生成英文评测报告"""
    model_name = metrics.get('model', 'Unknown Model')
    total = metrics.get('total', 0)
    correct = metrics.get('correct', 0)
    accuracy = metrics.get('accuracy', 0)
    by_task = metrics.get('by_task', {})
    eval_time = metrics.get('evaluation_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    perfect, high, medium, low = categorize_tasks(by_task)

    report = f"""# LoRA Model Evaluation Report (English)

## Model Information
- **Model**: {model_name}
- **Base Model**: Qwen3-4B-Instruct-2507
- **Evaluation Type**: Text-only (Pure LLM without graph encoding)
- **Language**: English
- **Evaluation Time**: {eval_time}

## Overall Performance

| Metric | Value |
|--------|-------|
| **Total Accuracy** | **{accuracy*100:.2f}%** |
| Total Samples | {total} |
| Correct Predictions | {correct} |
| Tasks Evaluated | {len(by_task)} |

## Performance by Task

"""

    if perfect:
        report += "### Perfect Performance (100%)\n"
        report += "| Task | Correct/Total | Accuracy |\n"
        report += "|------|---------------|----------|\n"
        for item in perfect:
            report += f"| {item['task']} | {item['correct']}/{item['total']} | {item['accuracy']*100:.1f}% |\n"
        report += "\n"

    if high:
        report += "### High Performance (80-99%)\n"
        report += "| Task | Correct/Total | Accuracy |\n"
        report += "|------|---------------|----------|\n"
        for item in high:
            report += f"| {item['task']} | {item['correct']}/{item['total']} | {item['accuracy']*100:.1f}% |\n"
        report += "\n"

    if medium:
        report += "### Medium Performance (40-79%)\n"
        report += "| Task | Correct/Total | Accuracy |\n"
        report += "|------|---------------|----------|\n"
        for item in medium:
            report += f"| {item['task']} | {item['correct']}/{item['total']} | {item['accuracy']*100:.1f}% |\n"
        report += "\n"

    if low:
        report += "### Low Performance (0-39%)\n"
        report += "| Task | Correct/Total | Accuracy |\n"
        report += "|------|---------------|----------|\n"
        for item in low:
            report += f"| {item['task']} | {item['correct']}/{item['total']} | {item['accuracy']*100:.1f}% |\n"
        report += "\n"

    # 任务类别分析
    report += """## Task Category Analysis

### Graph Traversal Tasks
"""
    traversal_tasks = ['BFS', 'DFS']
    for task in traversal_tasks:
        if task in by_task:
            acc = by_task[task]['correct'] / by_task[task]['total'] * 100
            status = "Excellent" if acc >= 90 else "Good" if acc >= 70 else "Moderate" if acc >= 50 else "Poor"
            report += f"- {task}: {acc:.0f}% - {status}\n"

    report += "\n### Graph Property Tasks\n"
    property_tasks = ['degree', 'neighbor', 'edge', 'connectivity', 'cycle', 'bipartite', 'connected_component', 'diameter']
    for task in property_tasks:
        if task in by_task:
            acc = by_task[task]['correct'] / by_task[task]['total'] * 100
            status = "Excellent" if acc >= 90 else "Good" if acc >= 70 else "Moderate" if acc >= 50 else "Poor"
            report += f"- {task}: {acc:.0f}% - {status}\n"

    report += "\n### Node Similarity Tasks\n"
    similarity_tasks = ['common_neighbor', 'jaccard']
    for task in similarity_tasks:
        if task in by_task:
            acc = by_task[task]['correct'] / by_task[task]['total'] * 100
            status = "Excellent" if acc >= 90 else "Good" if acc >= 70 else "Moderate" if acc >= 50 else "Poor"
            report += f"- {task}: {acc:.0f}% - {status}\n"

    report += "\n### Path & Flow Tasks\n"
    path_tasks = ['predecessor', 'topological_sort', 'maximum_flow', 'shortest_path']
    for task in path_tasks:
        if task in by_task:
            acc = by_task[task]['correct'] / by_task[task]['total'] * 100
            status = "Excellent" if acc >= 90 else "Good" if acc >= 70 else "Moderate" if acc >= 50 else "Poor"
            report += f"- {task}: {acc:.0f}% - {status}\n"

    report += "\n### Centrality Tasks\n"
    centrality_tasks = ['page_rank', 'clustering_coefficient']
    for task in centrality_tasks:
        if task in by_task:
            acc = by_task[task]['correct'] / by_task[task]['total'] * 100
            status = "Excellent" if acc >= 90 else "Good" if acc >= 70 else "Moderate" if acc >= 50 else "Poor"
            report += f"- {task}: {acc:.0f}% - {status}\n"

    report += "\n### Tree Tasks\n"
    tree_tasks = ['MST']
    for task in tree_tasks:
        if task in by_task:
            acc = by_task[task]['correct'] / by_task[task]['total'] * 100
            status = "Excellent" if acc >= 90 else "Good" if acc >= 70 else "Moderate" if acc >= 50 else "Poor"
            report += f"- {task}: {acc:.0f}% - {status}\n"

    # 关键观察
    report += f"""
## Key Observations

1. **Strengths**: The LoRA fine-tuned model excels at:
"""
    for item in perfect[:5]:
        report += f"   - {item['task']} ({item['accuracy']*100:.0f}%)\n"

    report += "\n2. **Weaknesses**: The model struggles with:\n"
    for item in low[:5]:
        report += f"   - {item['task']} ({item['accuracy']*100:.0f}%)\n"

    report += f"""
3. **Overall Assessment**: With {accuracy*100:.2f}% accuracy, the LoRA fine-tuned model {"demonstrates strong" if accuracy >= 0.7 else "shows moderate" if accuracy >= 0.5 else "needs improvement in"} graph reasoning capabilities.

## Recommendations

1. {"Continue current training approach for well-performing tasks" if accuracy >= 0.7 else "Consider additional training data"}
2. Focus on improving tasks with 0% accuracy through targeted fine-tuning
3. Consider chain-of-thought prompting for complex algorithmic tasks
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"English report saved to: {output_path}")


def generate_chinese_report(metrics: Dict, output_path: str, en_metrics: Dict = None):
    """生成中文评测报告"""
    model_name = metrics.get('model', 'Unknown Model')
    total = metrics.get('total', 0)
    correct = metrics.get('correct', 0)
    accuracy = metrics.get('accuracy', 0)
    by_task = metrics.get('by_task', {})
    eval_time = metrics.get('evaluation_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    perfect, high, medium, low = categorize_tasks(by_task)

    report = f"""# LoRA模型评测报告 (中文)

## 模型信息
- **模型**: {model_name}
- **基座模型**: Qwen3-4B-Instruct-2507
- **评测类型**: 纯文本 (不使用图编码器)
- **语言**: 中文
- **评测时间**: {eval_time}

## 总体性能

| 指标 | 数值 |
|------|------|
| **总体准确率** | **{accuracy*100:.2f}%** |
| 总样本数 | {total} |
| 正确预测数 | {correct} |
| 评测任务数 | {len(by_task)} |

## 各任务性能详情

"""

    if perfect:
        report += "### 完美表现 (100%)\n"
        report += "| 任务 | 正确/总数 | 准确率 |\n"
        report += "|------|-----------|--------|\n"
        for item in perfect:
            report += f"| {item['task']} | {item['correct']}/{item['total']} | {item['accuracy']*100:.1f}% |\n"
        report += "\n"

    if high:
        report += "### 高性能任务 (80-99%)\n"
        report += "| 任务 | 正确/总数 | 准确率 |\n"
        report += "|------|-----------|--------|\n"
        for item in high:
            report += f"| {item['task']} | {item['correct']}/{item['total']} | {item['accuracy']*100:.1f}% |\n"
        report += "\n"

    if medium:
        report += "### 中等性能任务 (40-79%)\n"
        report += "| 任务 | 正确/总数 | 准确率 |\n"
        report += "|------|-----------|--------|\n"
        for item in medium:
            report += f"| {item['task']} | {item['correct']}/{item['total']} | {item['accuracy']*100:.1f}% |\n"
        report += "\n"

    if low:
        report += "### 低性能任务 (0-39%)\n"
        report += "| 任务 | 正确/总数 | 准确率 |\n"
        report += "|------|-----------|--------|\n"
        for item in low:
            report += f"| {item['task']} | {item['correct']}/{item['total']} | {item['accuracy']*100:.1f}% |\n"
        report += "\n"

    # 中英文对比
    if en_metrics:
        en_by_task = en_metrics.get('by_task', {})
        en_accuracy = en_metrics.get('accuracy', 0)

        report += """## 英文 vs 中文性能对比

| 任务 | 英文准确率 | 中文准确率 | 差异 |
|------|------------|------------|------|
"""
        all_tasks = sorted(set(list(by_task.keys()) + list(en_by_task.keys())))
        for task in all_tasks:
            en_acc = en_by_task.get(task, {}).get('correct', 0) / en_by_task.get(task, {}).get('total', 1) * 100 if task in en_by_task else 0
            zh_acc = by_task.get(task, {}).get('correct', 0) / by_task.get(task, {}).get('total', 1) * 100 if task in by_task else 0
            diff = zh_acc - en_acc
            report += f"| {task} | {en_acc:.0f}% | {zh_acc:.0f}% | {diff:+.0f}% |\n"

        report += f"""
## 关键发现

### 1. 中英文性能差距
- **英文准确率**: {en_accuracy*100:.2f}%
- **中文准确率**: {accuracy*100:.2f}%
- **差距**: {(en_accuracy - accuracy)*100:.2f}个百分点

"""

    # 问题分析
    report += """### 2. 中文任务分析

"""
    if perfect or high:
        report += "**相对较好的任务**:\n"
        for item in (perfect + high)[:5]:
            report += f"- {item['task']}: {item['accuracy']*100:.0f}%\n"
        report += "\n"

    if low:
        report += "**需要改进的任务**:\n"
        for item in low[:5]:
            report += f"- {item['task']}: {item['accuracy']*100:.0f}%\n"
        report += "\n"

    report += f"""### 3. 问题分析

模型在中文图推理任务上表现{"较好" if accuracy >= 0.5 else "不佳"}，可能原因：

1. **训练数据分布**: LoRA微调数据中中文样本比例可能较低
2. **指令理解**: 中文指令的解析和答案格式化可能存在问题
3. **输出格式**: 模型可能没有正确遵循中文答案格式要求 `<<<答案>>>`

## 建议

1. **增加中文训练数据**: 在LoRA微调时加入更多中文图推理样本
2. **改进提示工程**: 优化中文指令模板，明确输出格式要求
3. **检查答案提取逻辑**: 确保中文答案的正则提取与实际输出匹配
4. **多语言对齐训练**: 考虑使用中英文对齐的训练策略
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Chinese report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate LoRA evaluation report')
    parser.add_argument('--en-results-dir', type=str,
                        help='Directory containing English evaluation results')
    parser.add_argument('--zh-results-dir', type=str,
                        help='Directory containing Chinese evaluation results')
    parser.add_argument('--language', type=str, choices=['en', 'zh', 'both'], default='both',
                        help='Language of report to generate')

    args = parser.parse_args()

    en_metrics = None
    zh_metrics = None

    # 加载英文结果
    if args.en_results_dir and os.path.exists(args.en_results_dir):
        try:
            en_metrics = load_metrics(args.en_results_dir)
            print(f"Loaded English metrics: {en_metrics.get('accuracy', 0)*100:.2f}% accuracy")
        except Exception as e:
            print(f"Warning: Could not load English metrics: {e}")

    # 加载中文结果
    if args.zh_results_dir and os.path.exists(args.zh_results_dir):
        try:
            zh_metrics = load_metrics(args.zh_results_dir)
            print(f"Loaded Chinese metrics: {zh_metrics.get('accuracy', 0)*100:.2f}% accuracy")
        except Exception as e:
            print(f"Warning: Could not load Chinese metrics: {e}")

    # 生成报告
    if args.language in ['en', 'both'] and en_metrics:
        output_path = os.path.join(args.en_results_dir, 'evaluation_report.md')
        generate_english_report(en_metrics, output_path)

    if args.language in ['zh', 'both'] and zh_metrics:
        output_path = os.path.join(args.zh_results_dir, 'evaluation_report.md')
        generate_chinese_report(zh_metrics, output_path, en_metrics)

    print("\nReport generation completed!")


if __name__ == '__main__':
    main()
