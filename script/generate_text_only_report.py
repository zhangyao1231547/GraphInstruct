#!/usr/bin/env python3
"""
纯文本评测报告生成脚本
从纯文本评测结果生成详细的Markdown格式报告
"""

import json
import os
import re
import argparse
from datetime import datetime
from typing import Dict, List


def extract_final_answer(text: str) -> str:
    """提取<<<...>>>中的最终答案"""
    pattern = r'<<<(.+?)>>>'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def normalize_answer(answer: str) -> str:
    """标准化答案用于比较"""
    return answer.lower().replace(' ', '').replace('[', '').replace(']', '')


def load_metrics(results_dir: str) -> Dict:
    """加载评测指标"""
    metrics_path = os.path.join(results_dir, 'metrics.json')
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, 'r') as f:
        return json.load(f)


def load_detailed_results(results_dir: str) -> List[Dict]:
    """加载详细结果"""
    details_path = os.path.join(results_dir, 'detailed_results.json')
    if not os.path.exists(details_path):
        return []
    with open(details_path, 'r') as f:
        return json.load(f)


def generate_report(metrics: Dict, detailed_results: List[Dict], output_path: str):
    """生成Markdown格式的评测报告"""

    report = []

    # 标题
    report.append("# Qwen3-4B 纯文本评测报告 (Baseline)")
    report.append("")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # 总体概述
    report.append("## 1. 总体概述")
    report.append("")
    report.append(f"- **模型**: {metrics.get('model', 'Qwen3-4B-Instruct')}")
    report.append(f"- **总样本数**: {metrics.get('total', 0)}")
    report.append(f"- **正确数**: {metrics.get('correct', 0)} ({metrics.get('accuracy', 0)*100:.2f}%)")
    report.append("")
    report.append("> 注: 纯文本评测使用原始Qwen3-4B模型,不使用图编码,作为baseline对比。")
    report.append("")

    # 任务类型分析表格
    report.append("## 2. 各任务类型评测结果")
    report.append("")
    report.append("| 任务类型 | 样本数 | 正确数 | 准确率 |")
    report.append("|----------|--------|--------|--------|")

    if 'by_task' in metrics:
        # 按准确率排序
        sorted_tasks = sorted(
            metrics['by_task'].items(),
            key=lambda x: x[1]['correct'] / x[1]['total'] if x[1]['total'] > 0 else 0,
            reverse=True
        )

        for task_type, task_stats in sorted_tasks:
            rate = task_stats['correct'] / task_stats['total'] * 100 if task_stats['total'] > 0 else 0
            report.append(
                f"| {task_type} | {task_stats['total']} | "
                f"{task_stats['correct']} | {rate:.1f}% |"
            )

    report.append("")

    # 任务难度分析
    report.append("## 3. 任务难度分析")
    report.append("")

    if 'by_task' in metrics:
        # 分类任务难度
        easy_tasks = []
        medium_tasks = []
        hard_tasks = []

        for task_type, task_stats in metrics['by_task'].items():
            rate = task_stats['correct'] / task_stats['total'] if task_stats['total'] > 0 else 0
            if rate >= 0.7:
                easy_tasks.append((task_type, rate))
            elif rate >= 0.3:
                medium_tasks.append((task_type, rate))
            else:
                hard_tasks.append((task_type, rate))

        if easy_tasks:
            report.append("### 简单任务 (准确率 >= 70%)")
            for task, rate in sorted(easy_tasks, key=lambda x: -x[1]):
                report.append(f"- {task}: {rate*100:.1f}%")
            report.append("")

        if medium_tasks:
            report.append("### 中等任务 (准确率 30-70%)")
            for task, rate in sorted(medium_tasks, key=lambda x: -x[1]):
                report.append(f"- {task}: {rate*100:.1f}%")
            report.append("")

        if hard_tasks:
            report.append("### 困难任务 (准确率 < 30%)")
            for task, rate in sorted(hard_tasks, key=lambda x: -x[1]):
                report.append(f"- {task}: {rate*100:.1f}%")
            report.append("")

    # 示例展示
    report.append("## 4. 预测示例")
    report.append("")

    if detailed_results:
        # 展示一些正确和错误的例子
        correct_examples = []
        wrong_examples = []

        for result in detailed_results:
            if result.get('is_correct', False) and len(correct_examples) < 3:
                correct_examples.append(result)
            elif not result.get('is_correct', True) and len(wrong_examples) < 3:
                wrong_examples.append(result)

            if len(correct_examples) >= 3 and len(wrong_examples) >= 3:
                break

        if correct_examples:
            report.append("### 正确预测示例")
            report.append("")
            for i, ex in enumerate(correct_examples, 1):
                report.append(f"**示例 {i}** (任务: {ex.get('task_type', 'unknown')})")
                report.append("```")
                report.append(f"Ground Truth: {ex.get('gt_answer', '')[:200]}")
                report.append(f"Prediction:   {ex.get('pred_answer', '')[:200]}")
                report.append("```")
                report.append("")

        if wrong_examples:
            report.append("### 错误预测示例")
            report.append("")
            for i, ex in enumerate(wrong_examples, 1):
                report.append(f"**示例 {i}** (任务: {ex.get('task_type', 'unknown')})")
                report.append("```")
                report.append(f"Ground Truth: {ex.get('gt_answer', '')[:200]}")
                report.append(f"Prediction:   {ex.get('pred_answer', '')[:200]}")
                report.append("```")
                report.append("")

    # 配置信息
    report.append("## 5. 评测配置")
    report.append("")
    report.append("| 参数 | 值 |")
    report.append("|------|-----|")
    report.append(f"| 模型 | Qwen3-4B-Instruct (原始) |")
    report.append(f"| 图编码 | 无 (纯文本) |")
    report.append(f"| 任务数 | 19 |")
    report.append(f"| 每任务样本数 | {metrics.get('total', 0) // 19 if metrics else 'N/A'} |")
    report.append(f"| 总样本数 | {metrics.get('total', 'N/A')} |")
    report.append("")

    # 结论
    report.append("## 6. 结论")
    report.append("")

    if metrics:
        avg_rate = metrics.get('accuracy', 0) * 100
        if avg_rate >= 70:
            report.append(f"原始模型整体表现**优秀**,平均准确率达到 {avg_rate:.1f}%。")
        elif avg_rate >= 50:
            report.append(f"原始模型整体表现**良好**,平均准确率为 {avg_rate:.1f}%。")
        elif avg_rate >= 30:
            report.append(f"原始模型整体表现**一般**,平均准确率为 {avg_rate:.1f}%,图编码可能有帮助。")
        else:
            report.append(f"原始模型整体表现**较差**,平均准确率仅为 {avg_rate:.1f}%,说明图推理任务需要专门的图结构理解能力。")

        report.append("")

        # 找出表现最好和最差的任务
        if 'by_task' in metrics:
            sorted_tasks = sorted(
                metrics['by_task'].items(),
                key=lambda x: x[1]['correct'] / x[1]['total'] if x[1]['total'] > 0 else 0,
                reverse=True
            )

            if sorted_tasks:
                best_task = sorted_tasks[0]
                best_rate = best_task[1]['correct'] / best_task[1]['total'] * 100 if best_task[1]['total'] > 0 else 0
                worst_task = sorted_tasks[-1]
                worst_rate = worst_task[1]['correct'] / worst_task[1]['total'] * 100 if worst_task[1]['total'] > 0 else 0

                report.append(f"- 表现最佳任务: **{best_task[0]}** ({best_rate:.1f}%)")
                report.append(f"- 表现最差任务: **{worst_task[0]}** ({worst_rate:.1f}%)")

    report.append("")
    report.append("---")
    report.append("*本报告由纯文本评测系统自动生成 (Baseline)*")

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"评测报告已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate text-only evaluation report')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing evaluation results')
    parser.add_argument('--output-report', type=str, default=None,
                        help='Output path for the report')

    args = parser.parse_args()

    if args.output_report is None:
        args.output_report = os.path.join(args.results_dir, 'evaluation_report.md')

    # 加载结果
    metrics = load_metrics(args.results_dir)
    detailed_results = load_detailed_results(args.results_dir)

    if metrics is None:
        print(f"ERROR: metrics.json not found in {args.results_dir}")
        return

    # 生成报告
    generate_report(metrics, detailed_results, args.output_report)


if __name__ == '__main__':
    main()
