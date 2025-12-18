#!/usr/bin/env python3
"""
GraphAgent 评测报告生成脚本
从评测结果生成详细的Markdown格式报告
  - metrics.json - 汇总评测指标
  - detailed_results.json - 每个样本的详细预测结果
  - sample_predictions.txt - 可读格式的样本预测(含完整token序列)
  - token_sequences_debug.json - token级调试信息
  - mismatches.txt - 错误预测分析
  - evaluation_report.md - Markdown评测报告
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
        return matches[-1].strip()  # 取最后一个答案
    return text.strip()


def normalize_answer(answer: str) -> str:
    """标准化答案用于比较"""
    # 移除空格、方括号，转小写
    return answer.lower().replace(' ', '').replace('[', '').replace(']', '')


def compute_answer_based_metrics(detailed_results: List[Dict]) -> Dict:
    """基于最终答案计算准确率"""
    total = len(detailed_results)
    correct = 0
    correct_by_task = {}

    for item in detailed_results:
        task = item.get('task_type', 'unknown')
        gt = extract_final_answer(item.get('ground_truth', ''))
        pred = extract_final_answer(item.get('prediction', ''))

        if task not in correct_by_task:
            correct_by_task[task] = {'total': 0, 'correct': 0}
        correct_by_task[task]['total'] += 1

        # 标准化比较
        gt_norm = normalize_answer(gt)
        pred_norm = normalize_answer(pred)

        is_correct = gt_norm == pred_norm
        if is_correct:
            correct += 1
            correct_by_task[task]['correct'] += 1

    return {
        'total': total,
        'correct': correct,
        'accuracy': correct / total if total > 0 else 0,
        'by_task': correct_by_task
    }


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


def analyze_errors(detailed_results: List[Dict]) -> Dict:
    """分析错误模式"""
    error_analysis = {
        'empty_predictions': 0,
        'wrong_format': 0,
        'wrong_answer': 0,
        'partial_correct': 0,
        'by_task': {}
    }

    for result in detailed_results:
        pred = result.get('prediction', '').strip()
        gt = result.get('ground_truth', '').strip()
        task_type = result.get('task_type', 'unknown')

        if task_type not in error_analysis['by_task']:
            error_analysis['by_task'][task_type] = {
                'empty': 0,
                'wrong_format': 0,
                'wrong_answer': 0,
                'partial': 0
            }

        if not pred:
            error_analysis['empty_predictions'] += 1
            error_analysis['by_task'][task_type]['empty'] += 1
        elif pred.lower() == gt.lower():
            continue  # 正确,跳过
        elif gt.lower() in pred.lower() or pred.lower() in gt.lower():
            error_analysis['partial_correct'] += 1
            error_analysis['by_task'][task_type]['partial'] += 1
        else:
            error_analysis['wrong_answer'] += 1
            error_analysis['by_task'][task_type]['wrong_answer'] += 1

    return error_analysis


def generate_report(metrics: Dict, detailed_results: List[Dict], output_path: str):
    """生成Markdown格式的评测报告"""

    error_analysis = analyze_errors(detailed_results) if detailed_results else {}

    # 计算基于最终答案的准确率
    answer_metrics = compute_answer_based_metrics(detailed_results) if detailed_results else None

    report = []

    # 标题
    report.append("# GraphAgent 评测报告")
    report.append("")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # 总体概述
    report.append("## 1. 总体概述")
    report.append("")

    if answer_metrics:
        report.append(f"- **总样本数**: {answer_metrics['total']}")
        report.append(f"- **最终答案正确数**: {answer_metrics['correct']} ({answer_metrics['accuracy']*100:.2f}%)")
        report.append("")
        report.append("> 注: 最终答案准确率基于提取`<<<...>>>`中的答案进行比较，忽略推理步骤的差异。")
        report.append("")

    if metrics:
        report.append("### 完整文本匹配指标（参考）")
        report.append(f"- **精确匹配数**: {metrics['exact_match']} ({metrics['exact_match_rate']*100:.2f}%)")
        report.append(f"- **部分匹配数**: {metrics['partial_match']} ({metrics['partial_match_rate']*100:.2f}%)")
        report.append("")

    # 任务类型分析表格 - 基于最终答案
    report.append("## 2. 各任务类型评测结果（基于最终答案）")
    report.append("")
    report.append("| 任务类型 | 样本数 | 正确数 | 准确率 |")
    report.append("|----------|--------|--------|--------|")

    if answer_metrics and 'by_task' in answer_metrics:
        # 按准确率排序
        sorted_tasks = sorted(
            answer_metrics['by_task'].items(),
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

    # 错误分析
    if error_analysis:
        report.append("## 3. 错误分析")
        report.append("")
        report.append(f"- **空预测**: {error_analysis['empty_predictions']}")
        report.append(f"- **错误答案**: {error_analysis['wrong_answer']}")
        report.append(f"- **部分正确**: {error_analysis['partial_correct']}")
        report.append("")

    # 任务难度分析
    report.append("## 4. 任务难度分析")
    report.append("")

    if answer_metrics and 'by_task' in answer_metrics:
        # 分类任务难度
        easy_tasks = []
        medium_tasks = []
        hard_tasks = []

        for task_type, task_stats in answer_metrics['by_task'].items():
            rate = task_stats['correct'] / task_stats['total'] if task_stats['total'] > 0 else 0
            if rate >= 0.7:
                easy_tasks.append((task_type, rate))
            elif rate >= 0.3:
                medium_tasks.append((task_type, rate))
            else:
                hard_tasks.append((task_type, rate))

        if easy_tasks:
            report.append("### 简单任务 (准确率 ≥ 70%)")
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
    report.append("## 5. 预测示例")
    report.append("")

    if detailed_results:
        # 展示一些正确和错误的例子（基于最终答案）
        correct_examples = []
        wrong_examples = []

        for result in detailed_results:
            gt_answer = extract_final_answer(result.get('ground_truth', ''))
            pred_answer = extract_final_answer(result.get('prediction', ''))
            gt_norm = normalize_answer(gt_answer)
            pred_norm = normalize_answer(pred_answer)

            if gt_norm == pred_norm and len(correct_examples) < 3:
                correct_examples.append(result)
            elif gt_norm != pred_norm and len(wrong_examples) < 3:
                wrong_examples.append(result)

            if len(correct_examples) >= 3 and len(wrong_examples) >= 3:
                break

        if correct_examples:
            report.append("### 正确预测示例")
            report.append("")
            for i, ex in enumerate(correct_examples, 1):
                gt_answer = extract_final_answer(ex.get('ground_truth', ''))
                pred_answer = extract_final_answer(ex.get('prediction', ''))
                report.append(f"**示例 {i}** (任务: {ex.get('task_type', 'unknown')})")
                report.append("```")
                report.append(f"Ground Truth Answer: {gt_answer}")
                report.append(f"Prediction Answer:   {pred_answer}")
                report.append("```")
                report.append("")

        if wrong_examples:
            report.append("### 错误预测示例")
            report.append("")
            for i, ex in enumerate(wrong_examples, 1):
                gt_answer = extract_final_answer(ex.get('ground_truth', ''))
                pred_answer = extract_final_answer(ex.get('prediction', ''))
                report.append(f"**示例 {i}** (任务: {ex.get('task_type', 'unknown')})")
                report.append("```")
                report.append(f"Ground Truth Answer: {gt_answer}")
                report.append(f"Prediction Answer:   {pred_answer}")
                report.append("```")
                report.append("")

    # 配置信息
    report.append("## 6. 评测配置")
    report.append("")
    report.append("| 参数 | 值 |")
    report.append("|------|-----|")
    report.append(f"| 模型 | GraphAgent-Qwen3-4B |")
    report.append(f"| 任务数 | 19 |")
    report.append(f"| 每任务样本数 | {metrics['total'] // 19 if metrics else 'N/A'} |")
    report.append(f"| 总样本数 | {metrics['total'] if metrics else 'N/A'} |")
    report.append("")

    # 结论
    report.append("## 7. 结论")
    report.append("")

    if answer_metrics:
        avg_rate = answer_metrics['accuracy'] * 100
        if avg_rate >= 70:
            report.append(f"模型整体表现**优秀**,平均准确率达到 {avg_rate:.1f}%。")
        elif avg_rate >= 50:
            report.append(f"模型整体表现**良好**,平均准确率为 {avg_rate:.1f}%。")
        elif avg_rate >= 30:
            report.append(f"模型整体表现**一般**,平均准确率为 {avg_rate:.1f}%,仍有提升空间。")
        else:
            report.append(f"模型整体表现**需改进**,平均准确率仅为 {avg_rate:.1f}%。")

        report.append("")

        # 找出表现最好和最差的任务
        if 'by_task' in answer_metrics:
            sorted_tasks = sorted(
                answer_metrics['by_task'].items(),
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
    report.append("*本报告由 GraphAgent 评测系统自动生成*")

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"评测报告已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate GraphAgent evaluation report')
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
