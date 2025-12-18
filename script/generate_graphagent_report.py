#!/usr/bin/env python3
"""
GraphAgent评测报告生成脚本
从GraphAgent评测结果生成详细的Markdown格式报告
"""

import json
import os
import re
import argparse
from datetime import datetime
from typing import Dict, List


def extract_final_answer(text: str) -> str:
    """提取<<<...>>>中的最终答案"""
    if not text:
        return ""
    pattern = r'<<<(.+?)>>>'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


def normalize_answer(answer: str) -> str:
    """标准化答案用于比较"""
    if not answer:
        return ""
    answer = answer.lower()
    answer = answer.replace(' ', '').replace('\n', '').replace('\t', '')
    answer = answer.replace('[', '').replace(']', '')
    answer = answer.replace('(', '').replace(')', '')
    return answer


def compare_answers(pred: str, gt: str) -> bool:
    """比较预测和标准答案"""
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    return pred_norm == gt_norm


def load_detailed_results(results_dir: str) -> List[Dict]:
    """加载详细结果"""
    details_path = os.path.join(results_dir, 'detailed_results.json')
    if not os.path.exists(details_path):
        return []
    with open(details_path, 'r') as f:
        return json.load(f)


def calculate_metrics(detailed_results: List[Dict]) -> Dict:
    """重新计算指标（使用答案提取）"""
    from collections import defaultdict

    stats = {
        'total': 0,
        'correct': 0,
        'by_task': defaultdict(lambda: {'total': 0, 'correct': 0})
    }

    for item in detailed_results:
        stats['total'] += 1
        task_type = item.get('task_type', 'unknown')
        stats['by_task'][task_type]['total'] += 1

        gt_answer = extract_final_answer(item.get('ground_truth', ''))
        pred_text = item.get('prediction', '')
        pred_answer = extract_final_answer(pred_text)

        if compare_answers(pred_answer, gt_answer):
            stats['correct'] += 1
            stats['by_task'][task_type]['correct'] += 1

    stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    stats['by_task'] = dict(stats['by_task'])

    return stats


def generate_report(results_dir: str, output_path: str, max_new_tokens: int = 1024):
    """生成Markdown格式的评测报告"""

    detailed_results = load_detailed_results(results_dir)
    if not detailed_results:
        print(f"ERROR: detailed_results.json not found in {results_dir}")
        return

    metrics = calculate_metrics(detailed_results)

    report = []

    # 标题
    report.append(f"# GraphAgent 评测报告 (max_new_tokens={max_new_tokens})")
    report.append("")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # 总体概述
    report.append("## 1. 总体概述")
    report.append("")
    report.append(f"- **模型**: GraphAgent-Qwen3-4B")
    report.append(f"- **总样本数**: {metrics.get('total', 0)}")
    report.append(f"- **正确数**: {metrics.get('correct', 0)} ({metrics.get('accuracy', 0)*100:.2f}%)")
    report.append(f"- **max_new_tokens**: {max_new_tokens}")
    report.append("")
    report.append("> 注: 使用`<<<...>>>`答案提取进行评分")
    report.append("")

    # 任务类型分析表格
    report.append("## 2. 各任务类型评测结果")
    report.append("")
    report.append("| 任务类型 | 样本数 | 正确数 | 准确率 |")
    report.append("|----------|--------|--------|--------|")

    if 'by_task' in metrics:
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

    correct_examples = []
    wrong_examples = []

    for result in detailed_results:
        gt_answer = extract_final_answer(result.get('ground_truth', ''))
        pred_text = result.get('prediction', '')
        pred_answer = extract_final_answer(pred_text)
        is_correct = compare_answers(pred_answer, gt_answer)

        if is_correct and len(correct_examples) < 3:
            correct_examples.append({
                'task_type': result.get('task_type', 'unknown'),
                'gt_answer': gt_answer,
                'pred_answer': pred_answer
            })
        elif not is_correct and len(wrong_examples) < 3:
            wrong_examples.append({
                'task_type': result.get('task_type', 'unknown'),
                'gt_answer': gt_answer,
                'pred_answer': pred_answer,
                'full_pred': pred_text[:500] if pred_text else ''
            })

        if len(correct_examples) >= 3 and len(wrong_examples) >= 3:
            break

    if correct_examples:
        report.append("### 正确预测示例")
        report.append("")
        for i, ex in enumerate(correct_examples, 1):
            report.append(f"**示例 {i}** (任务: {ex['task_type']})")
            report.append("```")
            report.append(f"Ground Truth: {ex['gt_answer'][:200]}")
            report.append(f"Prediction:   {ex['pred_answer'][:200]}")
            report.append("```")
            report.append("")

    if wrong_examples:
        report.append("### 错误预测示例")
        report.append("")
        for i, ex in enumerate(wrong_examples, 1):
            report.append(f"**示例 {i}** (任务: {ex['task_type']})")
            report.append("```")
            report.append(f"Ground Truth: {ex['gt_answer'][:200]}")
            report.append(f"Prediction:   {ex['pred_answer'][:200]}")
            if ex['full_pred']:
                report.append(f"Full output (first 300 chars): {ex['full_pred'][:300]}...")
            report.append("```")
            report.append("")

    # 配置信息
    report.append("## 5. 评测配置")
    report.append("")
    report.append("| 参数 | 值 |")
    report.append("|------|-----|")
    report.append(f"| 模型 | GraphAgent-Qwen3-4B |")
    report.append(f"| 图编码 | MetaHGT (768-dim) |")
    report.append(f"| max_new_tokens | {max_new_tokens} |")
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
            report.append(f"GraphAgent模型整体表现**优秀**,平均准确率达到 {avg_rate:.1f}%。")
        elif avg_rate >= 50:
            report.append(f"GraphAgent模型整体表现**良好**,平均准确率为 {avg_rate:.1f}%。")
        elif avg_rate >= 30:
            report.append(f"GraphAgent模型整体表现**一般**,平均准确率为 {avg_rate:.1f}%。")
        else:
            report.append(f"GraphAgent模型整体表现**有待提升**,平均准确率为 {avg_rate:.1f}%。")

        report.append("")

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
    report.append("*本报告由GraphAgent评测系统自动生成*")

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    # 保存更正后的指标
    metrics_path = os.path.join(results_dir, 'metrics_corrected.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"评测报告已保存到: {output_path}")
    print(f"更正后的指标已保存到: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate GraphAgent evaluation report')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing evaluation results')
    parser.add_argument('--output-report', type=str, default=None,
                        help='Output path for the report')
    parser.add_argument('--max-new-tokens', type=int, default=1024,
                        help='max_new_tokens setting used in evaluation')

    args = parser.parse_args()

    if args.output_report is None:
        args.output_report = os.path.join(args.results_dir, 'evaluation_report.md')

    generate_report(args.results_dir, args.output_report, args.max_new_tokens)


if __name__ == '__main__':
    main()
