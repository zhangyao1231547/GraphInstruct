#!/usr/bin/env python3
"""
重新计算评测指标 - 使用<<<...>>>答案提取
"""

import json
import re
import sys
from collections import defaultdict


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
    # 移除空格、换行、标点等
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


def recalculate_metrics(results_dir: str):
    """重新计算评测指标"""
    # 加载详细结果
    with open(f"{results_dir}/detailed_results.json", 'r') as f:
        results = json.load(f)

    stats = {
        'total': 0,
        'correct': 0,
        'by_task': defaultdict(lambda: {'total': 0, 'correct': 0})
    }

    correct_examples = []
    wrong_examples = []

    for item in results:
        stats['total'] += 1
        task_type = item.get('task_type', 'unknown')
        stats['by_task'][task_type]['total'] += 1

        # 提取答案
        gt_answer = extract_final_answer(item.get('ground_truth', ''))
        pred_text = item.get('prediction', '')
        pred_answer = extract_final_answer(pred_text)

        # 比较答案
        is_correct = compare_answers(pred_answer, gt_answer)

        if is_correct:
            stats['correct'] += 1
            stats['by_task'][task_type]['correct'] += 1
            if len(correct_examples) < 5:
                correct_examples.append({
                    'task': task_type,
                    'gt': gt_answer[:100],
                    'pred': pred_answer[:100]
                })
        else:
            if len(wrong_examples) < 5:
                wrong_examples.append({
                    'task': task_type,
                    'gt': gt_answer[:200],
                    'pred': pred_answer[:200],
                    'full_pred': pred_text[:500] if pred_text else ''
                })

    # 计算准确率
    accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

    print("=" * 60)
    print("GraphAgent 评测结果 (使用<<<答案>>>提取)")
    print("=" * 60)
    print(f"总样本数: {stats['total']}")
    print(f"正确数: {stats['correct']}")
    print(f"准确率: {accuracy * 100:.2f}%")
    print()

    print("各任务类型结果:")
    print("-" * 40)

    # 按准确率排序
    sorted_tasks = sorted(
        stats['by_task'].items(),
        key=lambda x: x[1]['correct'] / x[1]['total'] if x[1]['total'] > 0 else 0,
        reverse=True
    )

    for task, task_stats in sorted_tasks:
        task_acc = task_stats['correct'] / task_stats['total'] * 100 if task_stats['total'] > 0 else 0
        print(f"  {task}: {task_stats['correct']}/{task_stats['total']} ({task_acc:.1f}%)")

    print()
    print("正确预测示例:")
    print("-" * 40)
    for ex in correct_examples[:3]:
        print(f"  [{ex['task']}] GT: {ex['gt']} | Pred: {ex['pred']}")

    print()
    print("错误预测示例:")
    print("-" * 40)
    for ex in wrong_examples[:3]:
        print(f"  [{ex['task']}]")
        print(f"    GT: {ex['gt']}")
        print(f"    Pred: {ex['pred']}")
        if ex['full_pred']:
            print(f"    Full output (first 300 chars): {ex['full_pred'][:300]}...")
        print()

    # 保存新的指标
    new_metrics = {
        'total': stats['total'],
        'correct': stats['correct'],
        'accuracy': accuracy,
        'by_task': dict(stats['by_task'])
    }

    output_path = f"{results_dir}/metrics_corrected.json"
    with open(output_path, 'w') as f:
        json.dump(new_metrics, f, indent=2)

    print(f"\n更正后的指标已保存到: {output_path}")

    return new_metrics


if __name__ == '__main__':
    if len(sys.argv) < 2:
        results_dir = "/nvme0/work/workspaces-zy/GraphInstruct/data/eval/results_eval_19tasks_1024tokens"
    else:
        results_dir = sys.argv[1]

    recalculate_metrics(results_dir)
