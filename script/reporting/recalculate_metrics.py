#!/usr/bin/env python3
"""
重新计算评测指标 - 使用修正后的答案比较逻辑
从detailed_results.json读取已有结果，重新评估正确性
支持两种结果格式:
1. LoRA评测格式: gt_answer, pred_answer 字段
2. 图模型评测格式: ground_truth, prediction 字段 (需要提取答案)
"""

import json
import os
import re
import argparse
from datetime import datetime
from typing import Dict, List
import glob


def extract_final_answer(text: str) -> str:
    """提取<<<...>>>中的最终答案"""
    if not text:
        return ""
    pattern = r'<<<(.+?)>>>'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def is_numeric(s: str) -> bool:
    """检查字符串是否为数值"""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def normalize_numeric(s: str, precision: int = 4) -> str:
    """标准化数值字符串，处理浮点精度问题

    Args:
        s: 数值字符串
        precision: 小数位数精度 (默认4位)

    Returns:
        标准化后的数值字符串
    """
    try:
        val = float(s)
        # 如果是整数，返回整数形式
        if val == int(val):
            return str(int(val))
        # 否则四舍五入到指定精度
        return str(round(val, precision))
    except (ValueError, TypeError):
        return s


def normalize_answer(answer: str) -> str:
    """标准化答案用于比较

    处理以下格式差异:
    - [<7>, <10>] vs [7, 10]
    - <node_0> vs 0
    - True vs true vs TRUE
    - 多余空格
    - 中英文标点
    - 数值精度差异: 0.5833 vs 0.5833333333333334
    - 整数/浮点格式: 1.0 vs 1
    """
    if not answer:
        return ""

    # 移除尖括号中的内容，保留数字/文本
    # <7> -> 7, <node_0> -> node_0
    answer = re.sub(r'<([^>]+)>', r'\1', answer)

    # 移除 node_ 前缀
    answer = re.sub(r'node_?', '', answer, flags=re.IGNORECASE)

    # 统一为小写
    answer = answer.lower()

    # 移除所有空白字符
    answer = re.sub(r'\s+', '', answer)

    # 移除方括号、圆括号、花括号
    answer = answer.replace('[', '').replace(']', '')
    answer = answer.replace('(', '').replace(')', '')
    answer = answer.replace('{', '').replace('}', '')

    # 中英文标点统一
    answer = answer.replace('，', ',').replace('。', '.').replace('：', ':')

    # 处理数值精度问题
    # 如果答案是单个数值，进行数值标准化
    if is_numeric(answer):
        answer = normalize_numeric(answer)
    else:
        # 如果是逗号分隔的列表，对每个元素进行数值标准化
        parts = answer.split(',')
        if len(parts) > 1:
            normalized_parts = []
            for part in parts:
                part = part.strip()
                if is_numeric(part):
                    normalized_parts.append(normalize_numeric(part))
                else:
                    normalized_parts.append(part)
            answer = ','.join(normalized_parts)

    return answer


def recalculate_from_detailed_results(results_dir: str) -> Dict:
    """从detailed_results.json重新计算指标"""

    detailed_path = os.path.join(results_dir, 'detailed_results.json')
    metrics_path = os.path.join(results_dir, 'metrics.json')

    if not os.path.exists(detailed_path):
        print(f"  [SKIP] No detailed_results.json found in {results_dir}")
        return None

    # 读取详细结果
    with open(detailed_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 读取原始metrics获取模型信息
    original_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            original_metrics = json.load(f)

    # 重新计算
    correct_by_task = {}
    total_correct = 0
    changes = []

    for result in results:
        task_type = result.get('task_type', 'unknown')
        old_is_correct = result.get('is_correct', False)

        # 尝试获取答案 - 支持两种格式
        # 格式1: LoRA评测结果 (gt_answer, pred_answer)
        # 格式2: 图模型评测结果 (ground_truth, prediction)
        gt_answer = result.get('gt_answer', '')
        pred_answer = result.get('pred_answer', '')

        # 如果没有gt_answer/pred_answer，从ground_truth/prediction提取
        if not gt_answer and 'ground_truth' in result:
            gt_answer = extract_final_answer(result['ground_truth'])
        if not pred_answer and 'prediction' in result:
            pred_answer = extract_final_answer(result['prediction'])

        # 初始化任务统计
        if task_type not in correct_by_task:
            correct_by_task[task_type] = {'total': 0, 'correct': 0}
        correct_by_task[task_type]['total'] += 1

        # 使用新的normalize函数重新比较
        gt_norm = normalize_answer(gt_answer)
        pred_norm = normalize_answer(pred_answer)
        new_is_correct = gt_norm == pred_norm

        # 更新结果
        result['is_correct'] = new_is_correct
        result['gt_answer'] = gt_answer
        result['pred_answer'] = pred_answer
        result['gt_normalized'] = gt_norm
        result['pred_normalized'] = pred_norm

        if new_is_correct:
            total_correct += 1
            correct_by_task[task_type]['correct'] += 1

        # 记录变化
        if old_is_correct != new_is_correct:
            changes.append({
                'id': result.get('id', 'unknown'),
                'task_type': task_type,
                'gt_answer': gt_answer,
                'pred_answer': pred_answer,
                'gt_normalized': gt_norm,
                'pred_normalized': pred_norm,
                'old_result': old_is_correct,
                'new_result': new_is_correct
            })

    # 计算新指标
    total = len(results)
    accuracy = total_correct / total if total > 0 else 0

    # 获取原始准确率
    old_accuracy = original_metrics.get('accuracy', 0)
    old_correct = original_metrics.get('correct', 0)

    new_metrics = {
        'model': original_metrics.get('model', 'Unknown'),
        'is_lora': original_metrics.get('is_lora', False),
        'total': total,
        'correct': total_correct,
        'accuracy': accuracy,
        'by_task': correct_by_task,
        'evaluation_time': original_metrics.get('evaluation_time', ''),
        'recalculated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'original_correct': old_correct,
        'original_accuracy': old_accuracy,
        'accuracy_change': accuracy - old_accuracy,
        'correct_change': total_correct - old_correct
    }

    # 保存更新后的详细结果
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 保存新指标
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(new_metrics, f, indent=2, ensure_ascii=False)

    # 保存变化记录
    if changes:
        changes_path = os.path.join(results_dir, 'answer_changes.json')
        with open(changes_path, 'w', encoding='utf-8') as f:
            json.dump(changes, f, indent=2, ensure_ascii=False)

    return {
        'results_dir': results_dir,
        'old_accuracy': old_accuracy,
        'new_accuracy': accuracy,
        'old_correct': old_correct,
        'new_correct': total_correct,
        'total': total,
        'changes_count': len(changes)
    }


def main():
    parser = argparse.ArgumentParser(description='Recalculate evaluation metrics with improved answer normalization')
    parser.add_argument('--results-dir', type=str, nargs='+',
                        help='One or more result directories to recalculate')
    parser.add_argument('--all', action='store_true',
                        help='Recalculate all results in data/eval/results_*')
    parser.add_argument('--regenerate-reports', action='store_true',
                        help='Also regenerate evaluation reports')

    args = parser.parse_args()

    results_dirs = []

    if args.all:
        # 查找所有结果目录
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        eval_dir = os.path.join(base_dir, 'data', 'eval')
        pattern = os.path.join(eval_dir, 'results_*')
        results_dirs = glob.glob(pattern)
    elif args.results_dir:
        results_dirs = args.results_dir
    else:
        print("Please specify --results-dir or --all")
        return

    print(f"Found {len(results_dirs)} result directories to process")
    print("=" * 70)

    all_summaries = []

    for results_dir in sorted(results_dirs):
        if not os.path.isdir(results_dir):
            continue

        print(f"\nProcessing: {os.path.basename(results_dir)}")

        summary = recalculate_from_detailed_results(results_dir)
        if summary:
            all_summaries.append(summary)

            change_str = ""
            if summary['changes_count'] > 0:
                acc_diff = summary['new_accuracy'] - summary['old_accuracy']
                correct_diff = summary['new_correct'] - summary['old_correct']
                change_str = f" (changes: {summary['changes_count']}, Δcorrect: {correct_diff:+d}, Δacc: {acc_diff*100:+.2f}%)"

            print(f"  Old: {summary['old_correct']}/{summary['total']} ({summary['old_accuracy']*100:.2f}%)")
            print(f"  New: {summary['new_correct']}/{summary['total']} ({summary['new_accuracy']*100:.2f}%){change_str}")

    # 打印总结
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_changes = sum(s['changes_count'] for s in all_summaries)
    print(f"\nTotal directories processed: {len(all_summaries)}")
    print(f"Total answer corrections: {total_changes}")

    if total_changes > 0:
        print("\nDirectories with changes:")
        for s in all_summaries:
            if s['changes_count'] > 0:
                acc_diff = s['new_accuracy'] - s['old_accuracy']
                print(f"  {os.path.basename(s['results_dir'])}: "
                      f"{s['old_accuracy']*100:.2f}% -> {s['new_accuracy']*100:.2f}% "
                      f"({acc_diff*100:+.2f}%)")

    # 重新生成报告
    if args.regenerate_reports:
        print("\nRegenerating evaluation reports...")
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # LoRA报告
        lora_en = os.path.join(os.path.dirname(script_dir), 'data', 'eval', 'results_lora_en')
        lora_zh = os.path.join(os.path.dirname(script_dir), 'data', 'eval', 'results_lora_zh')
        if os.path.exists(lora_en) or os.path.exists(lora_zh):
            cmd = f"python {os.path.join(script_dir, 'generate_lora_report.py')}"
            if os.path.exists(lora_en):
                cmd += f" --en-results-dir {lora_en}"
            if os.path.exists(lora_zh):
                cmd += f" --zh-results-dir {lora_zh}"
            os.system(cmd)

    print("\nDone!")


if __name__ == '__main__':
    main()
