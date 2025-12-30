#!/usr/bin/env python3
"""
将GraphInstruct评测数据从图编码格式转换为纯文本格式。

用于评测纯文本LLM（不使用图编码器）的性能。
"""

import os
import sys
import torch
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm


def convert_eval_data_to_text(
    input_path: str,
    output_path: str,
    tokenizer_path: str,
    max_length: int = 8192
):
    """
    将图编码评测数据转换为纯文本格式。

    Args:
        input_path: 输入的.pt文件路径
        output_path: 输出的.pt文件路径
        tokenizer_path: tokenizer路径
        max_length: 最大序列长度
    """
    print(f"加载评测数据: {input_path}")
    data = torch.load(input_path, weights_only=False)
    print(f"原始样本数: {len(data)}")

    print(f"加载tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 设置pad_token
    if tokenizer.pad_token is None:
        eos_token = "<|im_end|>"
        eos_id = tokenizer.convert_tokens_to_ids(eos_token)
        if eos_id is None or eos_id == tokenizer.unk_token_id:
            eos_token = "<|endoftext|>"
            eos_id = tokenizer.convert_tokens_to_ids(eos_token)
        tokenizer.pad_token = eos_token
        tokenizer.pad_token_id = eos_id

    converted_data = []
    skipped_count = 0

    for idx, sample in enumerate(tqdm(data, desc="转换中")):
        try:
            # 获取原始数据
            input_ids = sample['input_ids']
            labels = sample['labels']
            task_type = sample.get('task_type', 'unknown')
            sample_id = sample.get('id', idx)

            # 解码input_ids为文本
            valid_input_ids = input_ids.clone()
            valid_input_ids[valid_input_ids < 0] = tokenizer.pad_token_id

            # 解码为文本
            input_text = tokenizer.decode(valid_input_ids, skip_special_tokens=False)

            # 解码labels为文本（只取非-100的部分作为答案）
            answer_mask = labels != -100
            answer_ids = labels[answer_mask]
            if len(answer_ids) > 0:
                answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
            else:
                answer_text = ""

            # 移除图相关的特殊token
            # <graph>, <g_patch>, <g_start>, <g_end> 等
            graph_tokens = ['<graph>', '<g_patch>', '<g_start>', '<g_end>']
            cleaned_text = input_text
            for token in graph_tokens:
                cleaned_text = cleaned_text.replace(token, '')

            # 清理多余的空白
            import re
            cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
            cleaned_text = re.sub(r'  +', ' ', cleaned_text)

            # 重新tokenize清理后的文本
            new_encoding = tokenizer(
                cleaned_text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding=False
            )
            new_input_ids = new_encoding['input_ids'].squeeze(0)

            # 创建新的labels（-100表示不计算loss的部分）
            # 找到答案开始的位置
            new_labels = torch.full_like(new_input_ids, -100)

            # 对于评测，我们需要找到答案在新input_ids中的位置
            # 简化处理：将原始answer部分重新tokenize并附加到末尾
            if answer_text:
                answer_encoding = tokenizer(
                    answer_text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=1024,
                    padding=False,
                    add_special_tokens=False
                )
                answer_token_ids = answer_encoding['input_ids'].squeeze(0)

                # 尝试在new_input_ids中找到答案的起始位置
                # 使用滑动窗口匹配
                answer_start = -1
                answer_len = len(answer_token_ids)
                if answer_len > 0 and answer_len < len(new_input_ids):
                    for i in range(len(new_input_ids) - answer_len + 1):
                        if torch.equal(new_input_ids[i:i+answer_len], answer_token_ids):
                            answer_start = i
                            break

                if answer_start >= 0:
                    # 找到了答案位置，设置labels
                    new_labels[answer_start:answer_start+answer_len] = new_input_ids[answer_start:answer_start+answer_len]
                else:
                    # 没找到完全匹配，使用原始labels的相对位置
                    # 计算原始答案在原始序列中的比例位置
                    orig_answer_start = answer_mask.nonzero(as_tuple=True)[0][0].item() if answer_mask.any() else -1
                    if orig_answer_start >= 0:
                        ratio = orig_answer_start / len(input_ids)
                        new_answer_start = int(ratio * len(new_input_ids))
                        new_answer_start = max(0, min(new_answer_start, len(new_input_ids) - 1))
                        # 设置从这个位置到结尾都是答案
                        new_labels[new_answer_start:] = new_input_ids[new_answer_start:]

            # 创建新的样本（不包含图数据）
            new_sample = {
                'id': sample_id,
                'input_ids': new_input_ids,
                'labels': new_labels,
                'task_type': task_type,
                # 保留空的图数据占位符，以兼容评测脚本
                'graph_data': None,
                'hetero_key_order': None,
                # 额外保存文本用于调试
                'input_text': cleaned_text,
                'answer_text': answer_text,
            }

            converted_data.append(new_sample)

        except Exception as e:
            print(f"跳过样本 {idx}: {e}")
            skipped_count += 1
            continue

    print(f"\n转换完成:")
    print(f"  成功转换: {len(converted_data)} 样本")
    print(f"  跳过: {skipped_count} 样本")

    # 保存
    print(f"保存到: {output_path}")
    torch.save(converted_data, output_path)

    # 打印统计信息
    task_counts = {}
    for sample in converted_data:
        task = sample['task_type']
        task_counts[task] = task_counts.get(task, 0) + 1

    print(f"\n任务分布:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")

    return converted_data


def main():
    parser = argparse.ArgumentParser(description="将图编码评测数据转换为纯文本格式")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="/nvme0/work/workspaces-zy/GraphInstruct/data/eval/graphinstruct_eval_19tasks_10samples_v3.pt",
        help="输入的.pt文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="/nvme0/work/workspaces-zy/GraphInstruct/data/eval/graphinstruct_eval_19tasks_10samples_v3_text.pt",
        help="输出的.pt文件路径"
    )
    parser.add_argument(
        "--tokenizer", "-t",
        type=str,
        default="/mnt/yrfs/GraphInstruct/model/qwen3-4b-graph-reasoning-merged",
        help="Tokenizer路径"
    )
    parser.add_argument(
        "--max-length", "-m",
        type=int,
        default=8192,
        help="最大序列长度"
    )

    args = parser.parse_args()

    convert_eval_data_to_text(
        input_path=args.input,
        output_path=args.output,
        tokenizer_path=args.tokenizer,
        max_length=args.max_length
    )


if __name__ == "__main__":
    main()
