#!/usr/bin/env python3
"""
将 graphinstruct_eval_19tasks_10samples_v3.pt 转换为文字版本的 JSON 格式。

输出格式参考: graphinstruct_eval_19tasks_10samples_text.json
每个样本包含: id, task_type, instruction, output
"""

import os
import sys
import torch
import json
import argparse
import re
from transformers import AutoTokenizer
from tqdm import tqdm


def convert_v3_to_text_json(
    input_path: str,
    output_path: str,
    tokenizer_path: str,
):
    """
    将图编码评测数据转换为纯文本JSON格式。

    Args:
        input_path: 输入的.pt文件路径
        output_path: 输出的.json文件路径
        tokenizer_path: tokenizer路径
    """
    print(f"加载评测数据: {input_path}")
    data = torch.load(input_path, weights_only=False, map_location='cpu')
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
            graph_tokens = ['<graph>', '<g_patch>', '<g_start>', '<g_end>']
            cleaned_text = input_text
            for token in graph_tokens:
                cleaned_text = cleaned_text.replace(token, '')

            # 清理多余的空白
            cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
            cleaned_text = re.sub(r'  +', ' ', cleaned_text)

            # 提取指令部分（从清理后的文本中）
            # 通常格式为: <|im_start|>system...user...question<|im_end|>
            # 我们需要提取 user 部分的内容作为 instruction

            # 尝试提取用户消息部分
            user_pattern = r'<\|im_start\|>user\s*(.*?)<\|im_end\|>'
            user_match = re.search(user_pattern, cleaned_text, re.DOTALL)
            if user_match:
                instruction = user_match.group(1).strip()
            else:
                # 如果没有找到user标记，尝试其他模式或使用整个文本
                # 移除特殊标记
                instruction = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', cleaned_text)
                instruction = instruction.strip()

            # 如果指令为空，使用清理后的全文
            if not instruction:
                instruction = cleaned_text.strip()

            # 清理答案文本中的特殊标记
            output = answer_text.strip()
            # 移除可能的assistant标记
            output = re.sub(r'<\|im_start\|>assistant\s*', '', output)
            output = re.sub(r'<\|im_end\|>', '', output)
            output = output.strip()

            # 创建新的样本（JSON格式）
            new_sample = {
                'id': sample_id,
                'task_type': task_type,
                'instruction': instruction,
                'output': output,
            }

            converted_data.append(new_sample)

        except Exception as e:
            print(f"\n跳过样本 {idx} (ID: {sample.get('id', 'unknown')}): {e}")
            skipped_count += 1
            continue

    print(f"\n转换完成:")
    print(f"  成功转换: {len(converted_data)} 样本")
    print(f"  跳过: {skipped_count} 样本")

    # 保存为JSON
    print(f"保存到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    # 打印统计信息
    task_counts = {}
    for sample in converted_data:
        task = sample['task_type']
        task_counts[task] = task_counts.get(task, 0) + 1

    print(f"\n任务分布:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")

    # 打印示例
    if converted_data:
        print(f"\n第一个样本示例:")
        print(f"  ID: {converted_data[0]['id']}")
        print(f"  Task: {converted_data[0]['task_type']}")
        print(f"  Instruction (前200字符): {converted_data[0]['instruction'][:200]}")
        print(f"  Output (前100字符): {converted_data[0]['output'][:100]}")

    return converted_data


def main():
    parser = argparse.ArgumentParser(description="将v3.pt转换为文字版JSON格式")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="/nvme0/work/workspaces-zy/GraphInstruct/data/eval/graphinstruct_eval_19tasks_10samples_v3.pt",
        help="输入的.pt文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="/nvme0/work/workspaces-zy/GraphInstruct/data/eval/graphinstruct_eval_19tasks_10samples_v3_text.json",
        help="输出的.json文件路径"
    )
    parser.add_argument(
        "--tokenizer", "-t",
        type=str,
        default="/mnt/yrfs/GraphInstruct/model/qwen3-4b-graph-reasoning-merged",
        help="Tokenizer路径"
    )

    args = parser.parse_args()

    convert_v3_to_text_json(
        input_path=args.input,
        output_path=args.output,
        tokenizer_path=args.tokenizer,
    )


if __name__ == "__main__":
    main()
