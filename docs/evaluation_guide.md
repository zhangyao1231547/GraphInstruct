# GraphAgent 评测指南

本文档介绍如何使用 GraphAgent 评测系统进行模型评估，包括图增强评测和纯文本基线评测。

## 目录

1. [概述](#概述)
2. [评测类型](#评测类型)
3. [快速开始](#快速开始)
4. [详细使用说明](#详细使用说明)
5. [评测数据准备](#评测数据准备)
6. [评测报告生成](#评测报告生成)
7. [评测结果解读](#评测结果解读)
8. [常见问题](#常见问题)

---

## 概述

GraphAgent 评测系统支持两种评测模式：

| 评测模式 | 说明 | 输入 |
|----------|------|------|
| **图增强评测** | 使用图编码器 + LLM 联合推理 | 图结构 + 文本 |
| **纯文本评测** | 仅使用文本描述，无图编码 | 自然语言图描述 |

### 核心组件

| 组件 | 路径 | 功能 |
|------|------|------|
| 图评测脚本 | `GraphAGent-training/pl_eval.py` | GraphAgent 模型评测 |
| 文本评测脚本 | `GraphInstruct/script/run_text_only_evaluation.sh` | 纯文本基线评测 |
| 评测数据生成 | `GraphInstruct/script/create_eval_data.py` | 生成评测数据集 |
| 报告生成器 | `GraphInstruct/script/generate_graphagent_report.py` | 生成评测报告 |
| 文本报告生成 | `GraphInstruct/script/generate_text_only_report.py` | 生成文本评测报告 |

---

## 评测类型

### 1. GraphAgent 图增强评测

使用预训练的图编码器将图结构编码为向量，与 LLM 联合推理。

**特点**：
- 输入包含 `<g_start>...<g_patch>...<g_end>` 图嵌入标记
- 使用 MetaHGT/GraphGPT 等图编码器
- 支持复杂图结构理解

### 2. 纯文本基线评测

将图结构转换为自然语言描述，仅使用 LLM 进行推理。

**特点**：
- 输入为纯文本图描述（如 "节点 0 连接到节点 1, 2"）
- 作为基线对比 GraphAgent 的图编码效果
- 可使用任意 LLM 模型

---

## 快速开始

### 1. GraphAgent 图增强评测

```bash
cd /nvme0/work/workspaces-zy/GraphAgent-zy/GraphAGent-training

# 运行评测 (max_new_tokens=1024)
CUDA_VISIBLE_DEVICES=0 python pl_eval.py \
    --version qwen \
    --model_name_or_path /nvme0/work/workspaces-zy/model/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507 \
    --checkpoint_path "/mnt/yrfs/GraphAgent_model/zy/model/graphagent-qwen3/stage2-epoch5-8192-graphinstruct-full_finetune/lightning_logs/version_0/model_epoch=3-step=23748.ckpt/pytorch_model.bin" \
    --eval_data_path /nvme0/work/workspaces-zy/GraphInstruct/data/eval/graphinstruct_eval_19tasks_10samples.pt \
    --output_dir /nvme0/work/workspaces-zy/GraphInstruct/data/eval/results_eval_19tasks_1024tokens \
    --batch_size 1 \
    --max_new_tokens 1024 \
    --bf16 True \
    --model_max_length 4096 \
    --use_graph_start_end True \
    --gpus 0
```

### 2. 纯文本基线评测

```bash
cd /nvme0/work/workspaces-zy/GraphInstruct

# 运行纯文本评测
bash script/run_text_only_evaluation.sh
```

### 3. 生成评测报告

```bash
# GraphAgent 评测报告
python script/generate_graphagent_report.py \
    --results-dir data/eval/results_eval_19tasks_1024tokens \
    --max-new-tokens 1024

# 纯文本评测报告
python script/generate_text_only_report.py \
    --results-dir data/eval/results_text_only
```

---

## 详细使用说明

### GraphAgent 评测参数

```bash
python pl_eval.py [OPTIONS]
```

| 参数 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `--version` | str | 是 | 模型版本: `qwen`, `llama` |
| `--model_name_or_path` | str | 是 | 基座模型路径 |
| `--checkpoint_path` | str | 是 | GraphAgent 检查点路径 |
| `--eval_data_path` | str | 是 | 评测数据 `.pt` 文件路径 |
| `--output_dir` | str | 是 | 评测结果输出目录 |
| `--batch_size` | int | 否 | 批大小，默认 1 |
| `--max_new_tokens` | int | 否 | 最大生成 token 数，默认 512 |
| `--bf16` | bool | 否 | 使用 bfloat16，默认 True |
| `--model_max_length` | int | 否 | 模型最大长度，默认 4096 |
| `--use_graph_start_end` | bool | 否 | 使用图起止标记，默认 True |
| `--gpus` | int | 否 | GPU 设备 ID |
| `--verbose` | bool | 否 | 详细输出模式 |

### max_new_tokens 参数说明

`max_new_tokens` 控制模型生成的最大 token 数量：

| 值 | 适用场景 | 说明 |
|----|----------|------|
| 512 | 简单任务 | degree, edge, neighbor 等 |
| 1024 | 复杂任务 | BFS, DFS, shortest_path 等需要详细推理步骤 |
| 2048 | 超长输出 | 大规模图的遍历序列 |

**建议**: 对于 BFS/DFS 等遍历任务，建议使用 `max_new_tokens=1024` 或更高。

---

## 评测数据准备

### 评测数据格式

评测数据为 `.pt` 格式的 PyTorch 文件，包含以下字段：

```python
{
    "id": "BFS_0",                    # 样本ID
    "input_ids": tensor([...]),       # tokenize 后的输入
    "labels": tensor([...]),          # 标签 (用于计算loss)
    "graph_data": HeteroData(...),    # PyG 图数据
    "hetero_key_order": ["node"],     # 异构图键顺序
    "task_type": "BFS",               # 任务类型
    "ground_truth": "[<0>, <1>, ...]" # 标准答案
}
```

### 生成评测数据

```bash
cd /nvme0/work/workspaces-zy/GraphInstruct

# 生成 19 个任务各 10 个样本的评测数据
python script/create_eval_data.py \
    --tasks all \
    --samples-per-task 10 \
    --output data/eval/graphinstruct_eval_19tasks_10samples.pt
```

### 评测数据位置

| 数据集 | 路径 | 说明 |
|--------|------|------|
| 标准评测集 | `data/eval/graphinstruct_eval_19tasks_10samples.pt` | 19任务×10样本 |
| 大规模评测 | `data/eval/graphinstruct_eval_19tasks_100samples.pt` | 19任务×100样本 |

---

## 评测报告生成

### GraphAgent 评测报告

```bash
python script/generate_graphagent_report.py \
    --results-dir <评测结果目录> \
    --output-report <报告输出路径> \
    --max-new-tokens <使用的max_new_tokens值>
```

**参数说明**:

| 参数 | 说明 |
|------|------|
| `--results-dir` | 包含 `detailed_results.json` 的目录 |
| `--output-report` | 报告输出路径，默认为 `results-dir/evaluation_report.md` |
| `--max-new-tokens` | 评测时使用的 max_new_tokens 值，用于报告标题 |

**生成的文件**:
- `evaluation_report.md` - Markdown 格式评测报告
- `metrics_corrected.json` - 使用答案提取后的准确指标

### 纯文本评测报告

```bash
python script/generate_text_only_report.py \
    --results-dir <评测结果目录>
```

---

## 评测结果解读

### 输出文件结构

评测完成后，输出目录包含：

```
results_eval_19tasks_1024tokens/
├── detailed_results.json      # 每个样本的详细预测结果
├── metrics.json               # 原始指标统计
├── metrics_corrected.json     # 答案提取后的指标 (更准确)
└── evaluation_report.md       # Markdown 格式报告
```

### detailed_results.json 格式

```json
[
    {
        "id": "BFS_0",
        "task_type": "BFS",
        "ground_truth": "[<0>, <1>, <2>, ...]",
        "prediction": "Let's run BFS step by step...\n<<<[<0>, <1>, <2>, ...]>>>",
        "input_text": "...",
        "correct": true
    },
    ...
]
```

### 答案提取机制

评测使用 `<<<答案>>>` 格式提取最终答案：

```
模型输出: "Let's solve it step by step...\nThe result is <<<[<0>, <1>, <2>]>>>"
提取答案: "[<0>, <1>, <2>]"
```

如果没有 `<<<>>>` 标记，则使用整个输出进行匹配。

### metrics_corrected.json 格式

```json
{
    "total": 190,
    "correct": 42,
    "accuracy": 0.2211,
    "by_task": {
        "BFS": {"total": 10, "correct": 0},
        "DFS": {"total": 10, "correct": 0},
        "degree": {"total": 10, "correct": 9},
        "edge": {"total": 10, "correct": 9},
        ...
    }
}
```

### 评测报告内容

生成的 Markdown 报告包含：

| 章节 | 内容 |
|------|------|
| 总体概述 | 模型名称、总样本数、正确数、准确率 |
| 各任务评测结果 | 按准确率排序的任务表格 |
| 任务难度分析 | 简单(≥70%)、中等(30-70%)、困难(<30%)任务分类 |
| 预测示例 | 正确和错误预测的具体示例 |
| 评测配置 | 模型、编码器、参数等配置信息 |
| 结论 | 最佳/最差任务总结 |

---

## 最新评测结果

### GraphAgent (max_new_tokens=1024)

| 指标 | 值 |
|------|-----|
| 总样本数 | 190 |
| 正确数 | 42 |
| 准确率 | **22.11%** |

**各任务准确率**:

| 任务类型 | 准确率 | 难度 |
|----------|--------|------|
| degree | 90.0% | 简单 |
| edge | 90.0% | 简单 |
| cycle | 60.0% | 中等 |
| neighbor | 60.0% | 中等 |
| connectivity | 20.0% | 困难 |
| diameter | 20.0% | 困难 |
| jaccard | 20.0% | 困难 |
| BFS | 0.0% | 困难 |
| DFS | 0.0% | 困难 |
| topological_sort | 0.0% | 困难 |

### 纯文本基线

| 指标 | 值 |
|------|-----|
| 总样本数 | 190 |
| 正确数 | 36 |
| 准确率 | **18.95%** |

### 对比分析

| 任务 | GraphAgent | 纯文本 | 差异 |
|------|------------|--------|------|
| degree | 90% | 80% | +10% |
| edge | 90% | 90% | 0% |
| cycle | 60% | 50% | +10% |
| neighbor | 60% | 50% | +10% |
| **平均** | **22.11%** | **18.95%** | **+3.16%** |

GraphAgent 图增强模型在大多数任务上优于纯文本基线。

---

## 完整评测流程

### 端到端评测示例

```bash
# 1. 准备评测数据 (如果需要)
cd /nvme0/work/workspaces-zy/GraphInstruct
python script/create_eval_data.py \
    --tasks all \
    --samples-per-task 10 \
    --output data/eval/graphinstruct_eval_19tasks_10samples.pt

# 2. 运行 GraphAgent 评测
cd /nvme0/work/workspaces-zy/GraphAgent-zy/GraphAGent-training
OUTPUT_DIR="/nvme0/work/workspaces-zy/GraphInstruct/data/eval/results_graphagent_1024"
CUDA_VISIBLE_DEVICES=0 python pl_eval.py \
    --version qwen \
    --model_name_or_path /nvme0/work/workspaces-zy/model/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507 \
    --checkpoint_path "/mnt/yrfs/GraphAgent_model/zy/model/graphagent-qwen3/stage2-epoch5-8192-graphinstruct-full_finetune/lightning_logs/version_0/model_epoch=3-step=23748.ckpt/pytorch_model.bin" \
    --eval_data_path /nvme0/work/workspaces-zy/GraphInstruct/data/eval/graphinstruct_eval_19tasks_10samples.pt \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 1 \
    --max_new_tokens 1024 \
    --bf16 True

# 3. 生成评测报告
cd /nvme0/work/workspaces-zy/GraphInstruct
python script/generate_graphagent_report.py \
    --results-dir "$OUTPUT_DIR" \
    --max-new-tokens 1024

# 4. 查看报告
cat "$OUTPUT_DIR/evaluation_report.md"
```

### 批量评测不同 max_new_tokens

```bash
for tokens in 512 1024 2048; do
    OUTPUT_DIR="data/eval/results_eval_${tokens}tokens"

    CUDA_VISIBLE_DEVICES=0 python pl_eval.py \
        --version qwen \
        --model_name_or_path ... \
        --checkpoint_path ... \
        --eval_data_path ... \
        --output_dir "$OUTPUT_DIR" \
        --max_new_tokens $tokens

    python script/generate_graphagent_report.py \
        --results-dir "$OUTPUT_DIR" \
        --max-new-tokens $tokens
done
```

---

## 常见问题

### Q1: 为什么 BFS/DFS 任务准确率为 0%？

**原因**: BFS/DFS 需要输出完整的遍历序列，即使中间步骤正确，最终答案也可能因为顺序问题而被判错。

**解决方案**:
1. 增加 `max_new_tokens` 到 1024 或更高
2. 检查模型是否正确学习了答案格式 `<<<答案>>>`
3. 考虑使用更宽松的评估标准（如节点集合匹配）

### Q2: 如何对比不同检查点的性能？

```bash
# 评测检查点 A
python pl_eval.py --checkpoint_path /path/to/checkpoint_A.bin --output_dir results_A

# 评测检查点 B
python pl_eval.py --checkpoint_path /path/to/checkpoint_B.bin --output_dir results_B

# 对比报告
python script/compare_evaluations.py --dirs results_A results_B
```

### Q3: 评测太慢怎么办？

1. **减少样本数**: 使用 `--samples-per-task 5` 生成更小的评测集
2. **使用多 GPU**: 修改 `--gpus` 参数
3. **降低 max_new_tokens**: 对于简单任务使用较小的值

### Q4: 如何添加新的评测任务？

1. 在 `GTG/tasks/` 下创建新任务目录和实现
2. 更新 `script/create_eval_data.py` 中的任务列表
3. 重新生成评测数据集

### Q5: 答案提取失败怎么办？

检查模型输出是否包含 `<<<>>>` 标记。如果没有，修改训练数据确保答案格式正确，或修改评测脚本使用其他提取方式。

---

## 文件结构

```
GraphInstruct/
├── data/
│   └── eval/
│       ├── graphinstruct_eval_19tasks_10samples.pt    # 评测数据
│       ├── results_eval_19tasks_1024tokens/           # 评测结果
│       │   ├── detailed_results.json
│       │   ├── metrics_corrected.json
│       │   └── evaluation_report.md
│       └── results_text_only/                         # 文本评测结果
├── script/
│   ├── create_eval_data.py                # 生成评测数据
│   ├── run_text_only_evaluation.sh        # 文本评测脚本
│   ├── generate_graphagent_report.py      # GraphAgent 报告生成
│   └── generate_text_only_report.py       # 文本评测报告生成
└── docs/
    └── evaluation_guide.md                # 本文档

GraphAgent-zy/
└── GraphAGent-training/
    └── pl_eval.py                         # GraphAgent 评测主脚本
```

---

## 更新日志

- **2025-12-18**:
  - 完成 19 任务评测，GraphAgent 准确率 22.11%
  - 增加 max_new_tokens=1024 评测
  - 添加纯文本基线对比（18.95%）
  - 创建评测报告生成脚本

---

*本文档由 GraphAgent 评测系统维护*
