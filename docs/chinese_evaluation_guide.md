# GraphAgent 中文评测指南

本文档介绍如何使用 GraphInstruct 中文评测系统进行模型评估，包括中文图增强评测和中文纯文本基线评测。

## 目录

1. [概述](#概述)
2. [快速开始](#快速开始)
3. [数据生成](#数据生成)
4. [图模型评测](#图模型评测)
5. [纯文本评测](#纯文本评测)
6. [报告生成](#报告生成)
7. [与英文评测对比](#与英文评测对比)
8. [常见问题](#常见问题)

---

## 概述

中文评测系统支持两种评测模式：

| 评测模式 | 说明 | 输入 | 输出目录 |
|----------|------|------|----------|
| **图增强评测** | GraphAgent 使用图编码器 + LLM 联合推理 | `.pt` 文件 | `results_zh_graph_*` |
| **纯文本评测** | 仅使用中文文本描述，无图编码 | `.json` 文件 | `results_zh_text_only` |

### 核心脚本

| 脚本 | 路径 | 功能 |
|------|------|------|
| 数据生成 | `script/create_chinese_eval_data.py` | 生成中文评测数据 |
| 一键评测 | `script/run_chinese_evaluation.sh` | 完整中文评测流程 |
| 报告生成 | `script/generate_graphagent_report.py` | 生成评测报告 |

### 支持的任务类型

19种图推理任务全部支持中文：

| 类别 | 任务 |
|------|------|
| 遍历算法 | BFS(广度优先搜索), DFS(深度优先搜索) |
| 路径算法 | shortest_path(最短路径), predecessor(前驱节点) |
| 连通性 | connectivity(连通性判断), connected_component(连通分量) |
| 图属性 | degree(度数), neighbor(邻居), edge(边查询) |
| 图结构 | cycle(环检测), bipartite(二分图), diameter(直径) |
| 排序算法 | topological_sort(拓扑排序), page_rank(PageRank) |
| 相似度 | common_neighbor(公共邻居), jaccard(Jaccard系数), clustering_coefficient(聚类系数) |
| 优化算法 | MST(最小生成树), maximum_flow(最大流) |

---

## 快速开始

### 一键运行完整评测

```bash
cd /nvme0/work/workspaces-zy/GraphInstruct

# 完整流程：生成数据 + 图模型评测 + 文本评测 + 生成报告
bash script/run_chinese_evaluation.sh
```

### 分步运行

```bash
# 只生成数据
bash script/run_chinese_evaluation.sh --generate-only

# 只运行图模型评测
bash script/run_chinese_evaluation.sh --graph-only

# 只运行文本评测
bash script/run_chinese_evaluation.sh --text-only

# 只生成报告
bash script/run_chinese_evaluation.sh --report-only
```

### 自定义参数

```bash
# 自定义样本数和生成token数
bash script/run_chinese_evaluation.sh --samples 20 --max-tokens 2048 --gpu 0
```

---

## 数据生成

### 生成中文评测数据

```bash
cd /nvme0/work/workspaces-zy/GraphInstruct

# 同时生成图模型(.pt)和文本(.json)格式
python script/create_chinese_eval_data.py \
    --output-dir data/eval \
    --samples-per-task 10 \
    --format both

# 只生成图模型格式
python script/create_chinese_eval_data.py \
    --format pt \
    --samples-per-task 10

# 只生成文本格式
python script/create_chinese_eval_data.py \
    --format text \
    --samples-per-task 10
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output-dir` | str | `data/eval` | 输出目录 |
| `--model-path` | str | Qwen3路径 | Qwen模型路径(用于tokenizer) |
| `--samples-per-task` | int | `10` | 每个任务的样本数 |
| `--format` | str | `both` | 输出格式: `pt`, `text`, `both` |
| `--max-length` | int | `4096` | 最大序列长度 |
| `--seed` | int | `42` | 随机种子 |
| `--no-graph-encoding` | flag | - | 禁用图编码(使用随机特征) |

### 生成的文件

```
data/eval/
├── graphinstruct_zh_19tasks_10samples.pt      # 图模型评测数据
└── graphinstruct_zh_19tasks_10samples_text.json  # 纯文本评测数据
```

---

## 图模型评测

### 运行 GraphAgent 图模型评测

```bash
cd /nvme0/work/workspaces-zy/GraphAgent-zy/GraphAGent-training

CUDA_VISIBLE_DEVICES=0 python pl_eval.py \
    --version qwen \
    --model_name_or_path /nvme0/work/workspaces-zy/model/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507 \
    --checkpoint_path "/mnt/yrfs/GraphAgent_model/zy/model/graphagent-qwen3/stage2-epoch5-8192-graphinstruct-full_finetune/lightning_logs/version_0/model_epoch=3-step=23748.ckpt/pytorch_model.bin" \
    --eval_data_path /nvme0/work/workspaces-zy/GraphInstruct/data/eval/graphinstruct_zh_19tasks_10samples.pt \
    --output_dir /nvme0/work/workspaces-zy/GraphInstruct/data/eval/results_zh_graph_1024tokens \
    --batch_size 1 \
    --max_new_tokens 1024 \
    --bf16 True
```

### 评测参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--max_new_tokens` | 最大生成token数 | BFS/DFS等遍历任务建议1024+ |
| `--batch_size` | 批大小 | 1 (显存有限时) |
| `--bf16` | 使用bfloat16 | True |

---

## 纯文本评测

### 运行纯文本LLM评测

```bash
cd /nvme0/work/workspaces-zy/GraphInstruct

CUDA_VISIBLE_DEVICES=0 python script/evaluate_text_only.py \
    --model-path /nvme0/work/workspaces-zy/model/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507 \
    --eval-data-path data/eval/graphinstruct_zh_19tasks_10samples_text.json \
    --output-dir data/eval/results_zh_text_only \
    --max-new-tokens 1024 \
    --bf16
```

---

## 报告生成

### 生成评测报告

```bash
cd /nvme0/work/workspaces-zy/GraphInstruct

# GraphAgent 图模型评测报告
python script/generate_graphagent_report.py \
    --results-dir data/eval/results_zh_graph_1024tokens \
    --max-new-tokens 1024 \
    --language zh

# 纯文本评测报告
python script/generate_text_only_report.py \
    --results-dir data/eval/results_zh_text_only \
    --language zh
```

### 报告内容

生成的 `evaluation_report.md` 包含：

| 章节 | 内容 |
|------|------|
| 总体概述 | 模型信息、总样本数、准确率 |
| 任务结果表格 | 19个任务的详细准确率 |
| 难度分析 | 简单/中等/困难任务分类 |
| 预测示例 | 正确和错误预测的具体示例 |
| 配置信息 | 评测参数配置 |
| 结论 | 最佳/最差任务总结 |

---

## 与英文评测对比

### 数据对比

| 数据集 | 语言 | 文件 |
|--------|------|------|
| 英文图模型 | en | `graphinstruct_eval_19tasks_10samples.pt` |
| 英文纯文本 | en | `graphinstruct_eval_19tasks_10samples_text.json` |
| 中文图模型 | zh | `graphinstruct_zh_19tasks_10samples.pt` |
| 中文纯文本 | zh | `graphinstruct_zh_19tasks_10samples_text.json` |

### 输出示例对比

**英文 BFS 问题**:
```
Node 0 is connected to Node 1, 2.
Node 1 is connected to Node 0, 3.
...
Starting from Node 5, output the BFS traversal sequence.
```

**中文 BFS 问题**:
```
节点 <0> 连接到节点 <1>, <2>。
节点 <1> 连接到节点 <0>, <3>。
...
从节点 <5> 开始，输出广度优先搜索 (BFS) 遍历序列。
```

---

## 文件结构

```
GraphInstruct/
├── data/
│   └── eval/
│       ├── graphinstruct_zh_19tasks_10samples.pt       # 中文图评测数据
│       ├── graphinstruct_zh_19tasks_10samples_text.json # 中文文本评测数据
│       ├── results_zh_graph_1024tokens/                # 中文图评测结果
│       │   ├── detailed_results.json
│       │   ├── metrics_corrected.json
│       │   └── evaluation_report.md
│       └── results_zh_text_only/                       # 中文文本评测结果
├── script/
│   ├── create_chinese_eval_data.py      # 中文数据生成
│   ├── run_chinese_evaluation.sh        # 一键评测脚本
│   ├── generate_graphagent_report.py    # 报告生成
│   └── evaluate_text_only.py            # 纯文本评测
└── docs/
    ├── chinese_evaluation_guide.md      # 本文档
    └── evaluation_guide.md              # 英文评测指南
```

---

## 常见问题

### Q1: 中文评测和英文评测的主要区别是什么？

**语言差异**:
- 问题和推理步骤使用中文
- 图描述使用中文格式 (如 "节点 <0> 连接到节点 <1>")
- 答案格式保持一致 (`<<<答案>>>`)

**预期表现**:
- 模型经过中英文数据混合训练后，两种语言表现应接近
- 某些任务可能因中文token化差异而有细微差别

### Q2: 如何只评测特定任务？

目前的评测脚本评测所有19个任务。如需只评测特定任务，可以：

1. 使用 `create_chinese_eval_data.py` 生成特定任务的数据：
   ```python
   # 在脚本中修改 ALL_TASKS 列表
   ALL_TASKS = ['BFS', 'DFS']  # 只生成 BFS 和 DFS
   ```

2. 或者在分析报告时过滤特定任务的结果

### Q3: 评测太慢怎么办？

1. **减少样本数**: `--samples-per-task 5`
2. **使用多GPU**: 修改 `--gpus` 参数
3. **降低 max_new_tokens**: 简单任务可用 512

### Q4: 中文数据生成失败怎么办？

检查多语言任务文件是否正确创建：
```bash
ls GTG/tasks/*/\*_multilang.py | wc -l  # 应该有 19 个
```

如果少于19个，运行修复脚本：
```bash
python script/fix_multilang_tasks.py
```

### Q5: 如何对比中英文评测结果？

评测完成后，比较两个报告中的 `metrics_corrected.json`：

```python
import json

# 加载中英文结果
with open('data/eval/results_eval_19tasks_1024tokens/metrics_corrected.json') as f:
    en_metrics = json.load(f)
with open('data/eval/results_zh_graph_1024tokens/metrics_corrected.json') as f:
    zh_metrics = json.load(f)

# 对比
print(f"英文准确率: {en_metrics['accuracy']*100:.2f}%")
print(f"中文准确率: {zh_metrics['accuracy']*100:.2f}%")

# 任务对比
for task in en_metrics['by_task']:
    en_acc = en_metrics['by_task'][task]['correct'] / en_metrics['by_task'][task]['total']
    zh_acc = zh_metrics['by_task'][task]['correct'] / zh_metrics['by_task'][task]['total']
    print(f"{task}: 英文 {en_acc*100:.1f}% vs 中文 {zh_acc*100:.1f}%")
```

---

## 更新日志

- **2025-12-18**:
  - 创建中文评测数据生成脚本
  - 添加一键评测脚本
  - 完成 19 任务中文评测数据生成
  - 创建中文评测使用文档

---

*本文档由 GraphAgent 中文评测系统维护*
