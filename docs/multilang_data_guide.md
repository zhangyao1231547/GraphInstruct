# GraphInstruct 多语言数据生成指南

本文档介绍如何使用 GraphInstruct 的多语言功能生成中英文图推理训练数据，并转换为 GraphAgent 训练格式。

## 目录

1. [概述](#概述)
2. [支持的任务类型](#支持的任务类型)
3. [快速开始](#快速开始)
4. [详细使用说明](#详细使用说明)
5. [数据格式说明](#数据格式说明)
6. [API 参考](#api-参考)
7. [常见问题](#常见问题)

---

## 概述

GraphInstruct 多语言模块支持生成中文和英文版本的图推理数据，涵盖 19 种图算法任务。

### 核心组件

| 组件 | 路径 | 功能 |
|------|------|------|
| 语言模板 | `GTG/utils/language.py` | 中英文模板定义 |
| 多语言任务 | `GTG/tasks/*/[task]_multilang.py` | 各任务的多语言生成器 |
| 数据生成脚本 | `script/generate_multilang_data.py` | 批量生成多语言数据 |
| 格式转换脚本 | `script/convert_multilang_to_training_format.py` | 转换为训练格式 |

---

## 支持的任务类型

所有 19 种图推理任务均支持多语言生成：

| 类别 | 任务 | 说明 |
|------|------|------|
| **遍历算法** | BFS, DFS | 广度/深度优先搜索 |
| **路径算法** | shortest_path, predecessor | 最短路径、前驱节点 |
| **连通性** | connectivity, connected_component | 连通性判断、连通分量 |
| **图属性** | degree, neighbor, edge | 度数、邻居、边查询 |
| **图结构** | cycle, bipartite, diameter | 环检测、二分图、直径 |
| **排序算法** | topological_sort, page_rank | 拓扑排序、PageRank |
| **相似度** | common_neighbor, jaccard, clustering_coefficient | 公共邻居、Jaccard系数、聚类系数 |
| **优化算法** | MST, maximum_flow | 最小生成树、最大流 |

---

## 快速开始

### 1. 生成中文数据

```bash
cd /nvme0/work/workspaces-zy/GraphInstruct

# 生成所有19个任务的中文数据，每个任务100个样本
python script/generate_multilang_data.py \
    --lang zh \
    --samples-per-task 100 \
    --output-dir data/generated
```

### 2. 转换为训练格式

```bash
# 转换为 GraphAgent 训练格式
python script/convert_multilang_to_training_format.py \
    --input data/generated/graphinstruct_zh_19tasks_100samples.json \
    --output data/training/graphinstruct_zh_training.json \
    --verify
```

### 3. 生成英文数据

```bash
python script/generate_multilang_data.py \
    --lang en \
    --samples-per-task 100 \
    --output-dir data/generated
```

---

## 详细使用说明

### 数据生成脚本参数

```bash
python script/generate_multilang_data.py [OPTIONS]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lang` | str | `zh` | 语言代码: `en`(英文) 或 `zh`(中文) |
| `--tasks` | str | 全部 | 逗号分隔的任务列表，如 `BFS,DFS,degree` |
| `--samples-per-task` | int | `10` | 每个任务生成的样本数 |
| `--output-dir` | str | `data/generated` | 输出目录 |
| `--seed` | int | `42` | 随机种子 |
| `--num-nodes-range` | str | `[8, 15]` | 图节点数范围 |
| `--alpaca-format` | flag | - | 同时输出 Alpaca 格式 |

### 使用示例

```bash
# 示例1: 只生成特定任务的中文数据
python script/generate_multilang_data.py \
    --lang zh \
    --tasks BFS,DFS,shortest_path \
    --samples-per-task 50

# 示例2: 生成更大规模的图
python script/generate_multilang_data.py \
    --lang zh \
    --num-nodes-range "[15, 25]" \
    --samples-per-task 100

# 示例3: 同时生成 Alpaca 格式（用于LLaMA-Factory等）
python script/generate_multilang_data.py \
    --lang zh \
    --samples-per-task 100 \
    --alpaca-format
```

### 格式转换脚本参数

```bash
python script/convert_multilang_to_training_format.py [OPTIONS]
```

| 参数 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `--input` | str | 是 | 输入 JSON 文件路径 |
| `--output` | str | 是 | 输出 JSON 文件路径 |
| `--max-samples` | int | 否 | 最大转换样本数 |
| `--verify` | flag | 否 | 验证转换结果 |

---

## 数据格式说明

### 多语言原始格式

`generate_multilang_data.py` 生成的数据格式：

```json
{
    "id": "BFS_zh_0",
    "task": "BFS",
    "language": "zh",
    "num_nodes": 12,
    "num_edges": 15,
    "directed": false,
    "graph": "[(<0>, <1>), (<0>, <2>), ...]",
    "graph_adj": "{<0>: [<1>, <2>], <1>: [<0>, <3>], ...}",
    "graph_nl": "节点 <0> 连接到节点 <1>, <2>。\n节点 <1> 连接到节点 <0>, <3>。...",
    "question": "从节点 <5> 开始，输出广度优先搜索 (BFS) 遍历序列。",
    "answer": "[<5>, <0>, <3>, <1>, <2>, ...]",
    "steps": "让我们逐步执行广度优先搜索 (BFS)。\n访问节点 <5>。节点 <5> 的未访问邻居是 [<0>, <3>]。...",
    "choices": "[[<5>, <0>, ...], [<5>, <3>, ...], ...]",
    "label": "0"
}
```

### GraphAgent 训练格式

`convert_multilang_to_training_format.py` 转换后的格式：

```json
{
    "id": "BFS_zh_0",
    "graph": {
        "node_idx": 0,
        "edge_index": [[0, 0, 1, 1, ...], [1, 2, 0, 3, ...]],
        "node_list": [0, 1, 2, 3, 4, 5, ...]
    },
    "conversations": [
        {
            "from": "human",
            "value": "<graph>\n节点 <0> 连接到节点 <1>, <2>。\n...\n\n从节点 <5> 开始，输出广度优先搜索 (BFS) 遍历序列。"
        },
        {
            "from": "gpt",
            "value": "让我们逐步执行广度优先搜索 (BFS)。\n访问节点 <5>。...<<<[<5>, <0>, <3>, ...]>>>"
        }
    ],
    "task_type": "BFS",
    "language": "zh",
    "num_nodes": 12,
    "num_edges": 15
}
```

### Alpaca 格式

使用 `--alpaca-format` 参数生成的格式（适用于 LLaMA-Factory）：

```json
{
    "instruction": "节点 <0> 连接到节点 <1>, <2>。\n...\n\n从节点 <5> 开始，输出广度优先搜索 (BFS) 遍历序列。",
    "input": "",
    "output": "让我们逐步执行广度优先搜索 (BFS)。\n...<<<[<5>, <0>, <3>, ...]>>>",
    "task_type": "BFS",
    "language": "zh"
}
```

---

## API 参考

### Python API 使用

#### 单个样本生成

```python
from GTG.tasks.BFS.BFS_multilang import generate_a_sample_multilang
from GTG.utils.language import LANG_ZH, LANG_EN

config = {'num_nodes_range': '[8, 15]'}

# 生成中文样本
sample_zh = generate_a_sample_multilang(config, lang=LANG_ZH)
print(f"问题: {sample_zh['question']}")
print(f"答案: {sample_zh['answer']}")

# 生成英文样本
sample_en = generate_a_sample_multilang(config, lang=LANG_EN)
print(f"Question: {sample_en['question']}")
print(f"Answer: {sample_en['answer']}")
```

#### 批量生成

```python
from script.generate_multilang_data import generate_sample_for_task, ALL_TASKS
from GTG.utils.language import LANG_ZH

config = {'num_nodes_range': '[8, 15]'}

for task in ALL_TASKS:
    sample = generate_sample_for_task(task, config, lang=LANG_ZH)
    if sample:
        print(f"{task}: {sample['question'][:50]}...")
```

#### 格式转换

```python
from script.convert_multilang_to_training_format import convert_sample_to_training_format

# 假设 sample 是多语言格式的样本
training_sample = convert_sample_to_training_format(sample, sample_idx=0)

print(f"Nodes: {training_sample['graph']['node_list']}")
print(f"Edges: {len(training_sample['graph']['edge_index'][0])}")
```

---

## 完整工作流程

### 生成中文训练数据的完整流程

```bash
# 步骤 1: 生成多语言数据
python script/generate_multilang_data.py \
    --lang zh \
    --samples-per-task 1000 \
    --output-dir data/generated \
    --seed 42

# 步骤 2: 转换为 GraphAgent 训练格式
python script/convert_multilang_to_training_format.py \
    --input data/generated/graphinstruct_zh_19tasks_1000samples.json \
    --output data/training/graphinstruct_zh_training.json \
    --verify

# 步骤 3: (可选) 进一步处理为 .pt 格式供训练使用
# 需要将转换后的 JSON 与现有的 prepare_sampled_data_qwen.py 配合使用
```

### 生成中英文混合数据

```bash
# 生成中文
python script/generate_multilang_data.py --lang zh --samples-per-task 500 --output-dir data/generated

# 生成英文
python script/generate_multilang_data.py --lang en --samples-per-task 500 --output-dir data/generated

# 合并（使用 Python）
python -c "
import json
with open('data/generated/graphinstruct_zh_19tasks_500samples.json') as f:
    zh_data = json.load(f)
with open('data/generated/graphinstruct_en_19tasks_500samples.json') as f:
    en_data = json.load(f)
merged = zh_data + en_data
with open('data/generated/graphinstruct_mixed_19000samples.json', 'w') as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)
print(f'Merged {len(merged)} samples')
"
```

---

## 常见问题

### Q1: 某些任务生成失败怎么办？

部分任务（如 `bipartite`, `maximum_flow`）需要特定的图结构。如果生成失败，脚本会自动跳过并记录日志。可以通过增加 `--samples-per-task` 来确保有足够的成功样本。

### Q2: 如何自定义图的大小？

使用 `--num-nodes-range` 参数：

```bash
# 小图 (5-10 节点)
python script/generate_multilang_data.py --num-nodes-range "[5, 10]" ...

# 大图 (20-30 节点)
python script/generate_multilang_data.py --num-nodes-range "[20, 30]" ...
```

### Q3: 转换后的数据可以直接用于 GraphAgent 训练吗？

转换后的 JSON 格式与 `sampled_instruct_ds_stage_2.json` 兼容。需要进一步使用 `prepare_sampled_data_qwen.py` 处理为 `.pt` 格式才能直接训练。

### Q4: 如何添加新的语言支持？

1. 在 `GTG/utils/language.py` 中添加新语言常量和模板
2. 更新 `TASK_TEMPLATES` 字典中的对应任务模板
3. 多语言生成器会自动使用新模板

---

## 文件结构

```
GraphInstruct/
├── GTG/
│   ├── tasks/
│   │   ├── BFS/
│   │   │   ├── BFS.py              # 原始英文版本
│   │   │   └── BFS_multilang.py    # 多语言版本
│   │   ├── DFS/
│   │   │   ├── DFS.py
│   │   │   └── DFS_multilang.py
│   │   └── ... (其他17个任务)
│   └── utils/
│       └── language.py             # 语言模板定义
├── script/
│   ├── generate_multilang_data.py           # 数据生成脚本
│   ├── convert_multilang_to_training_format.py  # 格式转换脚本
│   ├── create_multilang_tasks.py            # 批量创建多语言任务文件
│   └── fix_multilang_tasks.py               # 修复特殊任务的多语言文件
├── data/
│   ├── generated/                  # 生成的多语言数据
│   └── training/                   # 转换后的训练数据
└── docs/
    └── multilang_data_guide.md     # 本文档
```

---

## 更新日志

- **2025-12-18**: 初始版本，支持 19 种任务的中英文生成
- 修复了 `cycle`, `topological_sort`, `bipartite` 等任务的函数签名问题
- 添加了 `convert_multilang_to_training_format.py` 转换脚本

---

*本文档由 GraphInstruct 项目维护*
