# GraphInstruct 脚本工具集

将 GraphInstruct 数据集转换为 GraphAgent 训练所需的格式，支持 **LLaMA** 和 **Qwen** 两种模型。

## 目录结构

```
script/
├── data_processing/          # 数据处理
│   ├── conversion/           # 格式转换（Alpaca→对话格式、PT→JSON等）
│   ├── preparation/          # 数据准备（训练数据tokenization、图编码等）
│   ├── generation/           # 数据生成（多语言任务、GNN训练数据等）
│   ├── translation/          # 翻译脚本（英文→中文）
│   └── fix/                  # 数据修复（格式修复、结构修复等）
├── evaluation/               # 模型评估
│   ├── scripts/              # 评估脚本（GNN模型、API模型、纯文本模型等）
│   └── runners/              # 评估执行脚本
├── training/                 # 训练相关
│   └── runners/              # 训练执行脚本
├── reporting/                # 报告生成（综合报告、评估报告、图表等）
├── utils/                    # 工具脚本
├── logs/                     # 日志文件
├── dataset_generation/       # 数据集生成（原始）
├── meta_generation/          # 元数据生成（原始）
└── graph_autoencoder/        # 图自动编码器
```

## 概述

GraphInstruct 数据集包含 19 种图推理任务，每种任务 10,000 条数据，共计 **190,000 条**训练数据。

### 支持的任务类型

| 任务类型 | 说明 |
|---------|------|
| BFS | 广度优先遍历 |
| DFS | 深度优先遍历 |
| SHORTEST_PATH | 最短路径计算 (Dijkstra) |
| PAGE_RANK | PageRank算法 |
| MST | 最小生成树 |
| MAXIMUM_FLOW | 最大流 |
| CONNECTIVITY | 连通性判断 |
| CYCLE | 环检测 |
| DEGREE | 节点度查询 |
| NEIGHBOR | 邻居节点查询 |
| BIPARTITE | 二分图判断 |
| CLUSTERING_COEFFICIENT | 聚类系数 |
| COMMON_NEIGHBOR | 共同邻居 |
| JACCARD | Jaccard相似度 |
| DIAMETER | 图直径 |
| TOPOLOGICAL_SORT | 拓扑排序 |
| CONNECTED_COMPONENT | 连通分量 |
| PREDECESSOR | 前驱节点 |
| EDGE | 边存在判断 |

## 快速开始

### 1. 一键转换

```bash
cd /nvme0/work/workspaces-zy/GraphInstruct

# 生成 LLaMA 格式数据 (默认)
bash script/data_processing/conversion/run_conversion.sh --llama

# 生成 Qwen 格式数据
bash script/data_processing/conversion/run_conversion.sh --qwen

# 同时生成两种格式
bash script/data_processing/conversion/run_conversion.sh --both

# 测试模式 (每个任务10条)
bash script/data_processing/conversion/run_conversion.sh --both --max-samples 10
```

### 2. 分步执行

#### Step 1: 转换数据格式 (Alpaca → 对话格式)

```bash
python3 script/data_processing/conversion/convert_to_graphagent.py \
    --input-dir LLaMAFactory/data/reasoning \
    --output-dir data/converted \
    --chinese  # 使用中文提示，或 --english 使用英文
```

#### Step 2a: 生成 LLaMA 训练数据

```bash
python3 script/data_processing/preparation/prepare_training_data.py \
    --input-dir data/converted \
    --output-dir data/training/llama \
    --model-path /path/to/llama/model \
    --max-length 4096
```

#### Step 2b: 生成 Qwen 训练数据

```bash
python3 script/data_processing/preparation/prepare_training_data_qwen.py \
    --input-dir data/converted \
    --output-dir data/training/qwen \
    --model-path /path/to/qwen/model \
    --max-length 4096
```

## 输出目录结构

```
data/
├── converted/                    # 转换后的 JSON 文件
│   ├── bfs_converted.json
│   ├── dfs_converted.json
│   ├── ...
│   └── merged_graphinstruct.json # 合并的所有任务数据
│
└── training/                     # 训练用的 pt 文件
    ├── llama/                    # LLaMA 格式
    │   ├── bfs_train.pt
    │   ├── dfs_train.pt
    │   ├── ...
    │   └── graphinstruct_train_merged.pt
    │
    └── qwen/                     # Qwen 格式
        └── graphinstruct_train_qwen.pt
```

## LLaMA vs Qwen 格式差异

| 特性 | LLaMA-3 | Qwen |
|------|---------|------|
| 对话模板 | `<\|start_header_id\|>user<\|end_header_id\|>` | `<\|im_start\|>user` |
| 结束符 | `<\|eot_id\|>` | `<\|im_end\|>` |
| Pad token | `<\|eot_id\|>` | `<\|im_end\|>` |
| System 格式 | `<\|start_header_id\|>system...` | `<\|im_start\|>system\n...` |

### LLaMA-3 对话格式示例
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are GraphAgent...<|eot_id|><|start_header_id|>user<|end_header_id|>

给定以下图结构:
<graph>

请从节点 3 开始...<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Let's run BFS step by step...<|eot_id|>
```

### Qwen 对话格式示例
```
<|im_start|>system
You are GraphAgent...<|im_end|>
<|im_start|>user
给定以下图结构:
<graph>

请从节点 3 开始...<|im_end|>
<|im_start|>assistant
Let's run BFS step by step...<|im_end|>
```

## 数据格式说明

### 转换后的 JSON 格式

```json
{
    "id": "graphinstruct_bfs_0",
    "graph": {
        "node_list": [0, 1, 2, 3, 4, 5],
        "edge_index": [[0, 0, 1, ...], [1, 2, 3, ...]],
        "is_directed": false
    },
    "conversations": [
        {
            "from": "human",
            "value": "给定以下图结构:\n<graph>\n\n请从节点 3 开始..."
        },
        {
            "from": "gpt",
            "value": "Let's run BFS step by step..."
        }
    ],
    "task_type": "BFS",
    "task_params": {"start_node": 3}
}
```

### 训练数据 pt 格式

每个样本包含：
- `id`: 样本ID
- `input_ids`: Tokenized 输入 (Tensor)
- `labels`: 标签，instruction部分已mask为-100 (Tensor)
- `graph_data`: HeteroData 图数据结构
- `hetero_key_order`: 节点类型顺序 `['node']`
- `task_type`: 任务类型

## 在 GraphAgent 中使用

### 方法一：修改训练配置

在 GraphAgent 训练脚本中，更新数据路径：

```python
# LLaMA 训练
TRAIN_DATA_PATH = "/nvme0/work/workspaces-zy/GraphInstruct/data/training/llama/graphinstruct_train_merged.pt"

# Qwen 训练
TRAIN_DATA_PATH = "/nvme0/work/workspaces-zy/GraphInstruct/data/training/qwen/graphinstruct_train_qwen.pt"
```

### 方法二：混合训练

将 GraphInstruct 数据与原有数据混合：

```python
import torch

# 加载原有数据
original_data = torch.load("original_train_data.pt")

# 加载 GraphInstruct 数据
graphinstruct_data = torch.load("graphinstruct_train_merged.pt")

# 合并
merged_data = original_data + graphinstruct_data

# 保存
torch.save(merged_data, "merged_train_data.pt")
```

## 命令行参数

### run_conversion.sh

```bash
Usage: bash script/run_conversion.sh [OPTIONS]

Options:
  --llama        生成 LLaMA 格式数据 (默认)
  --qwen         生成 Qwen 格式数据
  --both         同时生成两种格式
  --max-samples N  限制每个任务的样本数 (用于测试)
  --chinese      使用中文提示 (默认)
  --english      使用英文提示
  --help         显示帮助信息
```

### convert_to_graphagent.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input-dir` | GraphInstruct 数据目录 | `LLaMAFactory/data/reasoning` |
| `--output-dir` | 转换输出目录 | `data/converted` |
| `--max-samples` | 每个任务最大样本数 | None (全部) |
| `--chinese` | 使用中文提示 | True |
| `--english` | 使用英文提示 | False |
| `--keep-reasoning` | 保留完整推理过程 | True |

### prepare_training_data.py / prepare_training_data_qwen.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input-dir` | 转换后的JSON目录 | `data/converted` |
| `--output-dir` | 训练数据输出目录 | `data/training/llama` 或 `data/training/qwen` |
| `--model-path` | 基座模型路径 | LLaMA-3.1-8B / Qwen3-4B |
| `--max-length` | 最大序列长度 | 4096 |
| `--max-samples` | 每个任务最大样本数 | None |
| `--node-dim` | 节点特征维度 | 768 |

## 注意事项

1. **图嵌入**: 当前使用随机初始化的节点特征。如需使用预训练图编码器，需要额外处理。

2. **序列长度**: 部分复杂任务（如最短路径推理）输出较长，可能超过 max_length。

3. **多图任务**: 当前不支持包含多个 `<graph>` token 的样本。

4. **磁盘空间**: 完整转换 190,000 条数据约需 20GB+ 磁盘空间。

5. **内存需求**: 转换过程需要加载 tokenizer，建议至少 16GB 内存。

## 脚本文件说明

### 数据处理 (data_processing/)

#### conversion/ - 格式转换
| 文件 | 功能 |
|------|------|
| `convert_to_graphagent.py` | 将 Alpaca 格式转换为 GraphAgent 对话格式，解析图结构 |
| `convert_*_to_text_json.py` | 将 PyTorch 格式数据转换为纯文本 JSON 格式 |
| `convert_to_chinese_training_data.py` | 生成中文训练数据 |
| `run_conversion.sh` | 一键运行转换脚本 |

#### preparation/ - 数据准备
| 文件 | 功能 |
|------|------|
| `prepare_training_data.py` | 生成 **LLaMA** 格式的训练数据 (.pt) |
| `prepare_training_data_qwen.py` | 生成 **Qwen** 格式的训练数据 (.pt) |
| `prepare_training_data_qwen_with_graph_encoding.py` | 生成带图编码的 Qwen 训练数据 |
| `prepare_evaluation_data.py` | 准备评估数据集 |

#### generation/ - 数据生成
| 文件 | 功能 |
|------|------|
| `create_multilang_tasks.py` | 创建多语言任务数据 |
| `generate_gnn_training_data.py` | 生成 GNN 训练数据 |
| `create_chinese_eval_data.py` | 创建中文评估数据 |

#### translation/ - 翻译
| 文件 | 功能 |
|------|------|
| `translate_en_to_zh.py` | 英文翻译为中文 |
| `translate_three_tasks.py` | 翻译特定三个任务 |

#### fix/ - 数据修复
| 文件 | 功能 |
|------|------|
| `fix_chinese_data_structure.py` | 修复中文数据结构 |
| `fix_evaluation_data.py` | 修复评估数据 |
| `fix_multilang_tasks.py` | 修复多语言任务数据 |

### 模型评估 (evaluation/)

#### scripts/ - 评估脚本
| 文件 | 功能 |
|------|------|
| `evaluate_gnn_model.py` | 评估 GNN 模型 |
| `evaluate_api_model.py` | 评估 API 模型 |
| `evaluate_text_only.py` | 评估纯文本模型 |

#### runners/ - 执行脚本
| 文件 | 功能 |
|------|------|
| `run_gnn_model_evaluation.sh` | 运行 GNN 模型评估 |
| `run_chinese_evaluation.sh` | 运行中文评估 |
| `run_all_evaluation.sh` | 运行所有评估 |

### 训练 (training/)

| 文件 | 功能 |
|------|------|
| `train_qwen32b_graphinstruct.sh` | 训练 Qwen 32B 模型 |
| `pl_train_directed_gnn_stage3_graphinstruct.sh` | PyTorch Lightning GNN 训练 |

### 报告生成 (reporting/)

| 文件 | 功能 |
|------|------|
| `generate_comprehensive_report.py` | 生成综合评估报告 |
| `generate_evaluation_report.py` | 生成评估报告 |
| `generate_svg_charts.py` | 生成 SVG 图表 |
| `recalculate_metrics.py` | 重新计算评估指标 |

## 引用

```bibtex
@article{graphinstruct,
  title={GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability},
  author={Zihan Luo and Xiran Song and Hong Huang and others},
  journal={CoRR},
  volume={abs/2403.04483},
  year={2024}
}
```
