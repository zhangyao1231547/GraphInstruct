# Qwen3-4B 纯文本评测报告 (Baseline)

**生成时间**: 2025-12-18 11:46:15

## 1. 总体概述

- **模型**: Qwen3-4B-Instruct (Original)
- **总样本数**: 190
- **正确数**: 36 (18.95%)

> 注: 纯文本评测使用原始Qwen3-4B模型,不使用图编码,作为baseline对比。

## 2. 各任务类型评测结果

| 任务类型 | 样本数 | 正确数 | 准确率 |
|----------|--------|--------|--------|
| degree | 10 | 10 | 100.0% |
| edge | 10 | 10 | 100.0% |
| neighbor | 10 | 6 | 60.0% |
| common_neighbor | 10 | 5 | 50.0% |
| connectivity | 10 | 2 | 20.0% |
| clustering_coefficient | 10 | 1 | 10.0% |
| jaccard | 10 | 1 | 10.0% |
| predecessor | 10 | 1 | 10.0% |
| BFS | 10 | 0 | 0.0% |
| DFS | 10 | 0 | 0.0% |
| bipartite | 10 | 0 | 0.0% |
| connected_component | 10 | 0 | 0.0% |
| cycle | 10 | 0 | 0.0% |
| diameter | 10 | 0 | 0.0% |
| maximum_flow | 10 | 0 | 0.0% |
| MST | 10 | 0 | 0.0% |
| page_rank | 10 | 0 | 0.0% |
| shortest_path | 10 | 0 | 0.0% |
| topological_sort | 10 | 0 | 0.0% |

## 3. 任务难度分析

### 简单任务 (准确率 >= 70%)
- degree: 100.0%
- edge: 100.0%

### 中等任务 (准确率 30-70%)
- neighbor: 60.0%
- common_neighbor: 50.0%

### 困难任务 (准确率 < 30%)
- connectivity: 20.0%
- clustering_coefficient: 10.0%
- jaccard: 10.0%
- predecessor: 10.0%
- BFS: 0.0%
- DFS: 0.0%
- bipartite: 0.0%
- connected_component: 0.0%
- cycle: 0.0%
- diameter: 0.0%
- maximum_flow: 0.0%
- MST: 0.0%
- page_rank: 0.0%
- shortest_path: 0.0%
- topological_sort: 0.0%

## 4. 预测示例

### 正确预测示例

**示例 1** (任务: clustering_coefficient)
```
Ground Truth: 1.0
Prediction:   1.0
```

**示例 2** (任务: common_neighbor)
```
Ground Truth: 1
Prediction:   1
```

**示例 3** (任务: common_neighbor)
```
Ground Truth: 8
Prediction:   8
```

### 错误预测示例

**示例 1** (任务: BFS)
```
Ground Truth: [14, 9, 6, 12, 8, 11, 2, 0, 4, 10, 1, 7, 5, 3]
Prediction:   We are given a directed graph and asked to perform a **Breadth-First Search (BFS)** starting from **node 14**, and output the sequence of nodes visited in BFS order.

---

### Step 1: Understand the G
```

**示例 2** (任务: BFS)
```
Ground Truth: [1, 4, 8, 5, 7, 0, 6, 3, 2]
Prediction:   We are given a directed graph and asked to perform a **Breadth-First Search (BFS)** starting from **node 1**, and output the sequence of nodes visited in BFS order.

---

### Step 1: Understand the Gr
```

**示例 3** (任务: BFS)
```
Ground Truth: [0, 1, 3, 2, 6, 13, 9, 12, 11, 10, 7, 8, 4, 5]
Prediction:   We are given an undirected graph and asked to perform a **Breadth-First Search (BFS)** starting from **node 0**, and output the sequence of nodes visited in BFS order.

---

### Step 1: Understand BFS
```

## 5. 评测配置

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-4B-Instruct (原始) |
| 图编码 | 无 (纯文本) |
| 任务数 | 19 |
| 每任务样本数 | 10 |
| 总样本数 | 190 |

## 6. 结论

原始模型整体表现**较差**,平均准确率仅为 18.9%,说明图推理任务需要专门的图结构理解能力。

- 表现最佳任务: **degree** (100.0%)
- 表现最差任务: **topological_sort** (0.0%)

---
*本报告由纯文本评测系统自动生成 (Baseline)*