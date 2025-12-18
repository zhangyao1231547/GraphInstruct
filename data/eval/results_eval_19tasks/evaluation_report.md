# GraphAgent 评测报告

**生成时间**: 2025-12-18 10:48:07 


## 1. 总体概述

- **总样本数**: 190
- **最终答案正确数**: 40 (21.05%)

> 注: 最终答案准确率基于提取`<<<...>>>`中的答案进行比较，忽略推理步骤的差异。

### 完整文本匹配指标（参考）
- **精确匹配数**: 0 (0.00%)
- **部分匹配数**: 0 (0.00%)

## 2. 各任务类型评测结果（基于最终答案）

| 任务类型 | 样本数 | 正确数 | 准确率 |
|----------|--------|--------|--------|
| degree | 10 | 9 | 90.0% |
| edge | 10 | 9 | 90.0% |
| cycle | 10 | 6 | 60.0% |
| neighbor | 10 | 6 | 60.0% |
| connectivity | 10 | 2 | 20.0% |
| jaccard | 10 | 2 | 20.0% |
| clustering_coefficient | 10 | 1 | 10.0% |
| common_neighbor | 10 | 1 | 10.0% |
| connected_component | 10 | 1 | 10.0% |
| diameter | 10 | 1 | 10.0% |
| MST | 10 | 1 | 10.0% |
| shortest_path | 10 | 1 | 10.0% |
| BFS | 10 | 0 | 0.0% |
| DFS | 10 | 0 | 0.0% |
| bipartite | 10 | 0 | 0.0% |
| maximum_flow | 10 | 0 | 0.0% |
| page_rank | 10 | 0 | 0.0% |
| predecessor | 10 | 0 | 0.0% |
| topological_sort | 10 | 0 | 0.0% |

## 3. 错误分析

- **空预测**: 0
- **错误答案**: 190
- **部分正确**: 0

## 4. 任务难度分析

### 简单任务 (准确率 ≥ 70%)
- degree: 90.0%
- edge: 90.0%

### 中等任务 (准确率 30-70%)
- cycle: 60.0%
- neighbor: 60.0%

### 困难任务 (准确率 < 30%)
- connectivity: 20.0%
- jaccard: 20.0%
- clustering_coefficient: 10.0%
- common_neighbor: 10.0%
- connected_component: 10.0%
- diameter: 10.0%
- MST: 10.0%
- shortest_path: 10.0%
- BFS: 0.0%
- DFS: 0.0%
- bipartite: 0.0%
- maximum_flow: 0.0%
- page_rank: 0.0%
- predecessor: 0.0%
- topological_sort: 0.0%

## 5. 预测示例

### 正确预测示例

**示例 1** (任务: clustering_coefficient)
```
Ground Truth Answer: 0.0
Prediction Answer:   0.0
```

**示例 2** (任务: common_neighbor)
```
Ground Truth Answer: 0
Prediction Answer:   0
```

**示例 3** (任务: connected_component)
```
Ground Truth Answer: [4]
Prediction Answer:   [4]
```

### 错误预测示例

**示例 1** (任务: BFS)
```
Ground Truth Answer: [14, 9, 6, 12, 8, 11, 2, 0, 4, 10, 1, 7, 5, 3]
Prediction Answer:   Let's run breadth-first search (BFS) step by step.
The queue of nodes to visit at the start is [14].
The queue of nodes to visit after processing node 14 is [11, 12].Node 11 is connected to nodes 0, 5.
Node 12 is connected to nodes 4, 2, 9.
Node 7 is connected to nodes 2, 6, 3, 1.
Node 2 is connected to node 10.
Node 6 is connected to node 0.
Node 3 is connected to nodes 5, 2.
Node 4 is connected to nodes 14, 12.
Node 8 is connected to nodes 10, 1, 7.
Node 10 is connected to nodes 0, 5, 12, 9.
Node 0 is connected to nodes 5, 2, 3.
Node 5 is connected to nodes 4, 3.
Node 9 is connected to nodes 6, 12, 8, 11, 2.
Node 13 is connected to nodes 1, 9.
The queue of nodes to visit after processing nodes 11, 12 is [0].Node 11 is connected to nodes 0, 5.
Node 12 is connected to nodes 4, 2, 9.
Node 7 is connected to nodes 2, 6, 3, 1.
Node 2 is connected to node 10.
Node 6 is connected to node 0.
Node 3 is connected to nodes 5, 2.
Node 4 is connected to nodes 14, 12.
Node 8 is connected to nodes 10, 1, 7.
Node 10 is connected to nodes 0, 5, 12, 9.
Node 0 is connected to nodes 5, 2, 3.
Node 5 is connected to nodes 4, 3.
Node 9 is connected to nodes 6, 12, 8, 11, 2.
Node 13 is connected to nodes 1, 9.
The queue of nodes to visit after processing node 0 is [5].Node 11 is connected to nodes 0, 5.
Node 12 is connected
```

**示例 2** (任务: BFS)
```
Ground Truth Answer: [1, 4, 8, 5, 7, 0, 6, 3, 2]
Prediction Answer:   Let's run breadth-first search (BFS) step by step.
The nodes at level 0 are: [1].
The nodes at level 1 are: [4, 8, 0].
The nodes at level 2 are: [6, 2].
The nodes at level 3 are: [7].
The nodes at level 4 are: [3].
The nodes at level 5 are: [5].
The nodes at level 6 are: [0].
The nodes at level 7 are: [2].
The nodes at level 8 are: [8].
The nodes at level 9 are: [0].
The nodes at level 10 are: [2].
The nodes at level 11 are: [0].
The nodes at level 12 are: [2].
The nodes at level 13 are: [0].
The nodes at level 14 are: [2].
The nodes at level 15 are: [0].
The nodes at level 16 are: [2].
The nodes at level 17 are: [0].
The nodes at level 18 are: [2].
The nodes at level 19 are: [0].
The nodes at level 20 are: [2].
The nodes at level 21 are: [0].
The nodes at level 22 are: [2].
The nodes at level 23 are: [0].
The nodes at level 24 are: [2].
The nodes at level 25 are: [0].
The nodes at level 26 are: [2].
The nodes at level 27 are: [0].
The nodes at level 28 are: [2].
The nodes at level 29 are: [0].
The nodes at level 30 are: [2].
The nodes at level 31 are: [0].
The nodes at level 32 are: [2].
The nodes at level 33 are: [0].
The nodes at level 34 are: [2].
The nodes at level 35 are: [0].
The nodes at level 36 are: [2].
The nodes at level 37 are: [0].
The nodes at level 38 are: [2].
The nodes at level 39 are: [0].
The nodes at level 40 are: [2].
The nodes at level 41
```

**示例 3** (任务: BFS)
```
Ground Truth Answer: [0, 1, 3, 2, 6, 13, 9, 12, 11, 10, 7, 8, 4, 5]
Prediction Answer:   Let's find the breadth-first traversal step by step.
The neighbors of node 0 are [10].The neighbors of node 10 are [7].The neighbors of node 7 are [3].The neighbors of node 3 are [8].The neighbors of node 8 are [9].The neighbors of node 9 are [1].The neighbors of node 1 are [13].The neighbors of node 13 are [2].The neighbors of node 2 are [4].The neighbors of node 4 are [12].The neighbors of node 12 are [5].The neighbors of node 5 are [6].The neighbors of node 6 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node 1 are [1].The neighbors of node
```

## 6. 评测配置

| 参数 | 值 |
|------|-----|
| 模型 | GraphAgent-Qwen3-4B |
| 任务数 | 19 |
| 每任务样本数 | 10 |
| 总样本数 | 190 |

## 7. 结论

模型整体表现**需改进**,平均准确率仅为 21.1%。

- 表现最佳任务: **degree** (90.0%)
- 表现最差任务: **topological_sort** (0.0%)

---
*本报告由 GraphAgent 评测系统自动生成*