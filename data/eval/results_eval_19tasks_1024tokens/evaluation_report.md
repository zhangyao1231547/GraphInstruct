# GraphAgent 评测报告 (max_new_tokens=1024)

**生成时间**: 2025-12-18 15:04:30

## 1. 总体概述

- **模型**: GraphAgent-Qwen3-4B
- **总样本数**: 190
- **正确数**: 42 (22.11%)
- **max_new_tokens**: 1024

> 注: 使用`<<<...>>>`答案提取进行评分

## 2. 各任务类型评测结果

| 任务类型 | 样本数 | 正确数 | 准确率 |
|----------|--------|--------|--------|
| degree | 10 | 9 | 90.0% |
| edge | 10 | 9 | 90.0% |
| cycle | 10 | 6 | 60.0% |
| neighbor | 10 | 6 | 60.0% |
| connectivity | 10 | 2 | 20.0% |
| diameter | 10 | 2 | 20.0% |
| jaccard | 10 | 2 | 20.0% |
| clustering_coefficient | 10 | 1 | 10.0% |
| common_neighbor | 10 | 1 | 10.0% |
| connected_component | 10 | 1 | 10.0% |
| MST | 10 | 1 | 10.0% |
| predecessor | 10 | 1 | 10.0% |
| shortest_path | 10 | 1 | 10.0% |
| BFS | 10 | 0 | 0.0% |
| DFS | 10 | 0 | 0.0% |
| bipartite | 10 | 0 | 0.0% |
| maximum_flow | 10 | 0 | 0.0% |
| page_rank | 10 | 0 | 0.0% |
| topological_sort | 10 | 0 | 0.0% |

## 3. 任务难度分析

### 简单任务 (准确率 >= 70%)
- degree: 90.0%
- edge: 90.0%

### 中等任务 (准确率 30-70%)
- cycle: 60.0%
- neighbor: 60.0%

### 困难任务 (准确率 < 30%)
- connectivity: 20.0%
- diameter: 20.0%
- jaccard: 20.0%
- clustering_coefficient: 10.0%
- common_neighbor: 10.0%
- connected_component: 10.0%
- MST: 10.0%
- predecessor: 10.0%
- shortest_path: 10.0%
- BFS: 0.0%
- DFS: 0.0%
- bipartite: 0.0%
- maximum_flow: 0.0%
- page_rank: 0.0%
- topological_sort: 0.0%

## 4. 预测示例

### 正确预测示例

**示例 1** (任务: clustering_coefficient)
```
Ground Truth: 0.0
Prediction:   0.0
```

**示例 2** (任务: common_neighbor)
```
Ground Truth: 0
Prediction:   0
```

**示例 3** (任务: connected_component)
```
Ground Truth: [4]
Prediction:   [4]
```

### 错误预测示例

**示例 1** (任务: BFS)
```
Ground Truth: [14, 9, 6, 12, 8, 11, 2, 0, 4, 10, 1, 7, 5, 3]
Prediction:   Let's run breadth-first search (BFS) step by step.
The queue of nodes to visit at the start is [14].
The queue of nodes to visit after processing node 14 is [11, 12].Node 11 is connected to nodes 0, 5
Full output (first 300 chars): Let's run breadth-first search (BFS) step by step.
The queue of nodes to visit at the start is [14].
The queue of nodes to visit after processing node 14 is [11, 12].Node 11 is connected to nodes 0, 5.
Node 12 is connected to nodes 4, 2, 9.
Node 7 is connected to nodes 2, 6, 3, 1.
Node 2 is connecte...
```

**示例 2** (任务: BFS)
```
Ground Truth: [1, 4, 8, 5, 7, 0, 6, 3, 2]
Prediction:   Let's run breadth-first search (BFS) step by step.
The nodes at level 0 are: [1].
The nodes at level 1 are: [4, 8, 0].
The nodes at level 2 are: [6, 2].
The nodes at level 3 are: [7].
The nodes at lev
Full output (first 300 chars): Let's run breadth-first search (BFS) step by step.
The nodes at level 0 are: [1].
The nodes at level 1 are: [4, 8, 0].
The nodes at level 2 are: [6, 2].
The nodes at level 3 are: [7].
The nodes at level 4 are: [3].
The nodes at level 5 are: [5].
The nodes at level 6 are: [0].
The nodes at level 7 ar...
```

**示例 3** (任务: BFS)
```
Ground Truth: [0, 1, 3, 2, 6, 13, 9, 12, 11, 10, 7, 8, 4, 5]
Prediction:   Let's find the breadth-first traversal step by step.
The neighbors of node 0 are [10].The neighbors of node 10 are [7].The neighbors of node 7 are [3].The neighbors of node 3 are [8].The neighbors of 
Full output (first 300 chars): Let's find the breadth-first traversal step by step.
The neighbors of node 0 are [10].The neighbors of node 10 are [7].The neighbors of node 7 are [3].The neighbors of node 3 are [8].The neighbors of node 8 are [9].The neighbors of node 9 are [1].The neighbors of node 1 are [13].The neighbors of nod...
```

## 5. 评测配置

| 参数 | 值 |
|------|-----|
| 模型 | GraphAgent-Qwen3-4B |
| 图编码 | MetaHGT (768-dim) |
| max_new_tokens | 1024 |
| 任务数 | 19 |
| 每任务样本数 | 10 |
| 总样本数 | 190 |

## 6. 结论

GraphAgent模型整体表现**有待提升**,平均准确率为 22.1%。

- 表现最佳任务: **degree** (90.0%)
- 表现最差任务: **topological_sort** (0.0%)

---
*本报告由GraphAgent评测系统自动生成*