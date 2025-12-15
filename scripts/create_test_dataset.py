#!/usr/bin/env python3
"""
从训练数据中划分测试集用于评测
每个任务取最后500条数据作为测试集
"""
import json
import os
from pathlib import Path

# 配置
REASONING_DATA_DIR = "/nvme0/work/workspaces-zy/GraphInstruct/LLaMAFactory/data/reasoning"
OUTPUT_DIR = "/nvme0/work/workspaces-zy/GraphInstruct/data/test"
TEST_SIZE = 500  # 每个任务取500条作为测试集

# 19个图推理任务
TASKS = [
    "BFS-int_id", "bipartite-int_id", "clustering_coefficient-int_id",
    "common_neighbor-int_id", "connected_component-int_id", "connectivity-int_id",
    "cycle-int_id", "degree-int_id", "DFS-int_id", "diameter-int_id",
    "edge-int_id", "jaccard-int_id", "maximum_flow-int_id", "MST-int_id",
    "neighbor-int_id", "page_rank-int_id", "predecessor-int_id",
    "shortest_path-int_id", "topological_sort-int_id"
]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_test_data = []

    for task in TASKS:
        task_file = os.path.join(REASONING_DATA_DIR, task, "train.json")

        if not os.path.exists(task_file):
            print(f"Warning: {task_file} not found, skipping...")
            continue

        with open(task_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 取最后500条作为测试集
        test_data = data[-TEST_SIZE:]

        # 添加任务标识
        for item in test_data:
            item['task'] = task.replace("-int_id", "")

        all_test_data.extend(test_data)

        # 保存单任务测试集
        task_output = os.path.join(OUTPUT_DIR, f"{task}_test.json")
        with open(task_output, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        print(f"[{task}] Created test set with {len(test_data)} samples")

    # 保存合并的测试集
    combined_output = os.path.join(OUTPUT_DIR, "all_tasks_test.json")
    with open(combined_output, 'w', encoding='utf-8') as f:
        json.dump(all_test_data, f, ensure_ascii=False, indent=2)

    print(f"\n=== Summary ===")
    print(f"Total test samples: {len(all_test_data)}")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
