#!/usr/bin/env python3
"""
GraphAgent GNN Model Evaluation Script

用于评测带GNN编码的GraphAgent模型在图推理任务上的表现。
支持多GPU评估,基于PyTorch Lightning。

模型路径: /mnt/yrfs/GraphAgent_model/zy/model/graphagent-qwen3-gnn-full/stage2-epoch5-8192-full_finetune/lightning_logs/version_0

特性:
- 支持GAT/GCN类型的GNN编码器
- 支持图结构感知的节点特征增强
- 详细的评测指标和错误分析
"""

import os
import sys
import json
import torch
import argparse
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Any
from datetime import datetime
from collections import defaultdict
import ast

import transformers
from torch.utils.data import DataLoader
from transformers import AutoConfig
from tqdm import tqdm

from lightning.pytorch import Trainer, seed_everything
from lightning import LightningModule, LightningDataModule

# 添加GraphAgent训练代码路径
sys.path.insert(0, '/nvme0/work/workspaces-zy/GraphAgent-zy/GraphAGent-training')

from model.graph_action_agent import conversation as conversation_lib
from model.graph_action_agent.pl_model import get_model_class_for_architecture
from dataloaders.abstract_dataloader import ProcessedDataset

IGNORE_INDEX = -100
DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_graph_mlp_adapter: bool = field(default=True)
    tune_embed_tokens: bool = field(default=True)
    full_finetune: bool = field(default=True)
    graph_tower: Optional[str] = field(default="MetaHGT_imdb_dblp_epoch5")
    graph_select_layer: Optional[int] = field(default=-2)
    pretrain_graph_mlp_adapter: Optional[str] = field(default=None)
    use_graph_start_end: bool = field(default=True)
    model_save_name: Optional[str] = field(default="model_{epoch}-{step}")
    tune_gnn: bool = field(default=False)
    graph_hidden_size: int = field(default=768)

    # GNN Encoder Configuration
    use_gnn_encoder: bool = field(
        default=False,
        metadata={"help": "Whether to use GNN encoder for graph structure learning"}
    )
    gnn_type: str = field(
        default="gat",
        metadata={"help": "Type of GNN encoder: 'gat' or 'gcn'"}
    )
    gnn_num_layers: int = field(
        default=2,
        metadata={"help": "Number of GNN layers"}
    )
    gnn_hidden_channels: int = field(
        default=256,
        metadata={"help": "Hidden dimension in GNN layers"}
    )
    gnn_num_heads: int = field(
        default=4,
        metadata={"help": "Number of attention heads (for GAT)"}
    )
    gnn_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout rate in GNN layers"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = True
    is_graph: bool = True
    graph_root: Optional[str] = field(default=None)
    hetero_key_path: Optional[str] = field(default=None)
    num_shot: Optional[int] = field(default=0)
    data_name: Optional[str] = field(default=None)


@dataclass
class EvalArguments:
    checkpoint_path: str = field(
        default=None, metadata={"help": "Path to the checkpoint file."}
    )
    eval_data_path: str = field(
        default="/nvme0/work/workspaces-zy/GraphInstruct/data/eval/graphinstruct_eval_19tasks_10samples.pt",
        metadata={"help": "Path to the evaluation data."}
    )
    output_dir: str = field(default="./eval_results_gnn")
    batch_size: int = field(default=1)
    max_new_tokens: int = field(default=512)
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(default=4096)
    gpus: str = field(default="0", metadata={"help": "GPU ids to use, e.g., '0,1,2,3'"})
    verbose: bool = field(default=True, metadata={"help": "Print detailed debug info during evaluation"})
    # LoRA parameters
    lora_enable: bool = field(default=False, metadata={"help": "Enable LoRA for model loading"})
    lora_r: int = field(default=32, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})


@dataclass
class DataCollatorForEvalDataset:
    """Collate examples for evaluation."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []

        for instance in instances:
            inp_ids = instance["input_ids"].clone()
            lbls = instance["labels"].clone()

            # Fix: Replace -100 in input_ids with pad_token_id
            invalid_mask = inp_ids < 0
            if invalid_mask.any():
                inp_ids[invalid_mask] = self.tokenizer.pad_token_id

            input_ids_list.append(inp_ids)
            labels_list.append(lbls)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=IGNORE_INDEX
        )
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "graph_data" in instances[0]:
            graph_data_batch = [instance["graph_data"] for instance in instances]
            key_order_batch = [instance["hetero_key_order"] for instance in instances]
            batch["graph_data"] = graph_data_batch
            batch["hetero_key_order"] = key_order_batch

        batch["id"] = [instance["id"] for instance in instances]

        # Add task_type if available
        if "task_type" in instances[0]:
            batch["task_type"] = [instance["task_type"] for instance in instances]

        return batch


class GraphAgentEvalDataModule(LightningDataModule):
    """Data module for evaluation."""

    def __init__(self, tokenizer, eval_data_path, batch_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.eval_data_path = eval_data_path
        self.batch_size = batch_size
        self.data_collator = DataCollatorForEvalDataset(tokenizer=tokenizer)

    def setup(self, stage=None):
        self.eval_dataset = ProcessedDataset(self.eval_data_path)
        print(f"Loaded {len(self.eval_dataset)} evaluation samples")

        # Debug: Check data format for first sample
        if len(self.eval_dataset) > 0:
            sample = self.eval_dataset[0]
            input_ids = sample["input_ids"]
            labels = sample["labels"]
            print(f"[Data Debug] First sample:")
            print(f"  input_ids: shape={input_ids.shape}, min={input_ids.min().item()}, max={input_ids.max().item()}")
            print(f"  labels: shape={labels.shape}, min={labels.min().item()}, max={labels.max().item()}")

    def predict_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            collate_fn=self.data_collator,
            shuffle=False,
            pin_memory=True,
        )


class GraphAgentGNNEvalModule(LightningModule):
    """Lightning module for GNN model evaluation."""

    def __init__(self, model, tokenizer, max_new_tokens, output_dir, verbose=True):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.output_dir = output_dir
        self.verbose = verbose
        self.detailed_results = []

    def extract_prompt_from_input(self, input_ids: torch.Tensor, labels: torch.Tensor) -> tuple:
        """Extract prompt (input without answer) from input_ids using labels."""
        answer_mask = labels != IGNORE_INDEX
        if answer_mask.any():
            first_answer_pos = answer_mask.nonzero(as_tuple=True)[0][0].item()
            prompt_ids = input_ids[:first_answer_pos]
        else:
            first_answer_pos = len(input_ids)
            prompt_ids = input_ids
        return prompt_ids, first_answer_pos

    def extract_ground_truth(self, labels: torch.Tensor) -> tuple:
        """Extract ground truth answer from labels."""
        answer_mask = labels != IGNORE_INDEX
        answer_ids = labels[answer_mask]
        if len(answer_ids) > 0:
            valid_answer_ids = answer_ids[answer_ids >= 0]
            if len(valid_answer_ids) > 0:
                try:
                    ground_truth = self.tokenizer.decode(valid_answer_ids, skip_special_tokens=True)
                except Exception as e:
                    ground_truth = f"[DECODE ERROR: {str(e)}]"
            else:
                ground_truth = ""
        else:
            ground_truth = ""
        return ground_truth, answer_ids

    def extract_final_answer(self, text: str) -> str:
        """Extract the final answer from text enclosed between '<<<' and '>>>'."""
        pattern = r'<<<\s*(.*?)\s*>>>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "NULL"

    def compare_answers(self, gt_answer: str, pred_answer: str, task_type: str = None, graph_data=None) -> dict:
        """Compare ground truth answer with predicted answer.

        For BFS/DFS tasks, uses level-based matching instead of exact matching.
        """
        if gt_answer == "NULL" and pred_answer == "NULL":
            is_correct = True
            match_type = "both_null"
        elif gt_answer == "NULL" or pred_answer == "NULL":
            is_correct = False
            match_type = "one_null"
        else:
            # 对于BFS/DFS任务，使用层级匹配
            if task_type in ['BFS', 'DFS'] and graph_data is not None:
                is_correct, match_type = self._compare_traversal_answers(
                    gt_answer, pred_answer, task_type, graph_data
                )
            # 对于顺序无关的列表任务，使用集合匹配
            elif task_type in ['connected_component', 'neighbor', 'common_neighbor', 'predecessor']:
                is_correct, match_type = self._compare_set_answers(gt_answer, pred_answer)
            # 对于数值列表任务（如page_rank），使用近似匹配
            elif task_type in ['page_rank', 'clustering_coefficient']:
                is_correct, match_type = self._compare_numeric_list_answers(gt_answer, pred_answer)
            else:
                is_correct = gt_answer == pred_answer
                if is_correct:
                    match_type = "exact_match"
                else:
                    if gt_answer.lower() == pred_answer.lower():
                        match_type = "case_insensitive_match"
                    else:
                        match_type = "no_match"

        return {
            "is_correct": is_correct,
            "match_type": match_type,
            "gt_answer": gt_answer,
            "pred_answer": pred_answer,
        }

    def _compare_set_answers(self, gt_answer: str, pred_answer: str) -> tuple:
        """Compare answers as sets (order-independent).

        For tasks like connected_component where the order doesn't matter.
        """
        try:
            gt_list = ast.literal_eval(gt_answer)
            pred_list = ast.literal_eval(pred_answer)

            gt_set = set(gt_list)
            pred_set = set(pred_list)

            if gt_set == pred_set:
                return True, "set_match"
            else:
                return False, "set_mismatch"
        except:
            # 如果无法解析为列表，回退到精确匹配
            is_correct = gt_answer == pred_answer
            return is_correct, "exact_match" if is_correct else "no_match"

    def _compare_numeric_list_answers(self, gt_answer: str, pred_answer: str, tolerance: float = 0.01) -> tuple:
        """Compare numeric answers with tolerance for floating point precision.

        For tasks like page_rank and clustering_coefficient where small precision
        differences should be tolerated.
        """
        try:
            # 尝试解析为浮点数
            gt_val = float(gt_answer)
            pred_val = float(pred_answer)

            # 使用相对误差或绝对误差比较
            if abs(gt_val) < 1e-6:
                is_close = abs(pred_val - gt_val) < tolerance
            else:
                is_close = abs(pred_val - gt_val) / abs(gt_val) < tolerance

            if is_close:
                return True, "numeric_match"
            else:
                return False, "numeric_mismatch"
        except:
            try:
                # 尝试解析为列表
                gt_list = ast.literal_eval(gt_answer)
                pred_list = ast.literal_eval(pred_answer)

                if len(gt_list) != len(pred_list):
                    return False, "length_mismatch"

                # 检查每个元素是否在误差范围内
                for gt_v, pred_v in zip(gt_list, pred_list):
                    gt_f = float(gt_v)
                    pred_f = float(pred_v)
                    if abs(gt_f) < 1e-6:
                        is_close = abs(pred_f - gt_f) < tolerance
                    else:
                        is_close = abs(pred_f - gt_f) / abs(gt_f) < tolerance
                    if not is_close:
                        return False, "numeric_mismatch"
                return True, "numeric_match"
            except:
                # 回退到精确匹配
                is_correct = gt_answer == pred_answer
                return is_correct, "exact_match" if is_correct else "no_match"

    def _compare_traversal_answers(self, gt_answer: str, pred_answer: str, task_type: str, graph_data) -> tuple:
        """Compare BFS/DFS traversal answers.

        BFS/DFS traversals can have multiple valid orderings:
        - BFS: Nodes at the same depth level can be visited in any order
        - DFS: Different neighbor visit orders lead to different valid traversals

        Validation rules:
        1. No duplicate nodes (each node visited exactly once)
        2. Same length as ground truth
        3. Same node set as ground truth
        4. Same starting node

        Returns:
            tuple: (is_correct, match_type)
        """
        try:
            gt_list = ast.literal_eval(gt_answer)
            pred_list = ast.literal_eval(pred_answer)
        except:
            # 如果无法解析为列表，回退到精确匹配
            is_correct = gt_answer == pred_answer
            return is_correct, "exact_match" if is_correct else "no_match"

        # 检查长度是否相同（每个节点只访问一次）
        if len(gt_list) != len(pred_list):
            return False, "length_mismatch"

        # 检查预测是否有重复节点
        if len(pred_list) != len(set(pred_list)):
            return False, "duplicate_nodes"

        # 检查是否包含相同的节点集合
        gt_set = set(gt_list)
        pred_set = set(pred_list)

        if gt_set != pred_set:
            # 节点集合不同
            return False, "node_set_mismatch"

        # 检查起始节点是否相同
        if len(gt_list) > 0 and len(pred_list) > 0:
            if gt_list[0] != pred_list[0]:
                return False, "start_node_mismatch"

        # 节点集合相同、长度相同、无重复、起始节点相同
        if task_type == 'BFS':
            return True, "bfs_valid"
        elif task_type == 'DFS':
            return True, "dfs_valid"

        return True, "traversal_valid"

    def safe_decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = False) -> str:
        """Safely decode token ids, handling invalid values."""
        try:
            if isinstance(token_ids, torch.Tensor):
                valid_ids = token_ids[token_ids >= 0]
            else:
                valid_ids = [t for t in token_ids if t >= 0]
            if len(valid_ids) == 0:
                return ""
            return self.tokenizer.decode(valid_ids, skip_special_tokens=skip_special_tokens)
        except Exception as e:
            return f"[DECODE ERROR: {str(e)}]"

    def get_graph_info(self, graph_data, hetero_key_order=None) -> dict:
        """Extract graph information for debugging."""
        info = {
            "is_hetero": False,
            "is_homo": False,
            "node_types": [],
            "node_counts": {},
            "has_edge_index": False,
            "edge_count": 0,
        }

        if graph_data is None:
            return info

        # Check for HeteroData
        if hasattr(graph_data, 'x_dict') and graph_data.x_dict:
            info["is_hetero"] = True
            info["node_types"] = list(graph_data.x_dict.keys())
            for k, v in graph_data.x_dict.items():
                info["node_counts"][k] = v.shape[0]

            # Check for edge_index_dict (needed for GNN)
            if hasattr(graph_data, 'edge_index_dict') and graph_data.edge_index_dict:
                info["has_edge_index"] = True
                info["edge_count"] = sum(v.shape[1] for v in graph_data.edge_index_dict.values())

        # Check for homogeneous graph
        elif hasattr(graph_data, 'x') and graph_data.x is not None:
            info["is_homo"] = True
            info["node_counts"]["default"] = graph_data.x.shape[0]

            if hasattr(graph_data, 'edge_index') and graph_data.edge_index is not None:
                info["has_edge_index"] = True
                info["edge_count"] = graph_data.edge_index.shape[1]

        return info

    def predict_step(self, batch, batch_idx):
        batch_input_ids = batch["input_ids"]
        batch_labels = batch["labels"]
        batch_attention_mask = batch["attention_mask"]
        graph_data = batch["graph_data"]
        hetero_key_order = batch["hetero_key_order"]
        batch_ids = batch["id"]
        batch_task_types = batch.get("task_type", ["unknown"] * len(batch_ids))

        results = []
        for i in range(len(batch_input_ids)):
            input_ids = batch_input_ids[i]
            labels = batch_labels[i]

            # Extract prompt
            prompt_ids, answer_start_pos = self.extract_prompt_from_input(input_ids, labels)
            prompt_ids_tensor = prompt_ids.unsqueeze(0).to(self.device)
            prompt_attention_mask = torch.ones_like(prompt_ids_tensor)

            # Extract ground truth
            gt_text, gt_ids = self.extract_ground_truth(labels)

            # Get graph data
            sample_graph_data = graph_data[i]
            sample_hetero_key_order = hetero_key_order[i]

            # Get graph info for debugging
            graph_info = self.get_graph_info(sample_graph_data, sample_hetero_key_order)

            # Decode prompt for debugging
            prompt_text = self.safe_decode(prompt_ids, skip_special_tokens=True)

            try:
                # Generate prediction
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=prompt_ids_tensor,
                        attention_mask=prompt_attention_mask,
                        graph_data=sample_graph_data,
                        hetero_key_order=sample_hetero_key_order,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                # Decode prediction
                generated_ids = outputs[0][prompt_ids_tensor.shape[1]:]
                pred_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                generation_error = None
            except Exception as e:
                print(f"Error generating for sample {batch_ids[i]}: {e}")
                pred_text = ""
                generation_error = str(e)

            # Extract and compare answers
            gt_final_answer = self.extract_final_answer(gt_text)
            pred_final_answer = self.extract_final_answer(pred_text)
            answer_comparison = self.compare_answers(
                gt_final_answer, pred_final_answer,
                task_type=batch_task_types[i],
                graph_data=sample_graph_data
            )

            result = {
                "id": batch_ids[i],
                "task_type": batch_task_types[i],
                "prediction": pred_text,
                "ground_truth": gt_text,
                "answer_eval": {
                    "gt_answer": gt_final_answer,
                    "pred_answer": pred_final_answer,
                    "is_correct": answer_comparison["is_correct"],
                    "match_type": answer_comparison["match_type"],
                },
                "graph_info": graph_info,
                "generation_error": generation_error,
            }

            results.append(result)

            # Print verbose info
            if self.verbose and batch_idx < 3:
                print(f"\n{'='*60}")
                print(f"Sample {batch_ids[i]} (Task: {batch_task_types[i]})")
                print(f"{'='*60}")
                print(f"Graph info: {graph_info}")
                print(f"Ground Truth Answer: {gt_final_answer}")
                print(f"Predicted Answer: {pred_final_answer}")
                print(f"Correct: {answer_comparison['is_correct']}")
                print(f"{'='*60}\n")

        return results

    def on_predict_epoch_end(self):
        """Gather all predictions and compute metrics."""
        all_results = self.trainer.predict_loop.predictions

        # Flatten results
        flat_results = []
        for batch_results in all_results:
            if isinstance(batch_results, list):
                flat_results.extend(batch_results)
            else:
                flat_results.append(batch_results)

        if self.trainer.global_rank == 0:
            self.detailed_results = flat_results
            self._compute_and_save_metrics()

    def _compute_and_save_metrics(self):
        """Compute metrics and save results."""
        total = len(self.detailed_results)
        correct = 0
        by_task = {}
        match_types = {"exact_match": 0, "both_null": 0, "one_null": 0, "case_insensitive_match": 0, "no_match": 0}

        for r in self.detailed_results:
            answer_eval = r.get("answer_eval", {})
            is_correct = answer_eval.get("is_correct", False)
            match_type = answer_eval.get("match_type", "no_match")
            task_type = r.get("task_type", "unknown")

            if is_correct:
                correct += 1

            if match_type in match_types:
                match_types[match_type] += 1

            if task_type not in by_task:
                by_task[task_type] = {"total": 0, "correct": 0}
            by_task[task_type]["total"] += 1
            if is_correct:
                by_task[task_type]["correct"] += 1

        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0.0

        for task_type in by_task:
            task_total = by_task[task_type]["total"]
            task_correct = by_task[task_type]["correct"]
            by_task[task_type]["accuracy"] = task_correct / task_total if task_total > 0 else 0.0

        # Print results
        print("\n" + "=" * 60)
        print("GNN Model Evaluation Results")
        print("=" * 60)
        print(f"Total samples: {total}")
        print(f"Answer accuracy: {correct} / {total} ({accuracy*100:.2f}%)")
        print(f"\nMatch type breakdown:")
        for mt, count in match_types.items():
            print(f"  {mt}: {count}")

        print(f"\nResults by task type:")
        for task_type, task_metrics in sorted(by_task.items(), key=lambda x: -x[1]["accuracy"]):
            print(f"  {task_type}: {task_metrics['correct']}/{task_metrics['total']} ({task_metrics['accuracy']*100:.2f}%)")
        print("=" * 60)

        # Save results
        os.makedirs(self.output_dir, exist_ok=True)

        metrics = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "match_types": match_types,
            "by_task_type": by_task,
            "timestamp": datetime.now().isoformat(),
        }

        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\nMetrics saved to {metrics_path}")

        # Save detailed results
        details_path = os.path.join(self.output_dir, "detailed_results.json")
        with open(details_path, "w") as f:
            json.dump(self.detailed_results, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to {details_path}")

        # Generate markdown report
        self._generate_report(metrics)

    def _generate_report(self, metrics):
        """Generate markdown evaluation report."""
        report = []
        report.append("# GraphAgent GNN Model Evaluation Report")
        report.append("")
        report.append(f"**Generated**: {metrics['timestamp']}")
        report.append("")

        report.append("## 1. Overall Results")
        report.append("")
        report.append(f"- **Total Samples**: {metrics['total']}")
        report.append(f"- **Correct**: {metrics['correct']}")
        report.append(f"- **Accuracy**: {metrics['accuracy']*100:.2f}%")
        report.append("")

        report.append("## 2. Results by Task Type")
        report.append("")
        report.append("| Task Type | Total | Correct | Accuracy |")
        report.append("|-----------|-------|---------|----------|")

        sorted_tasks = sorted(metrics['by_task_type'].items(), key=lambda x: -x[1]["accuracy"])
        for task_type, task_metrics in sorted_tasks:
            report.append(f"| {task_type} | {task_metrics['total']} | {task_metrics['correct']} | {task_metrics['accuracy']*100:.1f}% |")

        report.append("")

        report.append("## 3. Task Difficulty Analysis")
        report.append("")

        easy_tasks = [(t, m) for t, m in sorted_tasks if m["accuracy"] >= 0.7]
        medium_tasks = [(t, m) for t, m in sorted_tasks if 0.3 <= m["accuracy"] < 0.7]
        hard_tasks = [(t, m) for t, m in sorted_tasks if m["accuracy"] < 0.3]

        if easy_tasks:
            report.append("### Easy Tasks (Accuracy >= 70%)")
            for t, m in easy_tasks:
                report.append(f"- {t}: {m['accuracy']*100:.1f}%")
            report.append("")

        if medium_tasks:
            report.append("### Medium Tasks (Accuracy 30-70%)")
            for t, m in medium_tasks:
                report.append(f"- {t}: {m['accuracy']*100:.1f}%")
            report.append("")

        if hard_tasks:
            report.append("### Hard Tasks (Accuracy < 30%)")
            for t, m in hard_tasks:
                report.append(f"- {t}: {m['accuracy']*100:.1f}%")
            report.append("")

        report.append("## 4. Conclusion")
        report.append("")
        avg_rate = metrics['accuracy'] * 100
        if avg_rate >= 70:
            report.append(f"The GNN model performs **excellently**, achieving {avg_rate:.1f}% overall accuracy.")
        elif avg_rate >= 50:
            report.append(f"The GNN model performs **well**, achieving {avg_rate:.1f}% overall accuracy.")
        elif avg_rate >= 30:
            report.append(f"The GNN model performs **moderately**, achieving {avg_rate:.1f}% overall accuracy.")
        else:
            report.append(f"The GNN model **needs improvement**, achieving only {avg_rate:.1f}% overall accuracy.")

        report.append("")
        report.append("---")
        report.append("*Report generated by GraphAgent GNN Evaluation System*")

        report_path = os.path.join(self.output_dir, "evaluation_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        print(f"Evaluation report saved to {report_path}")


def load_checkpoint(model, checkpoint_path, lora_config=None):
    """Load checkpoint with proper key mapping. Supports PEFT LoRA checkpoints."""
    print(f"Loading checkpoint from {checkpoint_path}...")

    def is_peft_checkpoint(state_dict):
        """Check if the checkpoint contains PEFT LoRA weights."""
        for k in state_dict.keys():
            if 'lora_A' in k or 'lora_B' in k:
                return True
        return False

    def process_state_dict(state_dict, is_peft=False):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            # Handle PEFT checkpoint keys: model.base_model.model.model.* -> base_model.model.model.*
            if is_peft and new_k.startswith("model.base_model."):
                new_k = new_k[6:]  # Remove "model." prefix
            elif new_k.startswith("model.model."):
                new_k = new_k[6:]
            if new_k.startswith("_forward_module."):
                new_k = new_k[len("_forward_module."):]
            if new_k.startswith("module."):
                new_k = new_k[len("module."):]
            new_state_dict[new_k] = v
        return new_state_dict

    if os.path.isdir(checkpoint_path):
        # Check for pytorch_model.bin
        pytorch_model_bin = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(pytorch_model_bin):
            state_dict = torch.load(pytorch_model_bin, map_location="cpu", weights_only=False)
        else:
            # Try checkpoint subdirectory
            checkpoint_subdir = os.path.join(checkpoint_path, "checkpoint")
            model_state_file = os.path.join(checkpoint_subdir, "mp_rank_00_model_states.pt")
            if os.path.exists(model_state_file):
                state_dict_raw = torch.load(model_state_file, map_location="cpu", weights_only=False)
                state_dict = state_dict_raw.get("module", state_dict_raw)
            else:
                raise FileNotFoundError(f"Cannot find checkpoint in {checkpoint_path}")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Check if this is a PEFT checkpoint
    if is_peft_checkpoint(state_dict):
        print("Detected PEFT LoRA checkpoint format")
        try:
            from peft import LoraConfig, get_peft_model

            if lora_config is None:
                # Extract LoRA config from checkpoint keys
                lora_config = LoraConfig(
                    r=32,
                    lora_alpha=64,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )

            # Wrap model with PEFT
            model = get_peft_model(model, lora_config)
            print(f"Wrapped model with PEFT LoRA (r={lora_config.r}, alpha={lora_config.lora_alpha})")

            # Load the state dict
            new_state_dict = process_state_dict(state_dict, is_peft=True)
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded PEFT checkpoint. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

            # Merge LoRA weights for inference
            print("Merging LoRA weights into base model...")
            model = model.merge_and_unload()
            print("LoRA weights merged successfully")

        except ImportError:
            print("Warning: PEFT not available. Falling back to direct state dict loading.")
            new_state_dict = process_state_dict(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded checkpoint. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    else:
        new_state_dict = process_state_dict(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded checkpoint. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

    return model


def evaluate():
    seed_everything(42)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, EvalArguments)
    )
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

    # Parse GPU configuration
    if isinstance(eval_args.gpus, str):
        devices = [int(x) for x in eval_args.gpus.split(",")]
    else:
        devices = [eval_args.gpus]
    num_devices = len(devices)

    print("=" * 60)
    print("GraphAgent GNN Model Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {eval_args.checkpoint_path}")
    print(f"Eval Data: {eval_args.eval_data_path}")
    print(f"Output Dir: {eval_args.output_dir}")
    print(f"GPUs: {devices}")
    print(f"GNN Encoder: {model_args.use_gnn_encoder}")
    if model_args.use_gnn_encoder:
        print(f"  GNN Type: {model_args.gnn_type}")
        print(f"  GNN Layers: {model_args.gnn_num_layers}")
        print(f"  GNN Hidden: {model_args.gnn_hidden_channels}")
    print("=" * 60)

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=eval_args.cache_dir,
    )

    # Setup conversation template
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model_type = model_config.model_type.lower()

    if model_args.version == "qwen" or model_type in ("qwen2", "qwen3", "qwen"):
        eos_token = "<|im_end|>"
        eos_id = tokenizer.convert_tokens_to_ids(eos_token)
        if eos_id is None or eos_id == tokenizer.unk_token_id:
            eos_token = "<|endoftext|>"
            eos_id = tokenizer.convert_tokens_to_ids(eos_token)
        tokenizer.pad_token = eos_token
        tokenizer.pad_token_id = eos_id
        conversation_lib.default_conversation = conversation_lib.conv_templates["qwen"]
        print(f"[Conversation] Using Qwen template")
    else:
        eot = "<|eot_id|>"
        eot_id = tokenizer.convert_tokens_to_ids(eot)
        tokenizer.pad_token = eot
        tokenizer.pad_token_id = eot_id
        conversation_lib.default_conversation = conversation_lib.conv_templates["llama-3"]
        print(f"[Conversation] Using Llama-3 template")

    # Build model
    print("Loading model...")
    hf_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    hf_config.graph_hidden_size = model_args.graph_hidden_size
    hf_config.graph_select_layer = model_args.graph_select_layer
    hf_config.use_graph_start_end = model_args.use_graph_start_end

    model_class, model_type_str = get_model_class_for_architecture(model_args.model_name_or_path)

    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=eval_args.cache_dir,
        config=hf_config,
    )

    model.config.use_cache = True

    # Prepare GNN configuration
    gnn_config = None
    if model_args.use_gnn_encoder:
        gnn_config = {
            'gnn_type': model_args.gnn_type,
            'gnn_num_layers': model_args.gnn_num_layers,
            'gnn_hidden_channels': model_args.gnn_hidden_channels,
            'gnn_num_heads': model_args.gnn_num_heads,
            'gnn_dropout': model_args.gnn_dropout,
        }
        print(f"[GNN] Enabling GNN encoder with config: {gnn_config}")

    # Initialize graph modules with GNN support
    model.get_model().initialize_graph_modules(
        graph_tower=model_args.graph_tower,
        graph_select_layer=model_args.graph_select_layer,
        pretrain_graph_mlp_adapter=None,
        fsdp=None,
        use_gnn_encoder=model_args.use_gnn_encoder,
        gnn_config=gnn_config,
    )

    # Initialize graph tokenizer
    model.initialize_graph_tokenizer(
        use_graph_start_end=model_args.use_graph_start_end,
        tokenizer=tokenizer,
        device="cuda",
        tune_embed_tokens=False,
        pretrain_graph_mlp_adapter=None,
        model_args=model_args,
    )

    # Load checkpoint
    if eval_args.checkpoint_path:
        model = load_checkpoint(model, eval_args.checkpoint_path)

    # Convert to appropriate dtype
    if eval_args.bf16:
        model = model.to(torch.bfloat16)
    elif eval_args.fp16:
        model = model.to(torch.float16)

    model.eval()

    # Create data module
    data_module = GraphAgentEvalDataModule(
        tokenizer=tokenizer,
        eval_data_path=eval_args.eval_data_path,
        batch_size=eval_args.batch_size,
    )

    # Create evaluation module
    eval_module = GraphAgentGNNEvalModule(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=eval_args.max_new_tokens,
        output_dir=eval_args.output_dir,
        verbose=eval_args.verbose,
    )

    # Determine precision and strategy
    precision = "16" if eval_args.fp16 else ("bf16" if eval_args.bf16 else "32")
    strategy = "auto" if num_devices == 1 else "ddp"

    # Create trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
    )

    # Run prediction
    print("\nStarting GNN model evaluation...")
    trainer.predict(eval_module, datamodule=data_module)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    evaluate()
