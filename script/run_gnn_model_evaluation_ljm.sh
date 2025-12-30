#!/bin/bash
# GraphAgent GNN Model Evaluation Script
# 用于评测带GNN编码的GraphAgent模型
#
# 模型路径: /mnt/yrfs/GraphAgent_model/zy/model/graphagent-qwen3-gnn-full/stage2-epoch5-8192-full_finetune/lightning_logs/version_0
# 评测数据: /nvme0/work/workspaces-zy/GraphInstruct/data/eval/graphinstruct_eval_19tasks_10samples.pt

set -e

# =============================================================================
# Environment Setup
# =============================================================================
source /nvme0/work/workspaces-zy/GraphGPT-zy/miniconda3/etc/profile.d/conda.sh
conda activate graphagent

# =============================================================================
# Configuration
# =============================================================================

# Base model path (Qwen3-4B-Instruct)
base_model=/nvme0/work/workspaces-zy/model/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507

# Checkpoint path - GNN model trained with edge_index support
# LoRA weak tasks训练的checkpoint (lora-weak-tasks-r32-epoch2-test version_2)
checkpoint_path=/mnt/yrfs/GraphAgent_model/liangjinming/model/graphagent-qwen3-instruct-st2/stage2-epoch5-8192-graphinstruct-full_finetune-debug/lightning_logs/version_14/model_epoch=4-step=12500.ckpt

# Evaluation data path - 19 tasks × 10 samples (v2: fixed data)
eval_data=/nvme0/work/workspaces-zy/GraphInstruct/data/eval/graphinstruct_eval_19tasks_10samples_v2.pt

# Output directory for results
timestamp=$(date +%Y%m%d_%H%M%S)
output_dir=/nvme0/work/workspaces-zy/GraphInstruct/data/eval/results_gnn_model_ljm_${timestamp}

# GPU configuration - 使用空闲GPU 7
export CUDA_VISIBLE_DEVICES=7
gpus="0"  # 映射后的GPU索引

# Evaluation parameters
batch_size=1
max_new_tokens=1024

# GNN Configuration - 注意: 根据训练时的设置来配置
# 如果训练时use_gnn_encoder=True,这里也要设为True
use_gnn_encoder=True  # 使用GNN编码器处理图结构
gnn_type="gat"
gnn_num_layers=2
gnn_hidden_channels=256
gnn_num_heads=4
gnn_dropout=0.1

# Debug parameters
verbose=True

# =============================================================================
# Parse command line arguments (optional overrides)
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            gpus="$2"
            shift 2
            ;;
        --checkpoint)
            checkpoint_path="$2"
            shift 2
            ;;
        --eval_data)
            eval_data="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --max_new_tokens)
            max_new_tokens="$2"
            shift 2
            ;;
        --use_gnn_encoder)
            use_gnn_encoder="$2"
            shift 2
            ;;
        --verbose)
            verbose="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Check files exist
# =============================================================================

if [ ! -e "$checkpoint_path" ]; then
    echo "ERROR: Checkpoint not found: ${checkpoint_path}"
    echo "Please verify the checkpoint path exists."
    exit 1
fi

if [ ! -f "$eval_data" ]; then
    echo "ERROR: Evaluation data not found: ${eval_data}"
    exit 1
fi

echo "============================================"
echo "GraphAgent GNN Model Evaluation"
echo "============================================"
echo "Base Model: ${base_model}"
echo "Checkpoint: ${checkpoint_path}"
echo "Eval Data: ${eval_data}"
echo "Output Dir: ${output_dir}"
echo "GPUs: ${gpus}"
echo "Batch Size: ${batch_size}"
echo "Max New Tokens: ${max_new_tokens}"
echo "Use GNN Encoder: ${use_gnn_encoder}"
if [ "$use_gnn_encoder" = "True" ]; then
    echo "  GNN Type: ${gnn_type}"
    echo "  GNN Layers: ${gnn_num_layers}"
    echo "  GNN Hidden: ${gnn_hidden_channels}"
fi
echo "============================================"

# Create output directory
mkdir -p ${output_dir}

# =============================================================================
# Run Evaluation
# =============================================================================

cd /nvme0/work/workspaces-zy/GraphInstruct/script

python -u evaluate_gnn_model.py \
    --version qwen \
    --model_name_or_path ${base_model} \
    --checkpoint_path ${checkpoint_path} \
    --eval_data_path ${eval_data} \
    --output_dir ${output_dir} \
    --batch_size ${batch_size} \
    --max_new_tokens ${max_new_tokens} \
    --bf16 True \
    --model_max_length 4096 \
    --use_graph_start_end True \
    --gpus ${gpus} \
    --verbose ${verbose} \
    --use_gnn_encoder ${use_gnn_encoder} \
    --gnn_type ${gnn_type} \
    --gnn_num_layers ${gnn_num_layers} \
    --gnn_hidden_channels ${gnn_hidden_channels} \
    --gnn_num_heads ${gnn_num_heads} \
    --gnn_dropout ${gnn_dropout}

eval_status=$?

if [ $eval_status -ne 0 ]; then
    echo "ERROR: Evaluation failed with exit code $eval_status"
    exit 1
fi

echo ""
echo "============================================"
echo "Evaluation Complete!"
echo "============================================"
echo "Results saved to: ${output_dir}"
echo "  - metrics.json: Aggregated evaluation metrics"
echo "  - detailed_results.json: Per-sample predictions"
echo "  - evaluation_report.md: Markdown evaluation report"
echo "============================================"

# Show summary
if [ -f "${output_dir}/metrics.json" ]; then
    echo ""
    echo "Summary:"
    cat ${output_dir}/metrics.json | python -c "
import sys, json
data = json.load(sys.stdin)
print(f'  Total samples: {data[\"total\"]}')
print(f'  Correct: {data[\"correct\"]}')
print(f'  Accuracy: {data[\"accuracy\"]*100:.2f}%')
print()
print('  Results by task type:')
for task, metrics in sorted(data['by_task_type'].items(), key=lambda x: -x[1]['accuracy']):
    print(f'    {task}: {metrics[\"accuracy\"]*100:.1f}%')
"
fi
