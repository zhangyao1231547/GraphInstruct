#!/bin/bash
# Stage 3: Directed GNN Training for graphagent-from-graphinstruct pipeline
# Uses GPUs 4-7 (0-3 occupied by vLLM service)
# Date: 2024-12-24

set -e

# ============================================
# Environment Setup
# ============================================
source /home/test/miniconda3/etc/profile.d/conda.sh
conda activate graphagent

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=7200
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

# ============================================
# 配置路径
# ============================================
base_model=/mnt/yrfs/GraphInstruct/model/qwen3-4b-graph-reasoning-merged
data_dir=/mnt/yrfs/GraphAgent_model/zy/data/directed_gnn
stage2_checkpoint=/mnt/yrfs/GraphAgent_model/zy/model/graphagent-from-graphinstruct/stage2-english-95k-epoch5/lightning_logs/version_0/model_epoch=4-step=29690.ckpt
output_base=/mnt/yrfs/GraphAgent_model/zy/model/graphagent-from-graphinstruct/stage3-directed-gnn

# ============================================
# GPU配置 (4卡)
# ============================================
seq_gpus="0,1,2,3"  # 相对于CUDA_VISIBLE_DEVICES的索引
num_gpus=4

# ============================================
# GNN配置 - DirectedWeightedGAT
# ============================================
use_gnn_encoder=True
gnn_type=directed_weighted_gat
gnn_num_layers=3
gnn_hidden_channels=256
gnn_num_heads=8
gnn_dropout=0.1

# ============================================
# 训练超参数
# ============================================
num_epochs=5
context_len=2048
learning_rate=1e-5
batch_size=1
grad_accum=32

# ============================================
# 输出目录
# ============================================
run_name="directed-weighted-gnn-epoch${num_epochs}"
output_dir="${output_base}/${run_name}"

echo "=============================================="
echo "Stage 3: Directed GNN Training"
echo "=============================================="
echo "Pipeline: graphagent-from-graphinstruct"
echo "Date: $(date)"
echo "GPUs: 4,5,6,7 (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "Stage2 Checkpoint: ${stage2_checkpoint}"
echo "Data: ${data_dir}"
echo "Output: ${output_dir}"
echo "Python: $(which python)"
echo "Conda Env: ${CONDA_DEFAULT_ENV}"
echo "=============================================="

# 创建输出目录
mkdir -p ${output_dir}

# 启动训练
cd /nvme0/work/workspaces-zy/GraphAgent-zy/GraphAGent-training

python -u pl_train.py \
    --version v1 \
    --model_name_or_path ${base_model} \
    --flash_attn False \
    --data_name directed_gnn_weak_tasks \
    --data_path ${data_dir} \
    --tune_graph_mlp_adapter True \
    --tune_embed_tokens False \
    --full_finetune True \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum} \
    --save_every_n_epochs 1 \
    --learning_rate ${learning_rate} \
    --logging_steps 10 \
    --model_max_length ${context_len} \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --is_graph True \
    --gpus ${seq_gpus} \
    --strategy ddp_find_unused_parameters_true \
    --use_gnn_encoder ${use_gnn_encoder} \
    --gnn_type ${gnn_type} \
    --gnn_num_layers ${gnn_num_layers} \
    --gnn_hidden_channels ${gnn_hidden_channels} \
    --gnn_num_heads ${gnn_num_heads} \
    --gnn_dropout ${gnn_dropout} \
    --num_workers 0 \
    --pretrain_graph_mlp_adapter ${stage2_checkpoint}

echo "=============================================="
echo "Training completed at $(date)"
echo "=============================================="
