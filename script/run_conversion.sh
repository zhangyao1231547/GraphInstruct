#!/bin/bash
# GraphInstruct 数据转换运行脚本
# 将 GraphInstruct 数据转换为 GraphAgent 训练格式

set -e

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="${PROJECT_DIR}/LLaMAFactory/data/reasoning"
CONVERTED_DIR="${PROJECT_DIR}/data/converted"
TRAINING_DIR="${PROJECT_DIR}/data/training"

# 模型路径 (用于 tokenizer)
MODEL_PATH="/nvme0/work/workspaces-zy/model/Llama-3.1-8B-Instruct/LLM-Research/Meta-Llama-3.1-8B-Instruct"

# 可选: 设置 QWEN 模型路径
# MODEL_PATH="/nvme0/work/workspaces-zy/model/Qwen2.5-7B-Instruct"

# 参数
MAX_SAMPLES_PER_TASK=""  # 留空表示使用全部数据, 设置数字进行测试
MAX_LENGTH=4096
USE_CHINESE="--chinese"  # 使用中文提示, 使用英文则改为 "--english"

echo "=============================================="
echo "GraphInstruct -> GraphAgent 数据转换"
echo "=============================================="
echo "输入目录: ${INPUT_DIR}"
echo "转换输出: ${CONVERTED_DIR}"
echo "训练输出: ${TRAINING_DIR}"
echo "模型路径: ${MODEL_PATH}"
echo "=============================================="

# 创建输出目录
mkdir -p "${CONVERTED_DIR}"
mkdir -p "${TRAINING_DIR}"

# Step 1: 转换数据格式 (Alpaca -> 对话格式)
echo ""
echo "[Step 1/2] 转换数据格式..."
echo ""

CONVERT_ARGS="--input-dir ${INPUT_DIR} --output-dir ${CONVERTED_DIR} ${USE_CHINESE}"
if [ -n "$MAX_SAMPLES_PER_TASK" ]; then
    CONVERT_ARGS="${CONVERT_ARGS} --max-samples ${MAX_SAMPLES_PER_TASK}"
fi

python3 "${SCRIPT_DIR}/convert_to_graphagent.py" ${CONVERT_ARGS}

# Step 2: 准备训练数据 (生成 pt 文件)
echo ""
echo "[Step 2/2] 准备训练数据..."
echo ""

PREPARE_ARGS="--input-dir ${CONVERTED_DIR} --output-dir ${TRAINING_DIR} --model-path ${MODEL_PATH} --max-length ${MAX_LENGTH}"
if [ -n "$MAX_SAMPLES_PER_TASK" ]; then
    PREPARE_ARGS="${PREPARE_ARGS} --max-samples ${MAX_SAMPLES_PER_TASK}"
fi

python3 "${SCRIPT_DIR}/prepare_training_data.py" ${PREPARE_ARGS}

echo ""
echo "=============================================="
echo "转换完成!"
echo "=============================================="
echo "转换后的 JSON 文件: ${CONVERTED_DIR}"
echo "训练用的 PT 文件: ${TRAINING_DIR}"
echo ""
echo "文件列表:"
ls -lh "${TRAINING_DIR}"/*.pt 2>/dev/null || echo "  (暂无 pt 文件)"
echo ""
echo "使用方法:"
echo "  将训练数据路径更新到 GraphAgent 训练配置中:"
echo "  TRAIN_DATA_PATH=${TRAINING_DIR}/graphinstruct_train_merged.pt"
