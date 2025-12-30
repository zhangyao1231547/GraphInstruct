#!/bin/bash
# GraphInstruct 数据转换运行脚本
# 将 GraphInstruct 数据转换为 GraphAgent 训练格式
# 支持 LLaMA 和 Qwen 两种模型格式

set -e

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="${PROJECT_DIR}/LLaMAFactory/data/reasoning"
CONVERTED_DIR="${PROJECT_DIR}/data/converted"
TRAINING_BASE_DIR="${PROJECT_DIR}/data/training"
LLAMA_TRAINING_DIR="${TRAINING_BASE_DIR}/llama"
QWEN_TRAINING_DIR="${TRAINING_BASE_DIR}/qwen"

# 默认模型类型
MODEL_TYPE="${MODEL_TYPE:-llama}"  # llama 或 qwen

# 模型路径配置
LLAMA_MODEL_PATH="/nvme0/work/workspaces-zy/model/Llama-3.1-8B-Instruct/LLM-Research/Meta-Llama-3.1-8B-Instruct"
QWEN_MODEL_PATH="/nvme0/work/workspaces-zy/model/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --llama)
            MODEL_TYPE="llama"
            shift
            ;;
        --qwen)
            MODEL_TYPE="qwen"
            shift
            ;;
        --both)
            MODEL_TYPE="both"
            shift
            ;;
        --max-samples)
            MAX_SAMPLES_PER_TASK="$2"
            shift 2
            ;;
        --chinese)
            USE_CHINESE="--chinese"
            shift
            ;;
        --english)
            USE_CHINESE="--english"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --llama        Generate data for LLaMA models (default)"
            echo "  --qwen         Generate data for Qwen models"
            echo "  --both         Generate data for both LLaMA and Qwen"
            echo "  --max-samples N  Limit samples per task (for testing)"
            echo "  --chinese      Use Chinese prompts (default)"
            echo "  --english      Use English prompts"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 默认使用中文
USE_CHINESE="${USE_CHINESE:---chinese}"

echo "=============================================="
echo "GraphInstruct -> GraphAgent 数据转换"
echo "=============================================="
echo "输入目录: ${INPUT_DIR}"
echo "转换输出: ${CONVERTED_DIR}"
echo "训练输出: ${TRAINING_BASE_DIR}"
echo "  - LLaMA: ${LLAMA_TRAINING_DIR}"
echo "  - Qwen:  ${QWEN_TRAINING_DIR}"
echo "模型类型: ${MODEL_TYPE}"
echo "=============================================="

# 创建输出目录
mkdir -p "${CONVERTED_DIR}"
mkdir -p "${LLAMA_TRAINING_DIR}"
mkdir -p "${QWEN_TRAINING_DIR}"

# Step 1: 转换数据格式 (Alpaca -> 对话格式)
echo ""
echo "[Step 1] 转换数据格式..."
echo ""

CONVERT_ARGS="--input-dir ${INPUT_DIR} --output-dir ${CONVERTED_DIR} ${USE_CHINESE}"
if [ -n "$MAX_SAMPLES_PER_TASK" ]; then
    CONVERT_ARGS="${CONVERT_ARGS} --max-samples ${MAX_SAMPLES_PER_TASK}"
fi

python3 "${SCRIPT_DIR}/convert_to_graphagent.py" ${CONVERT_ARGS}

# Step 2: 准备训练数据
if [ "$MODEL_TYPE" = "llama" ] || [ "$MODEL_TYPE" = "both" ]; then
    echo ""
    echo "[Step 2a] 准备 LLaMA 训练数据..."
    echo ""

    PREPARE_ARGS="--input-dir ${CONVERTED_DIR} --output-dir ${LLAMA_TRAINING_DIR} --model-path ${LLAMA_MODEL_PATH} --max-length 4096"
    if [ -n "$MAX_SAMPLES_PER_TASK" ]; then
        PREPARE_ARGS="${PREPARE_ARGS} --max-samples ${MAX_SAMPLES_PER_TASK}"
    fi

    python3 "${SCRIPT_DIR}/prepare_training_data.py" ${PREPARE_ARGS}
fi

if [ "$MODEL_TYPE" = "qwen" ] || [ "$MODEL_TYPE" = "both" ]; then
    echo ""
    echo "[Step 2b] 准备 Qwen 训练数据..."
    echo ""

    PREPARE_ARGS="--input-dir ${CONVERTED_DIR} --output-dir ${QWEN_TRAINING_DIR} --model-path ${QWEN_MODEL_PATH} --max-length 4096"
    if [ -n "$MAX_SAMPLES_PER_TASK" ]; then
        PREPARE_ARGS="${PREPARE_ARGS} --max-samples ${MAX_SAMPLES_PER_TASK}"
    fi

    python3 "${SCRIPT_DIR}/prepare_training_data_qwen.py" ${PREPARE_ARGS}
fi

echo ""
echo "=============================================="
echo "转换完成!"
echo "=============================================="
echo "转换后的 JSON 文件: ${CONVERTED_DIR}"
echo "训练用的 PT 文件:"
echo ""
if [ "$MODEL_TYPE" = "llama" ] || [ "$MODEL_TYPE" = "both" ]; then
    echo "=== LLaMA 格式 (${LLAMA_TRAINING_DIR}) ==="
    ls -lh "${LLAMA_TRAINING_DIR}"/*.pt 2>/dev/null || echo "  (暂无 pt 文件)"
    echo ""
fi
if [ "$MODEL_TYPE" = "qwen" ] || [ "$MODEL_TYPE" = "both" ]; then
    echo "=== Qwen 格式 (${QWEN_TRAINING_DIR}) ==="
    ls -lh "${QWEN_TRAINING_DIR}"/*.pt 2>/dev/null || echo "  (暂无 pt 文件)"
    echo ""
fi
echo "使用方法:"
if [ "$MODEL_TYPE" = "llama" ] || [ "$MODEL_TYPE" = "both" ]; then
    echo "  LLaMA 训练: TRAIN_DATA_PATH=${LLAMA_TRAINING_DIR}/graphinstruct_train_merged.pt"
fi
if [ "$MODEL_TYPE" = "qwen" ] || [ "$MODEL_TYPE" = "both" ]; then
    echo "  Qwen 训练:  TRAIN_DATA_PATH=${QWEN_TRAINING_DIR}/graphinstruct_train_qwen.pt"
fi
