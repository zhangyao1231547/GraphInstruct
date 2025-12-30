#!/bin/bash
# GraphAgent 完整评测流程脚本
# 1. 从GraphInstruct测试数据采样19个任务各10条数据
# 2. 转换为GraphAgent数据格式
# 3. 运行GraphAgent评测
# 4. 生成评测报告

set -e

# =============================================================================
# 配置参数
# =============================================================================

# 测试数据目录
TEST_DATA_DIR="/nvme0/work/workspaces-zy/GraphInstruct/data/test"

# 评测数据输出路径
EVAL_DATA_PATH="/nvme0/work/workspaces-zy/GraphInstruct/data/eval/graphinstruct_eval_19tasks_10samples.pt"

# 模型路径
BASE_MODEL="/nvme0/work/workspaces-zy/model/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507"
CHECKPOINT_PATH="/mnt/yrfs/GraphAgent_model/zy/model/graphagent-qwen3/stage2-epoch5-8192-graphinstruct-full_finetune/lightning_logs/version_0/model_epoch=3-step=23748.ckpt/pytorch_model.bin"

# 评测结果输出目录
OUTPUT_DIR="/nvme0/work/workspaces-zy/GraphInstruct/data/eval/results_$(date +%Y%m%d_%H%M%S)"

# GPU配置
GPUS="0"

# 每个任务的采样数量
SAMPLES_PER_TASK=10

# 评测参数
BATCH_SIZE=1
MAX_NEW_TOKENS=512
MAX_LENGTH=4096

# =============================================================================
# 解析命令行参数
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --samples-per-task)
            SAMPLES_PER_TASK="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --skip-data-prep)
            SKIP_DATA_PREP=true
            shift
            ;;
        --eval-data)
            EVAL_DATA_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=${GPUS}

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPHAGENT_DIR="/nvme0/work/workspaces-zy/GraphAgent-zy/GraphAGent-training"

echo "============================================"
echo "GraphAgent 评测流程"
echo "============================================"
echo "测试数据目录: ${TEST_DATA_DIR}"
echo "评测数据路径: ${EVAL_DATA_PATH}"
echo "基础模型: ${BASE_MODEL}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "输出目录: ${OUTPUT_DIR}"
echo "GPUs: ${GPUS}"
echo "每任务采样数: ${SAMPLES_PER_TASK}"
echo "============================================"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname ${EVAL_DATA_PATH})"

# =============================================================================
# 步骤1: 准备评测数据
# =============================================================================
if [ "${SKIP_DATA_PREP}" != "true" ]; then
    echo ""
    echo "============================================"
    echo "步骤1: 准备评测数据"
    echo "============================================"

    cd "${SCRIPT_DIR}"

    python prepare_evaluation_data.py \
        --test-data-dir "${TEST_DATA_DIR}" \
        --output-path "${EVAL_DATA_PATH}" \
        --model-path "${BASE_MODEL}" \
        --samples-per-task ${SAMPLES_PER_TASK} \
        --max-length ${MAX_LENGTH} \
        --seed 42

    if [ $? -ne 0 ]; then
        echo "ERROR: 数据准备失败"
        exit 1
    fi

    echo "评测数据已保存到: ${EVAL_DATA_PATH}"
else
    echo ""
    echo "跳过数据准备步骤 (使用已有数据: ${EVAL_DATA_PATH})"
fi

# =============================================================================
# 步骤2: 运行GraphAgent评测
# =============================================================================
echo ""
echo "============================================"
echo "步骤2: 运行GraphAgent评测"
echo "============================================"

cd "${GRAPHAGENT_DIR}"

python -u pl_eval.py \
    --version qwen \
    --model_name_or_path "${BASE_MODEL}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --eval_data_path "${EVAL_DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --bf16 True \
    --model_max_length ${MAX_LENGTH} \
    --use_graph_start_end True \
    --gpus ${GPUS} \
    --verbose True

eval_status=$?

if [ $eval_status -ne 0 ]; then
    echo "ERROR: 评测失败,退出码: $eval_status"
    exit 1
fi

# =============================================================================
# 步骤3: 生成评测报告
# =============================================================================
echo ""
echo "============================================"
echo "步骤3: 生成评测报告"
echo "============================================"

cd "${SCRIPT_DIR}"

python generate_evaluation_report.py \
    --results-dir "${OUTPUT_DIR}" \
    --output-report "${OUTPUT_DIR}/evaluation_report.md"

if [ $? -ne 0 ]; then
    echo "WARNING: 报告生成失败,但评测结果已保存"
fi

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "============================================"
echo "评测完成!"
echo "============================================"
echo "结果目录: ${OUTPUT_DIR}"
echo ""
echo "生成的文件:"
echo "  - metrics.json: 汇总评测指标"
echo "  - detailed_results.json: 每个样本的详细预测结果"
echo "  - sample_predictions.txt: 可读格式的样本预测"
echo "  - mismatches.txt: 错误预测分析"
echo "  - evaluation_report.md: 评测报告"
echo "============================================"

# 显示简要统计
if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
    echo ""
    echo "评测摘要:"
    cat "${OUTPUT_DIR}/metrics.json" | python -c "
import sys, json
data = json.load(sys.stdin)
print(f'  总样本数: {data[\"total\"]}')
print(f'  精确匹配: {data[\"exact_match\"]} ({data[\"exact_match_rate\"]*100:.2f}%)')
print(f'  部分匹配: {data[\"partial_match\"]} ({data[\"partial_match_rate\"]*100:.2f}%)')
print()
print('  各任务准确率:')
for task, metrics in sorted(data['by_task_type'].items()):
    print(f'    {task}: {metrics[\"exact_match\"]}/{metrics[\"total\"]} ({metrics[\"exact_match_rate\"]*100:.1f}%)')
"
fi
