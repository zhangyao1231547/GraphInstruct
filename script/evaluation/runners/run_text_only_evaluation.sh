#!/bin/bash
# 纯文本评测脚本 - 使用原始Qwen3-4B模型(无图编码)
# 用于baseline对比

set -e

# 配置
MODEL_PATH="/nvme0/work/workspaces-zy/model/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507"
EVAL_DATA="/nvme0/work/workspaces-zy/GraphInstruct/data/eval/graphinstruct_eval_19tasks_10samples_text.json"
OUTPUT_DIR="/nvme0/work/workspaces-zy/GraphInstruct/data/eval/results_text_only_19tasks"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "Text-Only Evaluation (Baseline) - Qwen3-4B"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo "Eval Data: $EVAL_DATA"
echo "Output Dir: $OUTPUT_DIR"
echo "============================================================"

# 检查评测数据
if [ ! -f "$EVAL_DATA" ]; then
    echo "ERROR: Evaluation data not found at $EVAL_DATA"
    echo "Run prepare_text_only_evaluation.py first"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行评测
echo ""
echo "Step 1: Running text-only evaluation..."
CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/evaluate_text_only.py" \
    --model-path "$MODEL_PATH" \
    --eval-data-path "$EVAL_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --max-new-tokens 512 \
    --bf16

# 生成报告
echo ""
echo "Step 2: Generating evaluation report..."
python "$SCRIPT_DIR/generate_text_only_report.py" \
    --results-dir "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "Report: $OUTPUT_DIR/evaluation_report.md"
echo "============================================================"
