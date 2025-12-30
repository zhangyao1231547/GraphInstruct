#!/bin/bash
# API模型评测脚本 - DeepSeek, Doubao, Qwen-max
# 使用修复后的评测逻辑（集合匹配、数值容差、遍历验证）

SCRIPT_DIR="/nvme0/work/workspaces-zy/GraphInstruct/script"
EVAL_DATA="/nvme0/work/workspaces-zy/GraphInstruct/data/eval/graphinstruct_eval_19tasks_10samples_text.json"
OUTPUT_BASE="/nvme0/work/workspaces-zy/GraphInstruct/data/eval/api_results"

# API Keys
VOLCANO_KEY="e1d32c08-96c2-442e-8198-1930d8b71a07"
ALIYUN_KEY="sk-7c0124f3d3084de8802c153768f2f57c"

# 创建输出目录
mkdir -p ${OUTPUT_BASE}

# 设置max_tokens为4096
MAX_TOKENS=4096

echo "=============================================="
echo "API Model Evaluation with Enhanced Logic"
echo "=============================================="
echo "Evaluation Data: ${EVAL_DATA}"
echo "Max Tokens: ${MAX_TOKENS}"
echo "Output Base: ${OUTPUT_BASE}"
echo "=============================================="

# 运行DeepSeek评测（通过火山引擎）
echo ""
echo "[1/3] Starting DeepSeek V3 evaluation..."
nohup python ${SCRIPT_DIR}/evaluate_api_model.py \
    --provider volcano \
    --api-key "${VOLCANO_KEY}" \
    --model "deepseek-v3-2-251201" \
    --eval-data-path "${EVAL_DATA}" \
    --output-dir "${OUTPUT_BASE}/deepseek-v3" \
    --max-tokens ${MAX_TOKENS} \
    --rate-limit 0.5 \
    --verbose \
    --resume \
    > /tmp/eval_deepseek.log 2>&1 &
DEEPSEEK_PID=$!
echo "DeepSeek PID: ${DEEPSEEK_PID}"

# 运行Doubao评测
echo ""
echo "[2/3] Starting Doubao evaluation..."
nohup python ${SCRIPT_DIR}/evaluate_api_model.py \
    --provider volcano \
    --api-key "${VOLCANO_KEY}" \
    --model "doubao-seed-1-8-251215" \
    --eval-data-path "${EVAL_DATA}" \
    --output-dir "${OUTPUT_BASE}/doubao-seed" \
    --max-tokens ${MAX_TOKENS} \
    --rate-limit 0.5 \
    --verbose \
    --resume \
    > /tmp/eval_doubao.log 2>&1 &
DOUBAO_PID=$!
echo "Doubao PID: ${DOUBAO_PID}"

# 运行Qwen-max评测
echo ""
echo "[3/3] Starting Qwen-max evaluation..."
nohup python ${SCRIPT_DIR}/evaluate_api_model.py \
    --provider aliyun \
    --api-key "${ALIYUN_KEY}" \
    --model "qwen-max" \
    --eval-data-path "${EVAL_DATA}" \
    --output-dir "${OUTPUT_BASE}/qwen-max" \
    --max-tokens ${MAX_TOKENS} \
    --rate-limit 0.5 \
    --verbose \
    --resume \
    > /tmp/eval_qwen.log 2>&1 &
QWEN_PID=$!
echo "Qwen-max PID: ${QWEN_PID}"

echo ""
echo "=============================================="
echo "All evaluations started in background!"
echo "=============================================="
echo "Monitor logs:"
echo "  DeepSeek: tail -f /tmp/eval_deepseek.log"
echo "  Doubao:   tail -f /tmp/eval_doubao.log"
echo "  Qwen:     tail -f /tmp/eval_qwen.log"
echo ""
echo "Check progress:"
echo "  ps aux | grep evaluate_api_model"
echo "=============================================="
