#!/bin/bash
# 中文评测脚本 - GraphAgent 图模型评测和 LLM 纯文本评测
#
# Usage:
#   # 完整评测 (生成数据 + 图模型评测 + 文本评测 + 报告)
#   bash run_chinese_evaluation.sh
#
#   # 只生成数据
#   bash run_chinese_evaluation.sh --generate-only
#
#   # 只运行图模型评测
#   bash run_chinese_evaluation.sh --graph-only
#
#   # 只运行文本评测
#   bash run_chinese_evaluation.sh --text-only
#
#   # 只生成报告
#   bash run_chinese_evaluation.sh --report-only

set -e

# ============================================================
# 配置
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GRAPHAGENT_ROOT="/nvme0/work/workspaces-zy/GraphAgent-zy"

# 模型路径
MODEL_PATH="/nvme0/work/workspaces-zy/model/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507"
CHECKPOINT_PATH="/mnt/yrfs/GraphAgent_model/zy/model/graphagent-qwen3/stage2-epoch5-8192-graphinstruct-full_finetune/lightning_logs/version_0/model_epoch=3-step=23748.ckpt/pytorch_model.bin"

# 数据路径
SAMPLES_PER_TASK=10
EVAL_DATA_DIR="$PROJECT_ROOT/data/eval"
GRAPH_EVAL_DATA="$EVAL_DATA_DIR/graphinstruct_zh_19tasks_${SAMPLES_PER_TASK}samples.pt"
TEXT_EVAL_DATA="$EVAL_DATA_DIR/graphinstruct_zh_19tasks_${SAMPLES_PER_TASK}samples_text.json"

# 输出目录
GRAPH_OUTPUT_DIR="$EVAL_DATA_DIR/results_zh_graph_8192tokens"
TEXT_OUTPUT_DIR="$EVAL_DATA_DIR/results_zh_text_only"

# 评测参数
MAX_NEW_TOKENS=8192
GPU_ID=0

# ============================================================
# 解析参数
# ============================================================
GENERATE_ONLY=false
GRAPH_ONLY=false
TEXT_ONLY=false
REPORT_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --generate-only)
            GENERATE_ONLY=true
            shift
            ;;
        --graph-only)
            GRAPH_ONLY=true
            shift
            ;;
        --text-only)
            TEXT_ONLY=true
            shift
            ;;
        --report-only)
            REPORT_ONLY=true
            shift
            ;;
        --samples)
            SAMPLES_PER_TASK=$2
            GRAPH_EVAL_DATA="$EVAL_DATA_DIR/graphinstruct_zh_19tasks_${SAMPLES_PER_TASK}samples.pt"
            TEXT_EVAL_DATA="$EVAL_DATA_DIR/graphinstruct_zh_19tasks_${SAMPLES_PER_TASK}samples_text.json"
            shift 2
            ;;
        --max-tokens)
            MAX_NEW_TOKENS=$2
            GRAPH_OUTPUT_DIR="$EVAL_DATA_DIR/results_zh_graph_${MAX_NEW_TOKENS}tokens"
            shift 2
            ;;
        --gpu)
            GPU_ID=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================
# 打印配置
# ============================================================
echo "============================================================"
echo "中文评测配置"
echo "============================================================"
echo "模型路径: $MODEL_PATH"
echo "检查点: $CHECKPOINT_PATH"
echo "每任务样本数: $SAMPLES_PER_TASK"
echo "最大生成token: $MAX_NEW_TOKENS"
echo "GPU: $GPU_ID"
echo "图评测数据: $GRAPH_EVAL_DATA"
echo "文本评测数据: $TEXT_EVAL_DATA"
echo "图评测输出: $GRAPH_OUTPUT_DIR"
echo "文本评测输出: $TEXT_OUTPUT_DIR"
echo "============================================================"

# ============================================================
# Step 1: 生成中文评测数据
# ============================================================
generate_data() {
    echo ""
    echo "============================================================"
    echo "Step 1: 生成中文评测数据"
    echo "============================================================"

    cd "$PROJECT_ROOT"

    if [ -f "$GRAPH_EVAL_DATA" ] && [ -f "$TEXT_EVAL_DATA" ]; then
        echo "评测数据已存在，跳过生成"
        echo "  - $GRAPH_EVAL_DATA"
        echo "  - $TEXT_EVAL_DATA"
    else
        echo "生成中文评测数据..."
        python "$SCRIPT_DIR/create_chinese_eval_data.py" \
            --output-dir "$EVAL_DATA_DIR" \
            --model-path "$MODEL_PATH" \
            --samples-per-task $SAMPLES_PER_TASK \
            --format both \
            --max-length 4096

        echo "数据生成完成"
    fi
}

# ============================================================
# Step 2: GraphAgent 图模型评测
# ============================================================
run_graph_evaluation() {
    echo ""
    echo "============================================================"
    echo "Step 2: GraphAgent 图模型评测 (中文)"
    echo "============================================================"

    if [ ! -f "$GRAPH_EVAL_DATA" ]; then
        echo "ERROR: 图评测数据不存在: $GRAPH_EVAL_DATA"
        echo "请先运行数据生成步骤"
        exit 1
    fi

    mkdir -p "$GRAPH_OUTPUT_DIR"

    echo "运行GraphAgent评测..."
    cd "$GRAPHAGENT_ROOT/GraphAGent-training"

    CUDA_VISIBLE_DEVICES=$GPU_ID python -u pl_eval.py \
        --version qwen \
        --model_name_or_path "$MODEL_PATH" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --eval_data_path "$GRAPH_EVAL_DATA" \
        --output_dir "$GRAPH_OUTPUT_DIR" \
        --batch_size 1 \
        --max_new_tokens $MAX_NEW_TOKENS \
        --bf16 True \
        --model_max_length 4096 \
        --use_graph_start_end True \
        --gpus 0 \
        --verbose False

    echo "GraphAgent评测完成"
    echo "结果保存在: $GRAPH_OUTPUT_DIR"
}

# ============================================================
# Step 3: LLM 纯文本评测
# ============================================================
run_text_evaluation() {
    echo ""
    echo "============================================================"
    echo "Step 3: LLM 纯文本评测 (中文)"
    echo "============================================================"

    if [ ! -f "$TEXT_EVAL_DATA" ]; then
        echo "ERROR: 文本评测数据不存在: $TEXT_EVAL_DATA"
        echo "请先运行数据生成步骤"
        exit 1
    fi

    mkdir -p "$TEXT_OUTPUT_DIR"

    echo "运行纯文本评测..."
    cd "$PROJECT_ROOT"

    CUDA_VISIBLE_DEVICES=$GPU_ID python "$SCRIPT_DIR/evaluate_text_only.py" \
        --model-path "$MODEL_PATH" \
        --eval-data-path "$TEXT_EVAL_DATA" \
        --output-dir "$TEXT_OUTPUT_DIR" \
        --max-new-tokens $MAX_NEW_TOKENS \
        --bf16

    echo "纯文本评测完成"
    echo "结果保存在: $TEXT_OUTPUT_DIR"
}

# ============================================================
# Step 4: 生成评测报告
# ============================================================
generate_reports() {
    echo ""
    echo "============================================================"
    echo "Step 4: 生成评测报告"
    echo "============================================================"

    cd "$PROJECT_ROOT"

    # 生成GraphAgent报告
    if [ -d "$GRAPH_OUTPUT_DIR" ] && [ -f "$GRAPH_OUTPUT_DIR/detailed_results.json" ]; then
        echo "生成GraphAgent评测报告..."
        python "$SCRIPT_DIR/generate_graphagent_report.py" \
            --results-dir "$GRAPH_OUTPUT_DIR" \
            --max-new-tokens $MAX_NEW_TOKENS \
            --language zh
        echo "GraphAgent报告: $GRAPH_OUTPUT_DIR/evaluation_report.md"
    else
        echo "跳过GraphAgent报告生成 (结果文件不存在)"
    fi

    # 生成文本评测报告
    if [ -d "$TEXT_OUTPUT_DIR" ] && [ -f "$TEXT_OUTPUT_DIR/detailed_results.json" ]; then
        echo "生成纯文本评测报告..."
        python "$SCRIPT_DIR/generate_text_only_report.py" \
            --results-dir "$TEXT_OUTPUT_DIR" \
            --language zh
        echo "文本评测报告: $TEXT_OUTPUT_DIR/evaluation_report.md"
    else
        echo "跳过文本评测报告生成 (结果文件不存在)"
    fi
}

# ============================================================
# 主流程
# ============================================================
main() {
    if [ "$GENERATE_ONLY" = true ]; then
        generate_data
    elif [ "$GRAPH_ONLY" = true ]; then
        run_graph_evaluation
        generate_reports
    elif [ "$TEXT_ONLY" = true ]; then
        run_text_evaluation
        generate_reports
    elif [ "$REPORT_ONLY" = true ]; then
        generate_reports
    else
        # 完整流程
        generate_data
        run_graph_evaluation
        run_text_evaluation
        generate_reports
    fi

    echo ""
    echo "============================================================"
    echo "中文评测完成!"
    echo "============================================================"
    echo ""
    echo "输出文件:"

    if [ -f "$GRAPH_EVAL_DATA" ]; then
        echo "  图评测数据: $GRAPH_EVAL_DATA"
    fi
    if [ -f "$TEXT_EVAL_DATA" ]; then
        echo "  文本评测数据: $TEXT_EVAL_DATA"
    fi
    if [ -f "$GRAPH_OUTPUT_DIR/evaluation_report.md" ]; then
        echo "  图评测报告: $GRAPH_OUTPUT_DIR/evaluation_report.md"
    fi
    if [ -f "$TEXT_OUTPUT_DIR/evaluation_report.md" ]; then
        echo "  文本评测报告: $TEXT_OUTPUT_DIR/evaluation_report.md"
    fi
    echo ""
}

main
