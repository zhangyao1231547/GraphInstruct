#!/bin/bash
# 训练进度监控脚本
# 用法: bash scripts/monitor_training.sh [-w] [日志文件路径]
# -w: 持续监控模式

WATCH_MODE=false
LOG_FILE="/mnt/yrfs/GraphInstruct/model/qwen3-4b-graph-reasoning-full/trainer_log.jsonl"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -w|--watch)
            WATCH_MODE=true
            shift
            ;;
        *)
            LOG_FILE="$1"
            shift
            ;;
    esac
done

echo "========================================"
echo "  训练进度监控"
echo "========================================"
echo "日志文件: $LOG_FILE"
echo ""

if [ ! -f "$LOG_FILE" ]; then
    echo "错误: 日志文件不存在"
    echo "训练可能还在数据处理阶段，请稍后再试"
    exit 1
fi

# 获取最新一条日志
LATEST=$(tail -1 "$LOG_FILE")

# 解析JSON数据
CURRENT_STEPS=$(echo "$LATEST" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('current_steps', 'N/A'))")
TOTAL_STEPS=$(echo "$LATEST" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('total_steps', 'N/A'))")
LOSS=$(echo "$LATEST" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('loss', 0):.4f}\")")
LR=$(echo "$LATEST" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('lr', 0):.2e}\")")
EPOCH=$(echo "$LATEST" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('epoch', 0):.2f}\")")
PERCENTAGE=$(echo "$LATEST" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('percentage', 0):.1f}\")")
ELAPSED=$(echo "$LATEST" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('elapsed_time', 'N/A'))")
REMAINING=$(echo "$LATEST" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('remaining_time', 'N/A'))")

# 计算进度条
PROGRESS_WIDTH=40
FILLED=$(python3 -c "print(int($PERCENTAGE / 100 * $PROGRESS_WIDTH))")
EMPTY=$((PROGRESS_WIDTH - FILLED))
PROGRESS_BAR=$(printf "%${FILLED}s" | tr ' ' '█')$(printf "%${EMPTY}s" | tr ' ' '░')

echo "进度: [$PROGRESS_BAR] ${PERCENTAGE}%"
echo ""
echo "步数:        $CURRENT_STEPS / $TOTAL_STEPS"
echo "Epoch:       $EPOCH / 3.0"
echo "损失:        $LOSS"
echo "学习率:      $LR"
echo ""
echo "已用时间:    $ELAPSED"
echo "预计剩余:    $REMAINING"
echo ""
echo "========================================"

# 显示GPU使用情况
echo ""
echo "GPU 状态:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | while read line; do
    echo "  $line"
done
echo ""

# 可选: 持续监控模式
if [ "$2" == "-w" ] || [ "$2" == "--watch" ]; then
    echo "按 Ctrl+C 退出监控..."
    while true; do
        sleep 30
        clear
        bash "$0" "$LOG_FILE"
    done
fi
