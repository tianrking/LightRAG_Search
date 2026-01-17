#!/bin/bash
# ================================================
# vLLM 多 GPU 优化启动脚本
# 适配 8x RTX 4090 配置
# ================================================

set -e

# ============ 配置参数 ============
# 模型配置
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
TENSOR_PARALLEL_SIZE=8  # 8张4090并行

# 服务器配置
HOST="0.0.0.0"
PORT=38199
WORKERS=1

# 性能优化参数
GPU_MEMORY_UTILIZATION=0.90  # 每GPU预留10%显存
MAX_MODEL_LEN=32768          # Qwen2.5支持32K上下文
BLOCK_SIZE=16                # 块大小（页大小）
MAX_NUM_SEQS=256             # 最大并发序列数
MAX_NUM_BATCHED_TOKENS=8192  # 最大批处理token数

# 量化配置（可选，节省显存）
# QUANTIZATION="bitsandbytes"  # 取消注释启用4bit量化
QUANTIZATION=""

# 日志配置
LOG_LEVEL="INFO"
LOG_FILE="./logs/vllm_server.log"

# ============ 颜色输出 ============
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============ 工具函数 ============
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============ 检测 GPU ============
check_gpus() {
    log_info "检测 GPU 设备..."

    if ! command -v nvidia-smi &> /dev/null; then
        log_error "未找到 nvidia-smi，请检查 NVIDIA 驱动是否安装"
        exit 1
    fi

    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    log_info "检测到 ${GPU_COUNT} 个 GPU:"

    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | \
    while IFS=, read -r idx name mem; do
        echo "  GPU ${idx}: ${name} (${mem})"
    done

    if [ "$GPU_COUNT" -lt "$TENSOR_PARALLEL_SIZE" ]; then
        log_warn "GPU 数量 (${GPU_COUNT}) 少于张并行数 (${TENSOR_PARALLEL_SIZE})"
        log_warn "将使用所有 ${GPU_COUNT} 个 GPU"
        TENSOR_PARALLEL_SIZE=$GPU_COUNT
    fi
}

# ============ 检测模型 ============
check_model() {
    log_info "检查模型缓存..."

    if [ -d "$MODEL_NAME" ]; then
        log_info "使用本地模型: $MODEL_NAME"
    else
        log_info "使用 HuggingFace 模型: $MODEL_NAME"
        log_warn "首次运行需要下载模型，可能需要较长时间"
    fi
}

# ============ 创建日志目录 ============
setup_logging() {
    mkdir -p "$(dirname "$LOG_FILE")"
}

# ============ 构建启动命令 ============
build_vllm_command() {
    CMD="vllm serve $MODEL_NAME"
    CMD="$CMD --host $HOST"
    CMD="$CMD --port $PORT"
    CMD="$CMD --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
    CMD="$CMD --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
    CMD="$CMD --max-model-len $MAX_MODEL_LEN"
    CMD="$CMD --block-size $BLOCK_SIZE"
    CMD="$CMD --max-num-seqs $MAX_NUM_SEQS"
    CMD="$CMD --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS"
    CMD="$CMD --log-level $LOG_LEVEL"

    # 量化参数
    if [ -n "$QUANTIZATION" ]; then
        CMD="$CMD --quantization $QUANTIZATION"
        log_info "启用量化: $QUANTIZATION"
    fi

    # 日志文件
    CMD="$CMD >> $LOG_FILE 2>&1"

    echo "$CMD"
}

# ============ 打印配置 ============
print_config() {
    echo ""
    echo "============================================"
    echo "  vLLM 服务配置"
    echo "============================================"
    echo "  模型:                $MODEL_NAME"
    echo "  地址:                http://$HOST:$PORT/v1"
    echo "  张并行:              $TENSOR_PARALLEL_SIZE GPUs"
    echo "  GPU 显存利用率:      ${GPU_MEMORY_UTILIZATION}*100%"
    echo "  最大上下文:          $MAX_MODEL_LEN tokens"
    echo "  最大并发:            $MAX_NUM_SEQS sequences"
    echo "  批处理tokens:        $MAX_NUM_BATCHED_TOKENS"
    echo "============================================"
    echo ""
}

# ============ 启动服务 ============
start_server() {
    local cmd=$(build_vllm_command)

    log_info "启动 vLLM 服务..."
    log_info "日志文件: $LOG_FILE"
    log_info "启动命令:"
    echo "$cmd"
    echo ""

    # 启动服务
    eval $cmd

    # 如果服务异常退出
    log_error "vLLM 服务已停止，请检查日志: $LOG_FILE"
}

# ============ 主流程 ============
main() {
    echo ""
    echo "============================================"
    echo "  vLLM 多 GPU 服务启动脚本"
    echo "  LightRAG + 8x RTX 4090 最优配置"
    echo "============================================"
    echo ""

    check_gpus
    check_model
    setup_logging
    print_config

    # 询问确认
    read -p "是否启动服务? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "已取消启动"
        exit 0
    fi

    start_server
}

# ============ 信号处理 ============
trap 'log_error "收到中断信号，正在停止..."; exit 1' INT TERM

# ============ 执行 ============
main "$@"
