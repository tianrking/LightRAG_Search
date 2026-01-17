#!/bin/bash
# ================================================
# LightRAG 完整启动流程
# ================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}==>${NC} $1"
}

# 项目目录
PROJECT_DIR="/home/tianrui/gotohk/private_ragflow/workspace/LightRAG_Project"
cd "$PROJECT_DIR"

# ================================================
# 步骤 1: 检查环境
# ================================================
log_step "步骤 1: 检查运行环境"

# 检查 Python
if ! command -v python &> /dev/null; then
    log_error "Python 未安装"
    exit 1
fi
log_info "Python 版本: $(python --version)"

# 检查 GPU
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi 不可用，请检查 NVIDIA 驱动"
    exit 1
fi
log_info "GPU 状态:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read line; do
    log_info "  $line"
done

# 检查 vLLM
if ! command -v vllm &> /dev/null; then
    log_error "vllm 未安装，请先运行: pip install vllm"
    exit 1
fi
log_info "vLLM 已安装: $(vllm --version)"

# 检查依赖
log_info "检查 Python 依赖..."
python -c "import lightrag, PyMuPDF, sentence_transformers" 2>/dev/null || {
    log_error "缺少依赖，请运行: pip install -r requirements.txt"
    exit 1
}
log_info "所有依赖已安装"

echo ""

# ================================================
# 步骤 2: 启动 vLLM 服务
# ================================================
log_step "步骤 2: 启动 vLLM 服务"

# 检查 vLLM 是否已运行
if curl -s http://localhost:38199/v1/models > /dev/null 2>&1; then
    log_warn "vLLM 服务已在运行 (http://localhost:38199)"
    read -p "是否重启? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "正在停止旧的 vLLM 服务..."
        pkill -f "vllm serve" || true
        sleep 5
    else
        log_info "使用现有的 vLLM 服务"
        vllm_running=true
    fi
fi

if [ "$vllm_running" != true ]; then
    log_info "启动 vLLM 服务（8x4090 并行）..."
    log_info "这可能需要几分钟时间来加载模型..."

    # 使用启动脚本
    if [ -f "./scripts/start_vllm.sh" ]; then
        ./scripts/start_vllm.sh &
        VLLM_PID=$!
    else
        # 手动启动
        vllm serve Qwen/Qwen2.5-72B-Instruct \
          --host 0.0.0.0 \
          --port 8000 \
          --tensor-parallel-size 8 \
          --gpu-memory-utilization 0.90 \
          --max-model-len 32768 \
          --max-num-seqs 256 \
          --max-num-batched-tokens 8192 \
          > logs/vllm_server.log 2>&1 &
        VLLM_PID=$!
    fi

    log_info "vLLM PID: $VLLM_PID"

    # 等待服务启动
    log_info "等待 vLLM 服务启动..."
    max_wait=300  # 最多等待5分钟
    waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s http://localhost:38199/v1/models > /dev/null 2>&1; then
            log_info "vLLM 服务启动成功！"
            break
        fi
        echo -n "."
        sleep 2
        waited=$((waited + 2))
    done
    echo

    if [ $waited -ge $max_wait ]; then
        log_error "vLLM 服务启动超时，请检查日志: logs/vllm_server.log"
        exit 1
    fi
fi

echo ""

# ================================================
# 步骤 3: 验证服务
# ================================================
log_step "步骤 3: 验证服务状态"

# 测试 vLLM API
log_info "测试 vLLM API..."
response=$(curl -s http://localhost:38199/v1/models)
if echo "$response" | grep -q "object"; then
    log_info "vLLM API 响应正常"
else
    log_error "vLLM API 响应异常"
    exit 1
fi

# 测试 GPU 监控
log_info "测试 GPU 监控..."
python -c "from src.gpu_monitor import get_gpu_monitor; get_gpu_monitor().print_stats()" 2>/dev/null || true

echo ""

# ================================================
# 步骤 4: 索引文档
# ================================================
log_step "步骤 4: 索引 HKIPO 文档"

read -p "是否现在开始索引文档? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    log_info "开始并发索引（4个文件同时处理）..."
    log_info "按 Ctrl+C 可随时中断（支持断点续传）"
    echo ""

    python main.py index \
      --input /home/tianrui/gotohk/private_ragflow/workspace/HKIPO_resource/HKIPO-中介公共资料 \
      --concurrent \
      --workers 4

    echo ""
    log_info "索引完成！"
else
    log_info "跳过索引，稍后可手动运行:"
    echo "  python main.py index --concurrent --workers 4"
fi

echo ""

# ================================================
# 步骤 5: 查询测试
# ================================================
log_step "步骤 5: 查询测试"

read -p "是否进行查询测试? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    log_info "测试查询: 香港专利申请流程"
    python main.py query --query "香港专利申请流程" --mode hybrid
    echo ""
fi

# ================================================
# 完成
# ================================================
echo ""
log_info "========================================"
log_info "LightRAG 系统启动完成！"
log_info "========================================"
echo ""
echo "vLLM 服务:  http://localhost:38199/v1"
echo ""
echo "常用命令:"
echo "  # 查询文档"
echo "  python main.py query --query \"你的问题\" --mode hybrid"
echo ""
echo "  # 交互模式"
echo "  python main.py interactive"
echo ""
echo "  # 查看 GPU 状态"
echo "  python -c \"from src.gpu_monitor import get_gpu_monitor; get_gpu_monitor().print_stats()\""
echo ""
echo "  # 查看日志"
echo "  tail -f logs/vllm_server.log   # vLLM 日志"
echo "  tail -f logs/lightrag.log      # LightRAG 日志"
echo ""
echo "日志文件:"
echo "  vLLM:     logs/vllm_server.log"
echo "  LightRAG: logs/lightrag.log"
echo "  进度:     data/index_progress/"
echo ""
