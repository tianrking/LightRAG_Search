# LightRAG 部署指南

## 硬件配置

- **GPU**: 8x NVIDIA RTX 4090 (24GB each)
- **总显存**: 192GB
- **推荐模型**: Qwen2.5-72B-Instruct

## 为什么选择 vLLM？

### vLLM vs Ollama 对比

| 特性 | vLLM | Ollama |
|------|------|--------|
| **多 GPU 并行** | ✓ 张量并行，充分利用 8 卡 | ✗ 单卡为主 |
| **并发性能** | ✓ PagedAttention，极高并发 | △ 一般 |
| **显存利用率** | ✓ 90%+ 高效利用 | △ 中等 |
| **请求吞吐** | ✓ 适合高并发场景 | △ 适合单用户 |
| **LightRAG 兼容** | ✓ 完美适配 | △ 一般 |

**结论**: 你的 8x4090 配置 + LightRAG 高并发特性 = **vLLM 是最优选择**

## 快速部署

### 1. 安装依赖

```bash
# 基础依赖
pip install vllm>=0.6.0
pip install lightrag>=0.1.0
pip install PyMuPDF>=1.23.0
pip install sentence-transformers>=2.2.0
pip install python-docx>=1.1.0

# 或者使用项目依赖
pip install -r requirements.txt
```

### 2. 启动 vLLM 服务

使用提供的优化启动脚本：

```bash
./scripts/start_vllm.sh
```

或手动启动（自定义参数）：

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --host 0.0.0.0 \
  --port PORT \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --block-size 16
```

### 3. 验证服务

```bash
# 检查 GPU 状态
python -c "from src.gpu_monitor import get_gpu_monitor; get_gpu_monitor().print_stats()"

# 测试 vLLM API
curl http://localhost:PORT/v1/models

# 测试推理
curl http://localhost:PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-72B-Instruct",
    "messages": [{"role": "user", "content": "test"}],
    "max_tokens": 50
  }'
```

### 4. 索引 业务数据 文档

```bash
python main.py index \
  --input /path/to/documents \
  --recursive
```

### 5. 查询文档

```bash
# 单次查询
python main.py query \
  --query "文档查询示例问题？" \
  --mode hybrid

# 交互模式
python main.py interactive
```

## 性能优化建议

### vLLM 参数调优

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `tensor-parallel-size` | 8 | 8卡并行 |
| `gpu-memory-utilization` | 0.90 | 预留10%显存 |
| `max-model-len` | 32768 | Qwen2.5支持32K上下文 |
| `max-num-seqs` | 256 | 最大并发请求数 |
| `max-num-batched-tokens` | 8192 | 批处理token数 |

### GPU 资源分配

```
GPU 0-7: vLLM 张量并行 (Qwen2.5-72B)
GPU 0:   Embedding 模型 (bge-m3, 与 vLLM 共享)
```

### 批处理优化

```python
# 在 config.py 中调整
class RAGConfig:
    BATCH_SIZE: int = 32  # 根据显存调整，8卡可以更大
    CHUNK_SIZE: int = 1024  # 文本块大小
```

## 文件处理性能

### 业务数据 数据集 (2,377 文件)

| 格式 | 数量 | 提取器 | 速度 |
|------|------|--------|------|
| PDF | 1,792 | PyMuPDF | 极快 |
| TXT | 3 | 原生 | 极快 |
| DOCX | 84 | python-docx | 快 |
| 图片 | 261 | 需OCR | 慢 (可选) |

**预计索引时间**: 约 30-60 分钟（取决于文档大小）

## 监控和维护

### GPU 监控

```python
from src.gpu_monitor import get_gpu_monitor

monitor = get_gpu_monitor()
monitor.print_stats()
```

### 日志查看

```bash
# vLLM 日志
tail -f logs/vllm_server.log

# LightRAG 日志
tail -f logs/lightrag.log
```

## 故障排查

### vLLM 启动失败

1. **检查显存**: 确保每卡有足够显存
   ```bash
   nvidia-smi
   ```

2. **检查 CUDA**: 确保版本兼容
   ```bash
   nvcc --version
   ```

3. **减小并行度**: 如果显存不足，减少张量并行
   ```bash
   --tensor-parallel-size 4
   ```

### 索引缓慢

1. **调整批大小**: 在 [config.py](config.py) 中增大 `BATCH_SIZE`
2. **检查文档**: 跳过超大文件或图片
3. **监控 GPU**: 确保所有 GPU 都在工作

### 查询质量差

1. **尝试不同模式**: `naive`, `local`, `global`, `hybrid`
2. **调整 chunk_size**: 在 [config.py](config.py) 中调整
3. **检查 embedding**: 确保 bge-m3 正常加载

## 后续优化

- [ ] 启用量化 (AWQ/GPTQ) 节省显存
- [ ] 部署 FastAPI 提供 REST API
- [ ] 添加 Redis 缓存层
- [ ] 实现 OCR 处理扫描文档
- [ ] 部署 Web UI (Gradio/Streamlit)
