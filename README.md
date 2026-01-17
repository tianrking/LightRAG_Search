# LightRAG 智能文档检索系统

基于 LightRAG + vLLM 的高性能智能文档检索系统，专为 8x RTX 4090 配置优化，用于处理和分析 HKIPO 资源文件（2,377 个文档）。

## 技术栈优势

### 为什么选择 vLLM？

| 特性 | vLLM | Ollama | 推荐 |
|------|------|--------|------|
| 8x4090 并行 | ✓ 张量并行 | ✗ 单卡为主 | **vLLM** |
| 并发性能 | ✓ PagedAttention | △ 一般 | **vLLM** |
| 显存利用率 | ✓ 90%+ | △ 中等 | **vLLM** |
| LightRAG 兼容 | ✓ 完美适配 | △ 一般 | **vLLM** |

**结论**: 你的 8x4090 配置 + LightRAG 高并发特性 = **vLLM 是最优选择**

### 为什么选择 PyMuPDF？

- **性能**: 比 pdfplumber 快 10 倍+ (C++ 实现)
- **稳定**: 成熟库，bug 少
- **功能**: 支持文本、图片、表格提取

## 项目结构

```
LightRAG_Project/
├── config.py                 # 配置管理（含 GPU 配置）
├── main.py                   # 主入口 CLI
├── requirements.txt          # 依赖列表
├── README.md                 # 本文档
├── .env.example             # 环境变量模板
│
├── scripts/                  # 脚本目录
│   └── start_vllm.sh        # vLLM 8卡启动脚本
│
├── docs/                     # 文档目录
│   ├── DEPLOYMENT.md        # 部署指南
│   └── BEST_PRACTICES.md    # 最佳实践
│
├── src/                      # 源代码目录
│   ├── __init__.py          # 包初始化
│   ├── readers.py           # 文件读取器（旧版）
│   ├── extractors.py        # 优化的文档提取器 ⭐
│   ├── models.py            # LLM/Embedding 管理
│   ├── rag_engine.py        # RAG 核心引擎
│   └── gpu_monitor.py       # GPU 监控工具 ⭐
│
├── data/                     # 数据存储目录
│   └── rag_storage/         # RAG 向量数据库
│
├── logs/                     # 日志目录
│   ├── vllm_server.log      # vLLM 服务日志
│   └── lightrag.log         # LightRAG 运行日志
│
└── config/                   # 配置文件目录
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 vLLM 服务

使用提供的优化脚本（推荐）：

```bash
./scripts/start_vllm.sh
```

或手动启动：

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768

vllm serve Qwen/Qwen2.5-72B-Instruct \
  --host 0.0.0.0 \
  --port 38199 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 32768 \
  --dtype half \
  --disable-log-stats \
  --disable-custom-all-reduce

```

### 3. 验证服务

```bash
# 检查 GPU 状态
python -c "from src.gpu_monitor import get_gpu_monitor; get_gpu_monitor().print_stats()"

# 测试 vLLM API
curl http://localhost:8000/v1/models
```

### 4. 索引文档

```bash
# 索引 HKIPO 资源 (1792 PDF + 84 DOCX + ...)
python main.py index \
  --input /home/tianrui/gotohk/private_ragflow/workspace/HKIPO_resource \
  --recursive

# 或使用默认配置（已在 config.py 中设置路径）
python main.py index
```

### 5. 查询文档

```bash
# 单次查询
python main.py query --query "香港专利申请流程是什么？" --mode hybrid

# 交互模式
python main.py interactive
```

## GPU 资源分配

```
┌────────────────────────────────────────────────────────┐
│ GPU 0 │ GPU 1 │ GPU 2 │ GPU 3 │ GPU 4 │ GPU 5 │ GPU 6 │ GPU 7 │
├────────────────────────────────────────────────────────┤
│            vLLM 张量并行 (Qwen2.5-72B)                  │
│                    8x24GB = 192GB                      │
├────────────────────────────────────────────────────────┤
│   [共享]   │                                             │
│  bge-m3    │          (预留空间)                         │
│  (~2GB)    │                                             │
└────────────┴────────────────────────────────────────────┘
```

## 查询模式

LightRAG 支持四种查询模式：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `naive` | 简单向量检索 | 快速查询 |
| `local` | 局部知识图谱 | 细粒度问题 |
| `global` | 全局知识图谱 | 宏观问题 |
| `hybrid` | 混合模式（推荐） | 综合问题 |

```bash
# 示例
python main.py query --query "专利类型有哪些？" --mode local
python main.py query --query "知识产权保护趋势？" --mode global
python main.py query --query "如何申请商标？" --mode hybrid
```

## 配置说明

在 [config.py](config.py) 中可以修改：

- **路径配置**: 输入目录、工作目录、日志目录
- **LLM 配置**: 模型名称、API 地址、温度参数
- **Embedding 配置**: 模型名称、设备、向量维度
- **RAG 配置**: 查询模式、批处理大小、文本块大小
- **GPU 配置**: 张并行大小、显存利用率、并发数 ⭐

## 性能基准

### HKIPO 数据集 (2,377 文件)

| 格式 | 数量 | 提取器 | 索引时间 |
|------|------|--------|----------|
| PDF | 1,792 | PyMuPDF | ~30min |
| TXT | 3 | 原生 | <1min |
| DOCX | 84 | python-docx | ~5min |
| **总计** | **1,879** | | **~35min** |

### 查询性能

| 指标 | 数值 |
|------|------|
| 查询延迟 | 2-5s |
| 并发支持 | 100+ QPS |
| GPU 利用率 | >80% |

## 监控工具

```python
# GPU 监控
from src.gpu_monitor import get_gpu_monitor

monitor = get_gpu_monitor()
monitor.print_stats()

# 检查可用性
if monitor.check_availability(required_memory_mb=20000, gpu_count=8):
    print("✓ GPU 资源充足")
```

## 文档

- [部署指南](docs/DEPLOYMENT.md) - 详细部署步骤
- [最佳实践](docs/BEST_PRACTICES.md) - 架构设计和优化建议

## 后续优化

- [ ] 启用量化 (AWQ/GPTQ) 节省显存
- [ ] 部署 FastAPI 提供 REST API
- [ ] 添加 Redis 缓存层
- [ ] 实现 OCR 处理扫描文档
- [ ] 部署 Web UI (Gradio/Streamlit)
- [ ] 增量索引更新

## 常见问题

### Q: 为什么不用 Ollama？

A: Ollama 适合个人单用户，你的 8 卡配置用 Ollama 是浪费。vLLM 的张量并行和 PagedAttention 能充分利用 8 卡性能，与 LightRAG 的高并发特性完美配合。

### Q: GPU 资源不足怎么办？

A: 可以减少张并行数或启用量化：
```bash
# 4卡并行
--tensor-parallel-size 4

# 启用量化
--quantization bitsandbytes
```

### Q: 如何处理扫描的 PDF？

A: 需要添加 OCR 支持，参考 [docs/BEST_PRACTICES.md](docs/BEST_PRACTICES.md)

## License

MIT
