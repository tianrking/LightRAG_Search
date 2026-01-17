# LightRAG 智能文档检索系统

基于 LightRAG 的高性能智能文档检索系统，支持大规模文档的语义检索和知识图谱构建。

## 技术栈

### 核心组件

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| **RAG 框架** | LightRAG | 支持知识图谱 + 向量检索 |
| **LLM 推理** | vLLM | 高性能推理服务 |
| **Embedding** | BGE-M3 | 多语言语义向量 |
| **文档提取** | PyMuPDF | 高性能 PDF 处理 |
| **OCR 引擎** | RapidOCR/PaddleOCR | 扫描文档识别 |

### vLLM vs Ollama 选择

| 特性 | vLLM | Ollama |
|------|------|--------|
| **多 GPU 支持** | ✓ 原生张量并行 | △ 主要面向单卡 |
| **并发性能** | ✓ PagedAttention | △ 一般 |
| **显存利用率** | ✓ 90%+ | △ 中等 |
| **生产环境** | ✓ 云原生部署 | △ 主要面向本地 |
| **适用场景** | 大规模并发服务 | 个人开发测试 |

**说明**: vLLM 更适合高并发的生产环境，Ollama 更适合个人开发。根据实际需求选择合适的工具。

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
│   └── start_vllm.sh        # vLLM 启动脚本
│
├── docs/                     # 文档目录
│   ├── USAGE.md             # 使用说明
│   ├── DEPLOYMENT.md        # 部署指南
│   ├── BEST_PRACTICES.md    # 最佳实践
│   ├── ARCHITECTURE_ANALYSIS.md  # 架构分析
│   ├── QWEN_USAGE_FLOW.md   # Qwen 模型使用流程
│   └── LIGHTRAG_OPTIMIZATION.md  # LightRAG 优化指南
│
├── src/                      # 源代码目录
│   ├── __init__.py          # 包初始化
│   ├── extractors.py        # 文档提取器（支持多 OCR 引擎）
│   ├── models.py            # LLM/Embedding 管理
│   ├── rag_engine.py        # RAG 核心引擎
│   ├── index_manager.py     # 并发索引管理器
│   └── gpu_monitor.py       # GPU 监控工具
│
├── backend/                  # 后端服务（HTTP API）
│   ├── api.py               # API 接口定义
│   ├── service.py           # 业务逻辑实现
│   ├── server.py            # HTTP 服务器
│   └── client.py            # 客户端示例
│
├── data/                     # 数据存储目录
│   ├── rag_storage/         # RAG 向量数据库
│   └── index_progress/      # 索引进度记录
│
└── logs/                     # 日志目录
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 LLM 服务

使用 vLLM（推荐用于生产环境）：

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --host 0.0.0.0 \
  --port 38199 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 32768 \
  --dtype half \
  --disable-log-stats
```

或使用 Ollama（适合个人开发）：

```bash
ollama run Qwen2.5-72B-Instruct
```

### 3. 索引文档

```bash
# 索引文档目录（支持递归扫描）
python main.py index \
  --input /path/to/documents \
  --recursive

# 或使用默认配置
python main.py index
```

### 4. 查询文档

```bash
# 单次查询
python main.py query --query "查询问题" --mode hybrid

# 交互模式
python main.py interactive
```

## 功能特性

### 文档处理能力

| 格式 | 支持情况 | 处理方式 |
|------|----------|----------|
| PDF | ✅ | PyMuPDF + OCR 降级 |
| TXT | ✅ | 原生读取 |
| DOCX | ✅ | python-docx |
| 其他 | ⚠️ | 可扩展 |

### OCR 引擎支持

| 引擎 | 速度 | 精度 | 适用场景 |
|------|------|------|----------|
| **RapidOCR** | 快 | 中 | 大批量处理 |
| **PaddleOCR** | 中 | 高 | 中英文混合 |
| **Qwen2-VL** | 慢 | 最高 | 复杂布局/表格 |

### 查询模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `naive` | 简单向量检索 | 快速查询 |
| `local` | 局部知识图谱 | 细粒度问题 |
| `global` | 全局知识图谱 | 宏观问题 |
| `hybrid` | 混合模式（推荐） | 综合问题 |

## 配置说明

在 [config.py](config.py) 中可以修改：

- **路径配置**: 输入目录、工作目录、日志目录
- **LLM 配置**: 模型名称、API 地址、温度参数
- **Embedding 配置**: 模型名称、设备、向量维度
- **RAG 配置**: 查询模式、批处理大小、OCR 设置
- **GPU 配置**: 张并行大小、显存利用率、并发数

## 性能基准

### 数据集性能示例

| 格式 | 数量 | 提取器 | 索引时间 |
|------|------|--------|----------|
| PDF | 1,700+ | PyMuPDF + OCR | ~30min |
| TXT | 数十个 | 原生 | <1min |
| DOCX | 数十个 | python-docx | ~5min |
| **总计** | **~1,800** | | **~35min** |

### 查询性能

| 指标 | 数值 |
|------|------|
| 查询延迟 | 2-5s |
| 并发支持 | 100+ QPS |
| GPU 利用率 | >80% |

## 文档

- [使用说明](docs/USAGE.md) - 详细使用指南
- [部署指南](docs/DEPLOYMENT.md) - 部署步骤
- [最佳实践](docs/BEST_PRACTICES.md) - 架构设计和优化建议
- [架构分析](docs/ARCHITECTURE_ANALYSIS.md) - 系统架构详解
- [Qwen 使用流程](docs/QWEN_USAGE_FLOW.md) - 模型调用流程
- [LightRAG 优化](docs/LIGHTRAG_OPTIMIZATION.md) - 性能优化指南

## 后续优化方向

- [x] OCR 处理扫描文档
- [x] 并发索引支持
- [ ] 启用量化 (AWQ/GPTQ) 节省显存
- [ ] REST API 服务
- [ ] Web UI 界面
- [ ] 增量索引更新
- [ ] 缓存层优化

## 常见问题

### Q: 如何选择 vLLM 还是 Ollama？

A:
- **vLLM**: 适合生产环境、高并发场景、多 GPU 部署
- **Ollama**: 适合个人开发、本地测试、快速原型

根据实际需求选择合适的服务框架。

### Q: GPU 资源不足怎么办？

A: 可以调整配置：
- 减少张并行数
- 启用量化
- 使用更小的模型

### Q: 如何处理扫描的 PDF？

A: 已内置 OCR 支持，参考 [docs/USAGE.md](docs/USAGE.md) 中的 OCR 配置说明。

### Q: 如何切换 OCR 引擎？

A: 在 [config.py](config.py) 中修改 `OCR_ENGINE` 参数：
- `rapidocr` - 快速处理（默认）
- `paddleocr` - 高精度
- `qwen2_vl` - 最强识别能力

## License

MIT
