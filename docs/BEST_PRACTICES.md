# LightRAG 最佳实践

## 架构设计原则

### 1. 为什么 vLLM + PyMuPDF 是最优组合？

```
┌─────────────────────────────────────────────────────────┐
│                    LightRAG 应用层                       │
├─────────────────────────────────────────────────────────┤
│  文档提取  │  向量化  │  图谱构建  │  检索增强生成      │
└─────┬─────────────┬──────────────┬────────────┬────────┘
      │             │              │            │
      ▼             ▼              ▼            ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ PyMuPDF  │  │ bge-m3   │  │  Network │  │  vLLM    │
│ (CPU)    │  │(cuda:0)  │  │  Storage │  │(8x4090)  │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
     极快          中速          极快          极快并发
```

**关键优势**:
- **vLLM**: PagedAttention + 8卡并行 = 极高吞吐
- **PyMuPDF**: C++ 实现，比 pdfplumber 快 10 倍+
- **bge-m3**: 多语言支持，1024 维高质量向量
- **LightRAG**: 知识图谱 + 向量检索，质量更高

### 2. GPU 资源分配策略

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

## 代码模块化最佳实践

### 1. 配置管理

```python
# ❌ 不推荐: 硬编码
model = "Qwen/Qwen2.5-72B-Instruct"
base_url = "http://localhost:PORT/v1"

# ✓ 推荐: 集中配置
from config import get_config
cfg = get_config()
llm = init_llm_service(
    model_name=cfg.llm.MODEL_NAME,
    base_url=cfg.llm.BASE_URL
)
```

### 2. 错误处理

```python
# ✓ 优雅的错误处理
try:
    result = await rag.query(query_text)
except RAGError as e:
    logger.error(f"RAG 查询失败: {e}")
    return {"error": str(e), "fallback": True}
```

### 3. 资源管理

```python
# ✓ 使用上下文管理器
async with DocumentExtractor() as extractor:
    for file_path, result in extractor.extract_directory(dir):
        await process(result)
```

## 性能优化清单

### 文档提取

- [x] 使用 PyMuPDF (fitz) 而非 pdfplumber
- [x] 并发读取文件
- [ ] 考虑多进程处理大批量
- [ ] 添加文件指纹避免重复处理

### 向量化

- [x] 使用高质量模型 (bge-m3)
- [x] GPU 加速
- [ ] 批处理优化
- [ ] 向量缓存

### LLM 推理

- [x] vLLM 8卡并行
- [x] PagedAttention 优化
- [ ] 请求队列管理
- [ ] 响应流式输出

### 查询优化

- [x] 支持多种查询模式
- [ ] 查询缓存
- [ ] 结果重排序
- [ ] 查询扩展

## 常见问题

### Q: 为什么不用 Ollama?

**A**: Ollama 适合个人单用户，你的 8 卡配置用 Ollama 是浪费：
- Ollama 多卡支持有限
- 并发性能远低于 vLLM
- LightRAG 高并发特性无法发挥

### Q: PyMuPDF vs pdfplumber?

**A**: 性能对比 (1792个PDF):
```
PyMuPDF:      ~5-10 秒/100页
pdfplumber:   ~50-100 秒/100页
```

PyMuPDF 是 C++ 实现，速度快 10 倍，且功能完整。

### Q: 增量索引怎么做?

```python
# 方案1: 文件哈希去重
from hashlib import md5
file_hash = md5(file_content).hexdigest()

# 方案2: 元数据记录
last_modified = os.path.getmtime(file_path)
if last_modified > last_index_time:
    # 重新索引
```

### Q: 如何处理 OCR?

```python
# 添加 OCR 提取器
from rapidocr_onnxruntime import RapidOCR

class OCRExtractor(BaseExtractor):
    def extract(self, file_path: Path) -> ExtractResult:
        ocr = RapidOCR()
        result = ocr(str(file_path))
        return ExtractResult(content=result[1], ...)
```

## 监控指标

### 关键指标

| 指标 | 目标值 | 监控方法 |
|------|--------|----------|
| GPU 利用率 | >80% | `nvidia-smi` |
| 显存使用 | <90% | `nvidia-smi` |
| 查询延迟 | <5s | 应用日志 |
| 吞吐量 | >100 QPS | vLLM metrics |

### 告警阈值

```yaml
alerts:
  - GPU 利用率 < 50%  # 可能卡未充分利用
  - 显存使用 > 95%    # OOM 风险
  - 查询延迟 > 10s    # 性能下降
  - GPU 温度 > 85°C   # 过热风险
```

## 扩展路线图

### Phase 1: 基础设施 (当前)
- [x] vLLM 服务
- [x] 文档提取
- [x] RAG 引擎
- [ ] CLI 工具

### Phase 2: API 服务
- [ ] FastAPI 接口
- [ ] 认证授权
- [ ] 限流控制
- [ ] 监控面板

### Phase 3: 优化增强
- [ ] OCR 支持
- [ ] 多模态 (图片理解)
- [ ] 查询缓存
- [ ] 分布式部署

### Phase 4: 产品化
- [ ] Web UI
- [ ] 用户管理
- [ ] 权限控制
- [ ] 审计日志
