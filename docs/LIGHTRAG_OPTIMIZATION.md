# LightRAG 优化指南

## 当前问题分析

从日志中观察到 LightRAG 处理文档时的问题：

```
WARNING: chunk-xxx: LLM output format error; found 4/5 fields on REALTION `说明`~`内容提示, additional information`
WARNING: chunk-xxx: Complete delimiter can not be found in extraction result
```

这些问题表明 LLM 在提取实体和关系时输出格式不符合预期。

## 优化方案

### 1. 调整 Chunk 大小

**问题**: 当前 chunk_size=1024 可能导致某些 chunk 内容过长，LLM 处理时格式混乱。

**优化配置** ([config.py:102](LightRAG_Project/config.py#L102)):

```python
# 原配置
CHUNK_SIZE: int = 1024

# 建议配置（根据文档类型调整）
CHUNK_SIZE: int = 512  # 较小的 chunk 更稳定
# 或针对中文文档
CHUNK_SIZE: int = 800  # 中等大小，平衡性能和质量
```

### 2. LLM 温度参数调整

**问题**: 温度过高可能导致输出格式不稳定。

**优化配置** ([config.py:61](LightRAG_Project/config.py#L61)):

```python
# 原配置
TEMPERATURE: float = 0.7

# 建议配置（更稳定的输出）
TEMPERATURE: float = 0.1  # 低温，输出更确定
```

### 3. 添加自定义 LightRAG 配置

在 `src/rag_engine.py` 中添加更严格的提取参数：

```python
# 在 LightRAG 初始化时添加
self._rag = LightRAG(
    working_dir=str(self.working_dir),
    llm_model_func=self.llm_model_func,
    embedding_func=self.embedding_func,
    chunk_token_size=self.chunk_size,

    # 添加以下参数优化提取
    tiktoken_model_name="Qwen/Qwen2.5-3B",  # 指定 tokenizer
    extract_entity_types=["ORGANIZATION", "PERSON", "LOCATION", "PRODUCT", "EVENT"],  # 明确实体类型

    # 语言设置
    language="chinese",  # 明确指定中文

    # 提取策略
    chunk_overlap_token_size=50,  # 增加 chunk 重叠
)
```

### 4. 针对中文文档的优化

中文文档提取建议使用专门的中文实体类型：

```python
# 推荐的中文实体类型
CHINESE_ENTITY_TYPES = [
    "组织机构",    # ORGANIZATION
    "人物",        # PERSON
    "地理位置",    # LOCATION
    "产品",        # PRODUCT
    "事件",        # EVENT
    "时间",        # TIME
    "金额",        # MONEY
    "合同协议",    # CONTRACT
    "证书资质",    # CERTIFICATE
    "项目",        # PROJECT
]
```

### 5. 添加重试机制

在 LLM 服务中添加重试逻辑，当格式解析失败时自动重试：

```python
# 在 src/models.py 的 LLMService 中添加
MAX_RETRIES = 3

async def acomplete_with_retry(self, prompt, system_prompt=None, history_messages=[], max_retries=MAX_RETRIES):
    """带重试的完成方法"""
    for attempt in range(max_retries):
        result = await self.acomplete(prompt, system_prompt, history_messages)

        # 检查结果是否包含完整分隔符
        if self._validate_output_format(result):
            return result

        logger.warning(f"LLM 输出格式不完整，重试 {attempt + 1}/{max_retries}")

    return result  # 返回最后一次结果
```

### 6. 使用更强大的模型

如果资源允许，考虑使用更大的模型：

```python
# config.py
MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"  # 或 14B
# 或使用专门的代码/结构化输出模型
MODEL_NAME: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
```

### 7. 启用 LightRAG 的原始文本处理

对于复杂文档，可以跳过实体提取，直接使用原始文本：

```python
# 使用 naive 模式直接检索，不做知识图谱
await self.rag.aquery(query_text, param=QueryParam(mode="naive"))
```

## 推荐配置组合

### 场景 1: 稳定性优先（推荐用于生产环境）

```python
# config.py
CHUNK_SIZE: int = 512
TEMPERATURE: float = 0.1
MAX_TOKENS: int = 2048  # 减少 max_tokens 提高稳定性
```

### 场景 2: 质量优先（用于精细分析）

```python
CHUNK_SIZE: int = 800
TEMPERATURE: float = 0.0
MAX_TOKENS: int = 4096
# 使用 7B 或 14B 模型
```

### 场景 3: 速度优先（用于快速预览）

```python
CHUNK_SIZE: int = 1200  # 更大的 chunk
TEMPERATURE: float = 0.3
MAX_TOKENS: int = 1024
# 使用 naive 查询模式
```

## 监控和诊断

### 查看提取详情

启用 LightRAG 的详细日志：

```python
import logging
logging.getLogger("lightrag").setLevel(logging.DEBUG)
```

### 统计分析

定期检查提取成功率：

```python
# 统计日志中的 WARNING 数量
grep "LLM output format error" logs/lightrag.log | wc -l
grep "Complete delimiter" logs/lightrag.log | wc -l
```

## 迁移指南

如果已使用旧配置运行，需要重新索引：

```bash
# 清除旧索引
rm -rf data/rag_storage/*

# 重新运行索引
cd LightRAG_Project
python main.py index
```

## 参考资源

- LightRAG 官方文档: https://github.com/HKUDS/LightRAG
- Qwen2.5 模型文档: https://huggingface.co/Qwen
- 实体提取最佳实践: 参考 `docs/BEST_PRACTICES.md`
