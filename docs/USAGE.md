# LightRAG 使用说明

## 服务架构说明

### 两个模型的作用

你的理解很对！RAG 系统确实需要两个不同的模型：

```
┌─────────────────────────────────────────────────────────────┐
│                    LightRAG 工作流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  用户问题: "文档查询示例问题？"                              │
│     │                                                        │
│     ▼                                                        │
│  ┌─────────────────┐                                        │
│  │  1. 向量化查询   │  → bge-m3 (Embedding 模型)              │
│  │     问题 → 向量   │     本地运行，GPU 0                    │
│  └─────────────────┘                                        │
│     │                                                        │
│     ▼                                                        │
│  ┌─────────────────┐                                        │
│  │  2. 向量检索     │  → 在向量数据库中搜索相似文档           │
│  │     找相关文档   │     (已索引的业务文档)                   │
│  └─────────────────┘                                        │
│     │                                                        │
│     ▼                                                        │
│  ┌─────────────────┐                                        │
│  │  3. 生成回答     │  → Qwen2.5-72B (LLM)                   │
│  │  基于文档回答    │     vLLM 服务，8x4090 并行              │
│  └─────────────────┘                                        │
│     │                                                        │
│     ▼                                                        │
│  最终回答: "根据相关文档，查询结果如下..."                    │
└─────────────────────────────────────────────────────────────┘
```

### 模型对比

| 模型 | 类型 | 作用 | 运行位置 | 端口 |
|------|------|------|----------|------|
| **bge-m3** | Embedding 模型 | 将文本转换为向量 | 本地 (GPU 0) | 无 |
| **Qwen2.5-72B** | LLM | 理解问题 + 生成回答 | vLLM (8x4090) | PORT |

### 为什么需要两个模型？

1. **Embedding 模型 (bge-m3)**
   - 将文本转换为 1024 维向量
   - 用于向量检索（找相似文档）
   - 轻量级，可以和 vLLM 共享 GPU 0

2. **LLM (Qwen2.5-72B)**
   - 理解用户问题
   - 理解检索到的文档
   - 生成最终回答
   - 大模型，需要 8x4090 并行

## 索引完成后如何使用

### 1. 单次查询

```bash
python main.py query --query "你的问题" --mode hybrid
```

**示例**：
```bash
# 查询示例问题
python main.py query --query "文档查询示例问题？" --mode hybrid

# 查询商标注册
python main.py query --query "如何注册商标？" --mode hybrid

# 查询知识产权保护
python main.py query --query "知识产权保护有哪些类型？" --mode hybrid
```

### 2. 交互模式（推荐）

```bash
python main.py interactive
```

进入交互模式后，可以连续提问：
```
请输入你的问题: 文档查询示例问题？
[查询中...]
回答: 根据相关文档...

请输入你的问题: 费用大概是多少？
[查询中...]
回答: 相关费用包括...

请输入你的问题: exit
```

### 3. 查询模式选择

LightRAG 支持四种查询模式：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `naive` | 简单向量检索 | 快速查询，简单问题 |
| `local` | 局部知识图谱 | 细粒度问题，如"具体流程" |
| `global` | 全局知识图谱 | 宏观问题，如"总体趋势" |
| `highbrid` | 混合模式（推荐） | 综合问题 |

```bash
# 不同模式示例
python main.py query --query "文档类型有哪些？" --mode local
python main.py query --query "知识产权保护趋势？" --mode global
python main.py query --query "如何申请商标？" --mode hybrid
```

## 索引数据存储位置

索引完成后，向量数据存储在：

```
data/rag_storage/
├── kv_store_*.json         # 键值存储
├── vector_store_*.json     # 向量存储
├── graph_chunk_entity_relation.json  # 知识图谱
└── llm_response_cache_*.json         # LLM 响应缓存
```

这些数据会自动被 LightRAG 读取，无需手动操作。

## API 调用示例

如果你想在自己的代码中调用，可以这样做：

```python
import asyncio
from config import get_config
from src.models import init_llm_service, init_embedding_service, create_embedding_function
from src.rag_engine import init_rag_engine

async def search_documents(query: str):
    # 初始化配置
    cfg = get_config()

    # 初始化服务
    llm = init_llm_service(
        model_name=cfg.llm.MODEL_NAME,
        base_url=cfg.llm.BASE_URL,  # vLLM 服务地址
        api_key=cfg.llm.API_KEY
    )

    embedding = init_embedding_service(
        model_name=cfg.embedding.MODEL_NAME,
        device=cfg.embedding.get_embedding_device()
    )

    # 创建 RAG 引擎
    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return await llm.acomplete(prompt, system_prompt, history_messages, **kwargs)

    embedding_func = create_embedding_function(embedding)

    engine = init_rag_engine(
        working_dir=cfg.paths.WORKING_DIR,
        llm_model_func=llm_func,
        embedding_func=embedding_func
    )

    await engine.initialize()

    # 查询
    result = await engine.query(query, mode="hybrid")
    return result

# 使用
answer = asyncio.run(search_documents("文档查询示例"))
print(answer)
```

## 工作流程总结

1. **索引阶段**（一次性）
   - 读取业务文档
   - bge-m3 将每个文档转换为向量
   - 存储到向量数据库

2. **查询阶段**（重复使用）
   - 用户输入问题
   - bge-m3 将问题转换为向量
   - 在向量数据库中搜索相似文档
   - Qwen2.5-72B 基于文档生成回答
   - 返回答案给用户

## 端口说明

| 服务 | 端口 | 说明 |
|------|------|------|
| vLLM | PORT | LLM 推理服务 |

## 常见问题

### Q: 为什么向量模型不单独部署一个服务？

A: bge-m3 是轻量级模型（约 2GB 显存），可以直接在 Python 中加载，无需单独部署。它与 vLLM 共享 GPU 0，资源利用率更高。

### Q: 可以更换其他模型吗？

A: 可以，在 [config.py](config.py) 中修改：
- `LLM_CONFIG.MODEL_NAME` - 更换 LLM
- `EMBEDDING_CONFIG.MODEL_NAME` - 更换 Embedding 模型

### Q: 索引数据可以备份吗？

A: 可以，整个 `data/rag_storage/` 目录就是索引数据，直接复制备份即可。
