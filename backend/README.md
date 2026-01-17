# LightRAG Backend 模块

现代化的解耦 RAG 查询服务模块。

## 概述

Backend 模块提供了一个完全解耦的 RAG（检索增强生成）查询服务，支持多种使用方式：

- **直接 API 调用**: 简单的函数接口
- **服务对象**: 面向对象的使用方式
- **上下文管理器**: 自动资源管理
- **HTTP Server**: REST API 接口

## 目录结构

```
backend/
├── __init__.py       # 模块导出
├── exceptions.py     # 异常定义
├── models.py         # 数据模型
├── service.py        # 核心服务实现
├── api.py            # 高级 API 接口
├── server.py         # HTTP Server (FastAPI)
├── client.py         # 使用示例
└── README.md         # 本文档
```

## 快速开始

### 方式 1: 直接 API 调用

```python
from backend.api import initialize, query_simple, shutdown

async def main():
    # 初始化服务
    await initialize()

    # 执行查询
    result = await query_simple("专利申请流程是什么？", mode="hybrid")
    print(result)

    # 关闭服务
    await shutdown()
```

### 方式 2: 上下文管理器（推荐）

```python
from backend.api import RAGBackend

async def main():
    async with RAGBackend() as backend:
        result = await backend.query("如何申请专利？")
        print(result)
```

### 方式 3: 服务对象

```python
from backend import RAGService, ServiceConfig, QueryRequest, QueryMode

async def main():
    # 创建配置
    config = ServiceConfig(
        llm_model_name="path/to/model",
        llm_base_url="http://localhost:8000/v1",
        working_dir="./data/rag_storage",
    )

    # 创建并启动服务
    service = RAGService(config)
    await service.start()

    # 执行查询
    request = QueryRequest(query="问题", mode=QueryMode.HYBRID)
    response = await service.query(request)
    print(response.answer)

    # 停止服务
    await service.stop()
```

## HTTP Server

### 启动服务器

```bash
python -m backend.server --host 0.0.0.0 --port 8000
```

### API 端点

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/` | 服务信息 |
| GET | `/health` | 健康检查 |
| GET | `/status` | 服务状态 |
| POST | `/query` | 执行查询 |

### 查询示例

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "专利申请流程是什么？",
    "mode": "hybrid"
  }'
```

## 查询模式

| 模式 | 描述 |
|------|------|
| `naive` | 简单检索 |
| `local` | 局部知识图谱检索 |
| `global` | 全局知识图谱检索 |
| `hybrid` | 混合模式（推荐） |

## 配置

服务配置通过 `ServiceConfig` 模型进行：

```python
from backend import ServiceConfig

config = ServiceConfig(
    # LLM 配置
    llm_model_name="model_name",
    llm_base_url="http://localhost:8000/v1",
    llm_api_key="EMPTY",
    llm_max_tokens=4096,
    llm_temperature=0.7,

    # Embedding 配置
    embedding_model_name="BAAI/bge-m3",
    embedding_device="cuda:0",
    embedding_dim=1024,

    # RAG 配置
    working_dir="./data/rag_storage",
    chunk_size=1024,
    default_mode="hybrid",

    # 超时配置
    query_timeout=120,
    init_timeout=300,
)
```

## 数据文件位置

索引数据默认存储在：`./data/rag_storage/`

包含以下文件：
- `graph_chunk_entity_relation.graphml` - 知识图谱
- `kv_store_*.json` - 键值存储
- `vdb_*.json` - 向量数据库

## 异常处理

```python
from backend.exceptions import RAGServiceError, RAGNotReadyError, RAGQueryError

try:
    result = await query_simple("问题")
except RAGNotReadyError:
    print("服务未就绪")
except RAGQueryError as e:
    print(f"查询失败: {e}")
except RAGServiceError as e:
    print(f"服务错误: {e}")
```

## 完整示例

查看 [`client.py`](client.py) 获取更多使用示例。
