"""
LightRAG Backend HTTP Server
提供 REST API 接口
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .service import RAGService, create_service_from_config
from .models import QueryMode, ServiceStatus, ServiceState
from .exceptions import RAGServiceError, RAGNotReadyError, RAGQueryError


logger = logging.getLogger(__name__)

# 全局服务实例
_service: Optional[RAGService] = None


# Pydantic 模型
class QueryRequestPydantic(BaseModel):
    """查询请求模型"""
    query: str = Field(..., description="查询文本", min_length=1)
    mode: str = Field(default="hybrid", description="查询模式: naive, local, global, hybrid")
    only_context: bool = Field(default=False, description="是否只返回上下文")
    timeout: Optional[int] = Field(default=None, description="超时时间（秒）")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "专利申请流程是什么？",
                "mode": "hybrid",
                "only_context": False,
                "timeout": 120
            }
        }


class QueryResponsePydantic(BaseModel):
    """查询响应模型"""
    answer: str
    mode: str
    query: str
    timestamp: str
    latency_ms: Optional[float] = None
    context: Optional[str] = None


class ServiceStatusResponse(BaseModel):
    """服务状态响应"""
    state: str
    initialized: bool
    has_data: bool
    working_dir: str
    is_ready: bool
    error_message: Optional[str] = None
    chunk_size: int = 1024


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    detail: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _service

    logger.info("正在启动 LightRAG Backend 服务...")

    try:
        from config import get_config
        cfg = get_config()

        _service = create_service_from_config(cfg)
        await _service.start()

        logger.info("LightRAG Backend 服务启动完成")

        yield

    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise
    finally:
        logger.info("正在关闭 LightRAG Backend 服务...")
        if _service:
            await _service.stop()
        logger.info("LightRAG Backend 服务已关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title="LightRAG Backend API",
    description="LightRAG 智能文档检索系统 REST API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Root"])
async def root():
    """根路径"""
    return {
        "service": "LightRAG Backend",
        "version": "0.1.0",
        "status": "running" if _service and _service.is_ready() else "initializing",
    }


@app.get("/health", tags=["Health"])
async def health():
    """健康检查"""
    if _service is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    is_healthy = await _service.health_check()

    if not is_healthy:
        raise HTTPException(status_code=503, detail="服务未就绪")

    return {"status": "healthy", "ready": True}


@app.get("/status", response_model=ServiceStatusResponse, tags=["Status"])
async def status():
    """
    获取服务状态

    返回服务的详细状态信息，包括初始化状态、数据状态等。
    """
    if _service is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    service_status = _service.get_status()

    return ServiceStatusResponse(
        state=service_status.state.value,
        initialized=service_status.initialized,
        has_data=service_status.has_data,
        working_dir=service_status.working_dir,
        is_ready=service_status.is_ready,
        error_message=service_status.error_message,
        chunk_size=service_status.chunk_size,
    )


@app.post("/query", response_model=QueryResponsePydantic, tags=["Query"])
async def query(request: QueryRequestPydantic):
    """
    执行查询

    根据查询文本和模式执行 RAG 查询。

    查询模式：
    - `naive`: 简单检索，直接从文档中检索相关内容
    - `local`: 局部知识图谱检索，关注实体间的关系
    - `global`: 全局知识图谱检索，关注整体结构
    - `hybrid`: 混合模式（推荐），结合多种检索方式
    """
    if _service is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    try:
        from .models import QueryRequest as BackendQueryRequest

        backend_request = BackendQueryRequest(
            query=request.query,
            mode=QueryMode(request.mode),
            only_context=request.only_context,
            timeout=request.timeout,
        )

        response = await _service.query(backend_request)

        return QueryResponsePydantic(
            answer=response.answer,
            mode=response.mode.value,
            query=response.query,
            timestamp=response.timestamp.isoformat(),
            latency_ms=response.latency_ms,
            context=response.context,
        )

    except RAGNotReadyError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except RAGQueryError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={"error": "内部服务器错误", "detail": str(exc)},
    )


def run_server(host: str = "0.0.0.0", port: int = 8000, log_level: str = "info"):
    """
    运行 HTTP 服务器

    Args:
        host: 监听地址
        port: 监听端口
        log_level: 日志级别
    """
    uvicorn.run(
        "backend.server:app",
        host=host,
        port=port,
        log_level=log_level,
        access_log=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LightRAG Backend HTTP Server")
    parser.add_argument("--host", "-H", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", "-p", type=int, default=8000, help="监听端口")
    parser.add_argument("--log-level", "-l", default="info", choices=["critical", "error", "warning", "info", "debug"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    run_server(host=args.host, port=args.port, log_level=args.log_level)
