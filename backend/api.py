"""
LightRAG Backend API 接口
提供简单的 API 接口用于集成
"""

import asyncio
import logging
from typing import Optional

from .service import RAGService, create_service_from_config
from .models import QueryRequest, QueryResponse, ServiceStatus, QueryMode
from .exceptions import RAGServiceError


logger = logging.getLogger(__name__)


# 全局服务实例
_service: Optional[RAGService] = None
_lock = asyncio.Lock()


async def initialize(config_module=None) -> RAGService:
    """
    初始化并启动 RAG 服务

    Args:
        config_module: 配置模块，默认使用 config.get_config()

    Returns:
        RAG 服务实例
    """
    global _service

    async with _lock:
        if _service is not None:
            logger.info("服务已经初始化")
            return _service

        if config_module is None:
            from config import get_config
            config_module = get_config()

        logger.info("正在初始化 RAG Backend 服务...")
        _service = create_service_from_config(config_module)
        await _service.start()

        logger.info("RAG Backend 服务初始化完成")
        return _service


async def shutdown() -> None:
    """关闭 RAG 服务"""
    global _service

    async with _lock:
        if _service is None:
            return

        logger.info("正在关闭 RAG Backend 服务...")
        await _service.stop()
        _service = None
        logger.info("RAG Backend 服务已关闭")


async def query(
    query_text: str,
    mode: str = "hybrid",
    only_context: bool = False,
    timeout: Optional[int] = None,
) -> QueryResponse:
    """
    执行查询

    Args:
        query_text: 查询文本
        mode: 查询模式 (naive, local, global, hybrid)
        only_context: 是否只返回上下文
        timeout: 超时时间（秒）

    Returns:
        查询响应

    Raises:
        RAGServiceError: 服务未初始化或查询失败
    """
    global _service

    if _service is None:
        raise RAGServiceError("服务未初始化，请先调用 initialize()")

    request = QueryRequest(
        query=query_text,
        mode=QueryMode(mode),
        only_context=only_context,
        timeout=timeout,
    )

    return await _service.query(request)


async def query_simple(query_text: str, mode: str = "hybrid") -> str:
    """
    简化的查询接口，直接返回结果字符串

    Args:
        query_text: 查询文本
        mode: 查询模式 (naive, local, global, hybrid)

    Returns:
        查询结果字符串
    """
    response = await query(query_text, mode=mode)
    return response.answer


def get_status() -> ServiceStatus:
    """
    获取服务状态

    Returns:
        服务状态

    Raises:
        RAGServiceError: 服务未初始化
    """
    global _service

    if _service is None:
        raise RAGServiceError("服务未初始化")

    return _service.get_status()


def is_ready() -> bool:
    """
    检查服务是否就绪

    Returns:
        服务是否就绪
    """
    global _service

    if _service is None:
        return False

    return _service.is_ready()


async def health_check() -> bool:
    """
    健康检查

    Returns:
        服务是否健康
    """
    global _service

    if _service is None:
        return False

    return await _service.health_check()


# 上下文管理器
class RAGBackend:
    """
    RAG Backend 上下文管理器

    使用方式:
        async with RAGBackend() as backend:
            result = await backend.query("问题")
    """

    def __init__(self, config_module=None):
        self._config = config_module
        self._service: Optional[RAGService] = None

    async def __aenter__(self):
        self._service = await initialize(self._config)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await shutdown()

    async def query(self, query_text: str, mode: str = "hybrid") -> str:
        """执行查询"""
        return await query_simple(query_text, mode)

    def status(self) -> ServiceStatus:
        """获取状态"""
        return get_status()
