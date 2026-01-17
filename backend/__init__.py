"""
LightRAG Backend 模块
提供解耦的 RAG 查询服务接口
"""

from .service import RAGService
from .models import QueryRequest, QueryResponse, ServiceStatus
from .exceptions import RAGServiceError, RAGNotReadyError

__all__ = [
    "RAGService",
    "QueryRequest",
    "QueryResponse",
    "ServiceStatus",
    "RAGServiceError",
    "RAGNotReadyError",
]

__version__ = "0.1.0"
