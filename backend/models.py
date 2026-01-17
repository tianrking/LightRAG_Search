"""
Backend 数据模型定义
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime


class QueryMode(str, Enum):
    """查询模式枚举"""

    NAIVE = "naive"
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"

    @classmethod
    def default(cls) -> "QueryMode":
        return cls.HYBRID


class ServiceState(str, Enum):
    """服务状态枚举"""

    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    ERROR = "error"


@dataclass
class QueryRequest:
    """查询请求模型"""

    query: str
    mode: QueryMode = field(default_factory=QueryMode.default)
    only_context: bool = False
    timeout: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = QueryMode(self.mode)


@dataclass
class QueryResponse:
    """查询响应模型"""

    answer: str
    mode: QueryMode
    query: str
    timestamp: datetime = field(default_factory=datetime.now)
    latency_ms: Optional[float] = None
    context: Optional[str] = None

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "answer": self.answer,
            "mode": self.mode.value,
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
            "context": self.context,
        }


@dataclass
class ServiceStatus:
    """服务状态模型"""

    state: ServiceState
    initialized: bool
    has_data: bool
    working_dir: str
    error_message: Optional[str] = None
    chunk_size: int = 1024

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "state": self.state.value,
            "initialized": self.initialized,
            "has_data": self.has_data,
            "working_dir": self.working_dir,
            "error_message": self.error_message,
            "chunk_size": self.chunk_size,
        }

    @property
    def is_ready(self) -> bool:
        """是否就绪"""
        return self.state == ServiceState.READY and self.initialized and self.has_data


@dataclass
class ServiceConfig:
    """服务配置模型"""

    # LLM 配置
    llm_model_name: str
    llm_base_url: str
    llm_api_key: str = "EMPTY"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.7

    # Embedding 配置
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_device: str = "cuda:0"
    embedding_dim: int = 1024
    embedding_max_tokens: int = 8192

    # RAG 配置
    working_dir: str = "./data/rag_storage"
    chunk_size: int = 1024
    default_mode: QueryMode = field(default_factory=QueryMode.default)

    # 超时配置
    query_timeout: int = 120
    init_timeout: int = 300
