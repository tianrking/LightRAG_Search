"""
LightRAG Backend 服务实现
提供解耦的 RAG 查询服务
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from .models import (
    QueryRequest,
    QueryResponse,
    ServiceStatus,
    ServiceState,
    ServiceConfig,
    QueryMode,
)
from .exceptions import RAGNotReadyError, RAGInitError, RAGQueryError


logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG 查询服务

    提供解耦的 LightRAG 查询接口，支持异步操作和状态管理。
    """

    def __init__(self, config: ServiceConfig):
        """
        初始化 RAG 服务

        Args:
            config: 服务配置
        """
        self._config = config
        self._state = ServiceState.STOPPED
        self._rag_engine = None
        self._llm_service = None
        self._embedding_service = None
        self._lock = asyncio.Lock()
        self._error_message: Optional[str] = None

        logger.info(f"RAG 服务已创建 (工作目录: {config.working_dir})")

    @property
    def config(self) -> ServiceConfig:
        """获取服务配置"""
        return self._config

    @property
    def state(self) -> ServiceState:
        """获取服务状态"""
        return self._state

    async def start(self) -> None:
        """
        启动 RAG 服务

        初始化 LLM、Embedding 和 RAG 引擎
        """
        async with self._lock:
            if self._state == ServiceState.READY:
                logger.info("服务已经就绪，无需重复启动")
                return

            logger.info("正在启动 RAG 服务...")
            self._state = ServiceState.STARTING
            self._error_message = None

            try:
                # 动态导入，避免循环依赖
                from src.models import init_llm_service, init_embedding_service, create_embedding_function
                from src.rag_engine import RAGEngine

                # 初始化 LLM 服务
                logger.info("初始化 LLM 服务...")
                self._llm_service = init_llm_service(
                    model_name=self._config.llm_model_name,
                    base_url=self._config.llm_base_url,
                    api_key=self._config.llm_api_key,
                    max_tokens=self._config.llm_max_tokens,
                    temperature=self._config.llm_temperature,
                )

                # 测试 LLM 连接
                if not self._llm_service.test_connection():
                    raise RAGInitError("LLM 服务连接失败，请检查服务是否正常运行")

                # 初始化 Embedding 服务
                logger.info("初始化 Embedding 服务...")
                self._embedding_service = init_embedding_service(
                    model_name=self._config.embedding_model_name,
                    device=self._config.embedding_device,
                    embedding_dim=self._config.embedding_dim,
                    max_token_size=self._config.embedding_max_tokens,
                )

                # 创建 LLM 和 Embedding 函数
                async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                    return await self._llm_service.acomplete(
                        prompt, system_prompt, history_messages, **kwargs
                    )

                embedding_func = create_embedding_function(self._embedding_service)

                # 初始化 RAG 引擎
                logger.info("初始化 RAG 引擎...")
                self._rag_engine = RAGEngine(
                    working_dir=Path(self._config.working_dir),
                    llm_model_func=llm_func,
                    embedding_func=embedding_func,
                    chunk_size=self._config.chunk_size,
                )

                await asyncio.wait_for(
                    self._rag_engine.initialize(),
                    timeout=self._config.init_timeout
                )

                self._state = ServiceState.READY
                logger.info("RAG 服务启动完成")

            except asyncio.TimeoutError:
                self._state = ServiceState.ERROR
                self._error_message = f"初始化超时 (超过 {self._config.init_timeout} 秒)"
                logger.error(self._error_message)
                raise RAGInitError(self._error_message)

            except Exception as e:
                self._state = ServiceState.ERROR
                self._error_message = str(e)
                logger.error(f"RAG 服务启动失败: {e}")
                raise RAGInitError(f"服务启动失败: {e}") from e

    async def stop(self) -> None:
        """停止 RAG 服务"""
        async with self._lock:
            logger.info("正在停止 RAG 服务...")
            self._state = ServiceState.STOPPED
            self._rag_engine = None
            self._llm_service = None
            self._embedding_service = None
            self._error_message = None
            logger.info("RAG 服务已停止")

    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        执行查询

        Args:
            request: 查询请求

        Returns:
            查询响应

        Raises:
            RAGNotReadyError: 服务未就绪
            RAGQueryError: 查询失败
        """
        if self._state != ServiceState.READY:
            raise RAGNotReadyError(
                f"服务未就绪 (当前状态: {self._state.value})，"
                f"请先调用 start() 方法启动服务"
            )

        if not self._rag_engine:
            raise RAGNotReadyError("RAG 引擎未初始化")

        import time

        logger.info(f"执行查询 (模式: {request.mode.value}): {request.query[:50]}...")

        start_time = time.time()

        try:
            # 执行查询
            result = await asyncio.wait_for(
                self._rag_engine.query(
                    query_text=request.query,
                    mode=request.mode.value,
                    only_need_context=request.only_context,
                ),
                timeout=request.timeout or self._config.query_timeout,
            )

            latency = (time.time() - start_time) * 1000

            response = QueryResponse(
                answer=result,
                mode=request.mode,
                query=request.query,
                latency_ms=latency,
            )

            logger.info(f"查询完成 (耗时: {latency:.2f}ms)")
            return response

        except asyncio.TimeoutError:
            raise RAGQueryError(
                f"查询超时 (超过 {request.timeout or self._config.query_timeout} 秒)"
            )

        except Exception as e:
            logger.error(f"查询失败: {e}")
            raise RAGQueryError(f"查询失败: {e}") from e

    async def query_simple(
        self,
        query: str,
        mode: QueryMode | str = QueryMode.HYBRID,
    ) -> str:
        """
        简化的查询接口

        Args:
            query: 查询文本
            mode: 查询模式

        Returns:
            查询结果字符串
        """
        if isinstance(mode, str):
            mode = QueryMode(mode)

        request = QueryRequest(query=query, mode=mode)
        response = await self.query(request)
        return response.answer

    def get_status(self) -> ServiceStatus:
        """
        获取服务状态

        Returns:
            服务状态
        """
        has_data = False
        initialized = False

        if self._rag_engine:
            rag_status = self._rag_engine.get_status()
            has_data = rag_status.get("has_data", False)
            initialized = rag_status.get("initialized", False)

        return ServiceStatus(
            state=self._state,
            initialized=initialized,
            has_data=has_data,
            working_dir=self._config.working_dir,
            error_message=self._error_message,
            chunk_size=self._config.chunk_size,
        )

    def is_ready(self) -> bool:
        """检查服务是否就绪"""
        status = self.get_status()
        return status.is_ready

    async def health_check(self) -> bool:
        """
        健康检查

        Returns:
            服务是否健康
        """
        return self.is_ready()


# 服务工厂函数
def create_service_from_config(config_module) -> RAGService:
    """
    从配置模块创建 RAG 服务

    Args:
        config_module: 配置模块 (如 config.py 中的 get_config())

    Returns:
        RAG 服务实例
    """
    cfg = config_module

    service_config = ServiceConfig(
        llm_model_name=cfg.llm.MODEL_NAME,
        llm_base_url=cfg.llm.BASE_URL,
        llm_api_key=cfg.llm.API_KEY,
        llm_max_tokens=cfg.llm.MAX_TOKENS,
        llm_temperature=cfg.llm.TEMPERATURE,
        embedding_model_name=cfg.embedding.MODEL_NAME,
        embedding_device=cfg.embedding.DEVICE,
        embedding_dim=cfg.embedding.EMBEDDING_DIM,
        embedding_max_tokens=cfg.embedding.MAX_TOKEN_SIZE,
        working_dir=str(cfg.paths.WORKING_DIR),
        chunk_size=cfg.rag.CHUNK_SIZE,
        default_mode=QueryMode(cfg.rag.QUERY_MODE),
    )

    return RAGService(service_config)
