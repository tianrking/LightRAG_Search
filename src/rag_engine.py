"""
RAG 引擎模块
封装 LightRAG 的核心功能，提供文档索引和查询接口
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Union

from lightrag import LightRAG, QueryParam

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG 检索增强生成引擎"""

    def __init__(self,
                 working_dir: Union[str, Path],
                 llm_model_func,
                 embedding_func,
                 chunk_size: int = 1024):
        """
        初始化 RAG 引擎

        Args:
            working_dir: 工作目录（存储向量数据库）
            llm_model_func: LLM 模型函数
            embedding_func: Embedding 函数
            chunk_size: 文本块大小
        """
        self.working_dir = Path(working_dir)
        self.llm_model_func = llm_model_func
        self.embedding_func = embedding_func
        self.chunk_size = chunk_size

        self._rag: Optional[LightRAG] = None
        self._initialized = False

    @property
    def rag(self) -> LightRAG:
        """获取 LightRAG 实例"""
        if self._rag is None:
            raise RuntimeError("RAG 引擎未初始化，请先调用 initialize()")
        return self._rag

    async def initialize(self) -> None:
        """初始化 RAG 引擎"""
        if self._initialized:
            logger.info("RAG 引擎已经初始化")
            return

        try:
            logger.info(f"初始化 RAG 引擎，工作目录: {self.working_dir}")

            self._rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=self.llm_model_func,
                embedding_func=self.embedding_func,
                chunk_token_size=self.chunk_size,
            )

            await self._rag.initialize_storages()
            self._initialized = True

            logger.info("RAG 引擎初始化完成")

        except Exception as e:
            logger.error(f"RAG 引擎初始化失败: {e}")
            raise

    async def insert_documents(self,
                              documents: list[str],
                              batch_size: int = 16) -> None:
        """
        插入文档到索引

        Args:
            documents: 文档内容列表
            batch_size: 批处理大小
        """
        if not self._initialized:
            await self.initialize()

        if not documents:
            logger.warning("没有文档需要插入")
            return

        try:
            total = len(documents)
            logger.info(f"开始插入 {total} 个文档...")

            # 批量插入
            for i in range(0, total, batch_size):
                batch = documents[i:i + batch_size]
                await self.rag.ainsert(batch)
                logger.info(f"已插入 {min(i + batch_size, total)}/{total} 个文档")

            logger.info("文档插入完成")

        except Exception as e:
            logger.error(f"文档插入失败: {e}")
            raise

    async def insert_documents_from_dict(self,
                                        doc_dict: dict[str, str],
                                        batch_size: int = 16) -> None:
        """
        从字典插入文档

        Args:
            doc_dict: {文件路径: 文件内容} 字典
            batch_size: 批处理大小
        """
        documents = list(doc_dict.values())
        await self.insert_documents(documents, batch_size)

    async def query(self,
                   query_text: str,
                   mode: str = "hybrid",
                   **kwargs) -> str:
        """
        查询 RAG 系统

        Args:
            query_text: 查询文本
            mode: 查询模式
                - "naive": 简单检索
                - "local": 局部知识图谱检索
                - "global": 全局知识图谱检索
                - "hybrid": 混合模式（推荐）
            **kwargs: 额外参数

        Returns:
            查询结果
        """
        if not self._initialized:
            await self.initialize()

        valid_modes = ["naive", "local", "global", "hybrid"]
        if mode not in valid_modes:
            logger.warning(f"无效的查询模式: {mode}，使用默认模式: hybrid")
            mode = "hybrid"

        try:
            logger.info(f"执行查询 (模式: {mode}): {query_text[:50]}...")

            param = QueryParam(mode=mode, **kwargs)
            result = await self.rag.aquery(query_text, param=param)

            logger.info("查询完成")
            return result

        except Exception as e:
            logger.error(f"查询失败: {e}")
            raise

    async def query_with_context(self,
                                 query_text: str,
                                 mode: str = "hybrid",
                                 only_need_context: bool = False) -> str:
        """
        带上下文的查询

        Args:
            query_text: 查询文本
            mode: 查询模式
            only_need_context: 是否只返回上下文

        Returns:
            查询结果或上下文
        """
        return await self.query(
            query_text,
            mode=mode,
            only_need_context=only_need_context
        )

    def get_status(self) -> dict:
        """
        获取 RAG 引擎状态

        Returns:
            状态信息字典
        """
        return {
            "initialized": self._initialized,
            "working_dir": str(self.working_dir),
            "chunk_size": self.chunk_size,
            "has_data": self.working_dir.exists() and any(self.working_dir.iterdir()),
        }


# 全局 RAG 引擎实例
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """获取全局 RAG 引擎实例"""
    global _rag_engine
    if _rag_engine is None:
        raise RuntimeError("RAG 引擎未初始化，请先调用 init_rag_engine()")
    return _rag_engine


def init_rag_engine(working_dir: Union[str, Path],
                   llm_model_func,
                   embedding_func,
                   chunk_size: int = 1024) -> RAGEngine:
    """
    初始化全局 RAG 引擎

    Args:
        working_dir: 工作目录
        llm_model_func: LLM 模型函数
        embedding_func: Embedding 函数
        chunk_size: 文本块大小

    Returns:
        RAG 引擎实例
    """
    global _rag_engine
    _rag_engine = RAGEngine(working_dir, llm_model_func, embedding_func, chunk_size)
    return _rag_engine


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    async def test_rag():
        # 这里需要先初始化 LLM 和 Embedding 服务
        from .models import init_llm_service, init_embedding_service, create_embedding_function
        from ..config import get_config

        cfg = get_config()
        cfg.setup()

        # 初始化服务
        llm = init_llm_service(
            model_name=cfg.llm.MODEL_NAME,
            base_url=cfg.llm.BASE_URL,
            api_key=cfg.llm.API_KEY
        )

        embedding = init_embedding_service(
            model_name=cfg.embedding.MODEL_NAME,
            device=cfg.embedding.DEVICE,
            embedding_dim=cfg.embedding.EMBEDDING_DIM,
            max_token_size=cfg.embedding.MAX_TOKEN_SIZE
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

        # 测试状态
        status = engine.get_status()
        print(f"RAG 引擎状态: {status}")

    asyncio.run(test_rag())
