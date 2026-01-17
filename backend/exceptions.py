"""
Backend 异常定义
"""


class RAGServiceError(Exception):
    """RAG 服务基础异常"""

    pass


class RAGNotReadyError(RAGServiceError):
    """RAG 服务未就绪异常"""

    pass


class RAGConfigError(RAGServiceError):
    """RAG 配置错误异常"""

    pass


class RAGQueryError(RAGServiceError):
    """RAG 查询错误异常"""

    pass


class RAGInitError(RAGServiceError):
    """RAG 初始化错误异常"""

    pass
