"""
模型管理模块
封装 LLM 和 Embedding 模型的初始化与调用

优化内容：
- LLM 调用：添加指数退避重试机制
- 更详细的错误处理和日志
- 连接健康检查
"""

import asyncio
import logging
from typing import Optional, List, Callable, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


# ========== 错误分类 ==========

class LLMErrorType(Enum):
    """LLM 错误类型"""
    CONNECTION_ERROR = "connection_error"       # 连接失败
    TIMEOUT_ERROR = "timeout_error"             # 请求超时
    RATE_LIMIT_ERROR = "rate_limit_error"       # 速率限制
    SERVER_ERROR = "server_error"               # 服务器错误 (5xx)
    INVALID_REQUEST = "invalid_request"         # 无效请求 (4xx)
    API_KEY_ERROR = "api_key_error"             # API 密钥错误
    CONTEXT_TOO_LONG = "context_too_long"       # 上下文过长
    MODEL_NOT_FOUND = "model_not_found"         # 模型不存在
    UNKNOWN_ERROR = "unknown_error"             # 未知错误


class EmbeddingErrorType(Enum):
    """Embedding 错误类型"""
    MODEL_LOAD_ERROR = "model_load_error"       # 模型加载失败
    CUDA_ERROR = "cuda_error"                   # CUDA 错误
    ENCODING_ERROR = "encoding_error"           # 编码错误
    UNKNOWN_ERROR = "unknown_error"             # 未知错误


@dataclass
class ModelError:
    """结构化的模型错误"""
    error_type: Enum
    message: str
    details: dict = None
    suggestion: str = None
    retry_able: bool = False  # 是否可重试

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def to_dict(self) -> dict:
        return {
            "type": self.error_type.value,
            "message": self.message,
            "details": self.details,
            "suggestion": self.suggestion,
            "retry_able": self.retry_able
        }


# ========== 重试配置 ==========

@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3           # 最大重试次数
    initial_delay: float = 1.0      # 初始延迟（秒）
    max_delay: float = 60.0         # 最大延迟（秒）
    exponential_base: float = 2.0   # 指数基数
    jitter: bool = True             # 是否添加随机抖动

    # 可重试的错误类型
    retryable_errors: List[LLMErrorType] = None

    def __post_init__(self):
        if self.retryable_errors is None:
            # 默认可重试的错误类型
            self.retryable_errors = [
                LLMErrorType.CONNECTION_ERROR,
                LLMErrorType.TIMEOUT_ERROR,
                LLMErrorType.RATE_LIMIT_ERROR,
                LLMErrorType.SERVER_ERROR,
            ]

    def is_retryable(self, error_type: LLMErrorType) -> bool:
        """判断错误是否可重试"""
        return error_type in self.retryable_errors

    def get_delay(self, attempt: int) -> float:
        """计算延迟时间（指数退避）"""
        delay = self.initial_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            import random
            delay *= (0.5 + random.random())

        return delay


# ========== LLM 服务（优化版）==========

class LLMService:
    """LLM 服务类（增强版：重试机制和详细错误处理）"""

    def __init__(self,
                 model_name: str,
                 base_url: str,
                 api_key: str = "EMPTY",
                 max_tokens: int = 4096,
                 temperature: float = 0.7,
                 retry_config: RetryConfig = None):
        """
        初始化 LLM 服务

        Args:
            model_name: 模型名称
            base_url: API 基础地址
            api_key: API 密钥
            max_tokens: 最大 token 数
            temperature: 温度参数
            retry_config: 重试配置
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retry_config = retry_config or RetryConfig()

        # 延迟导入，避免启动时加载失败
        self._openai_complete = None
        self._initialized = False

        # 统计信息
        self._call_count = 0
        self._retry_count = 0
        self._error_count = 0

    def _ensure_imports(self):
        """确保必要的依赖已导入"""
        if not self._initialized:
            try:
                from lightrag.llm.openai import openai_complete_if_cache
                self._openai_complete = openai_complete_if_cache
                self._initialized = True
                logger.info(f"LLM 服务初始化成功: {self.model_name}")
            except ImportError as e:
                logger.error(f"无法导入 lightrag: {e}")
                raise ModelError(
                    error_type=LLMErrorType.UNKNOWN_ERROR,
                    message=f"无法导入 lightrag: {str(e)}",
                    suggestion="请确保已安装 lightrag: pip install lightrag"
                )

    def _classify_error(self, exception: Exception) -> ModelError:
        """分类错误并返回结构化错误信息"""
        error_str = str(exception).lower()

        # 连接错误
        if any(keyword in error_str for keyword in ["connection", "connect", "network", "unreachable"]):
            return ModelError(
                error_type=LLMErrorType.CONNECTION_ERROR,
                message=f"无法连接到 LLM 服务: {str(exception)}",
                details={"base_url": self.base_url, "original_error": str(exception)},
                suggestion=f"请检查 vLLM 服务是否运行在 {self.base_url}",
                retry_able=True
            )

        # 超时错误
        if any(keyword in error_str for keyword in ["timeout", "timed out"]):
            return ModelError(
                error_type=LLMErrorType.TIMEOUT_ERROR,
                message=f"请求超时: {str(exception)}",
                details={"base_url": self.base_url, "original_error": str(exception)},
                suggestion="请求超时，可能是服务负载过高",
                retry_able=True
            )

        # 速率限制
        if any(keyword in error_str for keyword in ["rate limit", "429", "too many requests"]):
            return ModelError(
                error_type=LLMErrorType.RATE_LIMIT_ERROR,
                message=f"请求速率受限: {str(exception)}",
                details={"original_error": str(exception)},
                suggestion="请降低请求频率",
                retry_able=True
            )

        # API 密钥错误
        if any(keyword in error_str for keyword in ["api key", "unauthorized", "401", "authentication"]):
            return ModelError(
                error_type=LLMErrorType.API_KEY_ERROR,
                message=f"API 密钥错误: {str(exception)}",
                details={"original_error": str(exception)},
                suggestion="请检查 API 密钥配置",
                retry_able=False
            )

        # 上下文过长
        if any(keyword in error_str for keyword in ["context", "too long", "max token", "token limit"]):
            return ModelError(
                error_type=LLMErrorType.CONTEXT_TOO_LONG,
                message=f"上下文过长: {str(exception)}",
                details={"max_tokens": self.max_tokens, "original_error": str(exception)},
                suggestion="请减少输入长度或增加 max_tokens",
                retry_able=False
            )

        # 模型不存在
        if any(keyword in error_str for keyword in ["model not found", "invalid model", "404"]):
            return ModelError(
                error_type=LLMErrorType.MODEL_NOT_FOUND,
                message=f"模型不存在: {str(exception)}",
                details={"model_name": self.model_name, "original_error": str(exception)},
                suggestion=f"请检查模型名称 {self.model_name} 是否正确",
                retry_able=False
            )

        # 服务器错误 (5xx)
        if any(keyword in error_str for keyword in ["500", "502", "503", "504", "internal server error"]):
            return ModelError(
                error_type=LLMErrorType.SERVER_ERROR,
                message=f"服务器错误: {str(exception)}",
                details={"original_error": str(exception)},
                suggestion="服务器内部错误，请稍后重试",
                retry_able=True
            )

        # 默认未知错误
        return ModelError(
            error_type=LLMErrorType.UNKNOWN_ERROR,
            message=f"未知错误: {str(exception)}",
            details={"original_error": str(exception), "exception_type": type(exception).__name__},
            suggestion="请查看详细错误信息",
            retry_able=False
        )

    async def _acomplete_with_retry(self,
                                     prompt: str,
                                     system_prompt: Optional[str] = None,
                                     history_messages: list = None,
                                     **kwargs) -> str:
        """带重试的异步调用"""
        if history_messages is None:
            history_messages = []

        last_error = None

        for attempt in range(self.retry_config.max_attempts):
            try:
                self._call_count += 1

                # 记录尝试
                if attempt > 0:
                    delay = self.retry_config.get_delay(attempt - 1)
                    logger.info(f"LLM 重试 (尝试 {attempt + 1}/{self.retry_config.max_attempts})，延迟 {delay:.1f}s")
                    await asyncio.sleep(delay)
                    self._retry_count += 1

                # 调用 LLM
                response = await self._openai_complete(
                    self.model_name,
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    **kwargs
                )

                # 成功返回
                if attempt > 0:
                    logger.info(f"LLM 调用成功（重试 {attempt} 次后）")

                return response

            except Exception as e:
                last_error = e
                model_error = self._classify_error(e)

                # 记录错误
                logger.debug(f"LLM 调用失败 (尝试 {attempt + 1}/{self.retry_config.max_attempts}): {model_error.message}")

                # 判断是否可重试
                if not model_error.retry_able:
                    logger.error(f"LLM 调用失败（不可重试）: {model_error.message}")
                    self._error_count += 1
                    raise RuntimeError(model_error.message) from e

                # 检查是否还有重试机会
                if attempt >= self.retry_config.max_attempts - 1:
                    self._error_count += 1
                    logger.error(f"LLM 调用失败（已达最大重试次数）: {model_error.message}")
                    raise RuntimeError(f"LLM 调用失败（重试 {self.retry_config.max_attempts} 次）: {model_error.message}") from e

        # 不应该到这里
        raise RuntimeError(f"LLM 调用失败: {str(last_error)}")

    async def acomplete(self,
                       prompt: str,
                       system_prompt: Optional[str] = None,
                       history_messages: list = None,
                       **kwargs) -> str:
        """
        异步调用 LLM（带重试）

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            history_messages: 历史消息
            **kwargs: 额外参数

        Returns:
            模型响应文本
        """
        self._ensure_imports()

        try:
            return await self._acomplete_with_retry(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs
            )
        except RuntimeError as e:
            # 重新抛出我们处理过的错误
            raise
        except Exception as e:
            # 未预期的错误
            model_error = self._classify_error(e)
            self._error_count += 1
            raise RuntimeError(f"LLM 调用异常: {model_error.message}") from e

    async def acomplete_with_stream(self,
                                    prompt: str,
                                    system_prompt: Optional[str] = None,
                                    history_messages: list = None,
                                    **kwargs) -> Callable:
        """
        异步调用 LLM（流式输出）

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            history_messages: 历史消息
            **kwargs: 额外参数

        Returns:
            异步生成器，逐块返回响应
        """
        self._ensure_imports()
        # TODO: 实现流式输出
        raise NotImplementedError("流式输出功能待实现")

    def test_connection(self) -> dict:
        """
        测试 LLM 连接（增强版：详细诊断）

        Returns:
            连接状态字典
        """
        result = {
            "connected": False,
            "base_url": self.base_url,
            "model_name": self.model_name,
            "message": "",
            "details": {}
        }

        try:
            import requests

            # 测试基础连接
            try:
                response = requests.get(f"{self.base_url}/models", timeout=5)
                http_status = response.status_code
                result["details"]["http_status"] = http_status

                if http_status == 200:
                    data = response.json()
                    result["details"]["models"] = [m.get("id", "unknown") for m in data.get("data", [])]

                    # 检查模型是否存在
                    if any(self.model_name in m for m in result["details"]["models"]):
                        result["connected"] = True
                        result["message"] = f"连接成功，模型 {self.model_name} 可用"
                        logger.info(f"LLM 服务连接成功: {self.base_url}")
                    else:
                        result["message"] = f"连接成功，但模型 {self.model_name} 不在可用列表中"
                        result["message"] += f"\n可用模型: {result['details']['models']}"
                        logger.warning(result["message"])
                else:
                    result["message"] = f"服务响应异常: HTTP {http_status}"
                    logger.warning(result["message"])

            except requests.exceptions.Timeout:
                result["message"] = "连接超时，请检查服务是否运行"
                result["details"]["error_type"] = "timeout"
                logger.error(result["message"])

            except requests.exceptions.ConnectionError as e:
                result["message"] = f"无法连接到服务: {str(e)}"
                result["details"]["error_type"] = "connection_error"
                logger.error(result["message"])

        except ImportError:
            result["message"] = "requests 库未安装"
            result["details"]["error_type"] = "missing_dependency"
            logger.error(result["message"])

        except Exception as e:
            result["message"] = f"连接测试失败: {str(e)}"
            result["details"]["error_type"] = "unknown"
            logger.error(result["message"])

        return result

    def get_stats(self) -> dict:
        """获取服务统计信息"""
        return {
            "call_count": self._call_count,
            "retry_count": self._retry_count,
            "error_count": self._error_count,
            "success_rate": f"{((self._call_count - self._error_count) / max(self._call_count, 1) * 100):.1f}%"
        }


# ========== Embedding 服务 ==========

class EmbeddingService:
    """Embedding 服务类"""

    def __init__(self,
                 model_name: str = "BAAI/bge-m3",
                 device: str = "cuda:0",
                 embedding_dim: int = 1024,
                 max_token_size: int = 8192,
                 normalize: bool = True):
        """
        初始化 Embedding 服务

        Args:
            model_name: 模型名称
            device: 运行设备
            embedding_dim: 向量维度
            max_token_size: 最大 token 数
            normalize: 是否归一化
        """
        self.model_name = model_name
        self.device = device
        self.embedding_dim = embedding_dim
        self.max_token_size = max_token_size
        self.normalize = normalize

        self._model = None
        self._initialized = False

    def _ensure_imports(self):
        """确保必要的依赖已导入并加载模型"""
        if not self._initialized:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"正在加载 Embedding 模型: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                self._initialized = True
                logger.info(f"Embedding 模型加载成功，设备: {self.device}")
            except ImportError as e:
                logger.error(f"无法导入 sentence_transformers: {e}")
                raise ModelError(
                    error_type=EmbeddingErrorType.MODEL_LOAD_ERROR,
                    message=f"无法导入 sentence_transformers: {str(e)}",
                    suggestion="请运行: pip install sentence-transformers"
                )
            except Exception as e:
                error_msg = str(e)
                if "CUDA" in error_msg or "cuda" in error_msg:
                    logger.error(f"CUDA 错误: {e}")
                    raise ModelError(
                        error_type=EmbeddingErrorType.CUDA_ERROR,
                        message=f"CUDA 错误: {error_msg}",
                        details={"device": self.device, "original_error": error_msg},
                        suggestion="请检查 CUDA 是否可用，或尝试使用 CPU 模式 (device='cpu')"
                    )
                else:
                    logger.error(f"Embedding 模型加载失败: {e}")
                    raise ModelError(
                        error_type=EmbeddingErrorType.MODEL_LOAD_ERROR,
                        message=f"模型加载失败: {error_msg}",
                        details={"model_name": self.model_name, "device": self.device}
                    )

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        编码文本为向量

        Args:
            texts: 文本列表

        Returns:
            向量数组
        """
        self._ensure_imports()

        try:
            return self._model.encode(texts, normalize_embeddings=self.normalize)
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            raise ModelError(
                error_type=EmbeddingErrorType.ENCODING_ERROR,
                message=f"文本编码失败: {str(e)}",
                details={"text_count": len(texts)}
            )

    def encode_single(self, text: str) -> np.ndarray:
        """
        编码单条文本

        Args:
            text: 文本内容

        Returns:
            向量数组
        """
        return self.encode([text])[0]


def create_embedding_function(embedding_service: EmbeddingService):
    """
    创建 LightRAG 兼容的 embedding 函数

    Args:
        embedding_service: Embedding 服务实例

    Returns:
        包装后的 embedding 函数
    """
    from lightrag.utils import wrap_embedding_func_with_attrs

    @wrap_embedding_func_with_attrs(
        embedding_dim=embedding_service.embedding_dim,
        max_token_size=embedding_service.max_token_size,
        model_name=embedding_service.model_name
    )
    async def embedding_func(texts: list[str]) -> np.ndarray:
        return embedding_service.encode(texts)

    return embedding_func


# ========== 全局服务实例 ==========

_llm_service: Optional[LLMService] = None
_embedding_service: Optional[EmbeddingService] = None


def get_llm_service() -> LLMService:
    """获取全局 LLM 服务实例"""
    global _llm_service
    if _llm_service is None:
        raise RuntimeError("LLM 服务未初始化，请先调用 init_llm_service()")
    return _llm_service


def get_embedding_service() -> EmbeddingService:
    """获取全局 Embedding 服务实例"""
    global _embedding_service
    if _embedding_service is None:
        raise RuntimeError("Embedding 服务未初始化，请先调用 init_embedding_service()")
    return _embedding_service


def init_llm_service(model_name: str,
                    base_url: str,
                    api_key: str = "EMPTY",
                    max_tokens: int = 4096,
                    temperature: float = 0.7,
                    max_retries: int = 3,
                    retry_delay: float = 1.0) -> LLMService:
    """
    初始化全局 LLM 服务

    Args:
        model_name: 模型名称
        base_url: API 基础地址
        api_key: API 密钥
        max_tokens: 最大 token 数
        temperature: 温度参数
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）

    Returns:
        LLM 服务实例
    """
    global _llm_service

    # 创建重试配置
    retry_config = RetryConfig(
        max_attempts=max_retries,
        initial_delay=retry_delay
    )

    _llm_service = LLMService(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        retry_config=retry_config
    )

    return _llm_service


def init_embedding_service(model_name: str = "BAAI/bge-m3",
                          device: str = "cuda:0",
                          embedding_dim: int = 1024,
                          max_token_size: int = 8192,
                          normalize: bool = True) -> EmbeddingService:
    """
    初始化全局 Embedding 服务

    Args:
        model_name: 模型名称
        device: 运行设备
        embedding_dim: 向量维度
        max_token_size: 最大 token 数
        normalize: 是否归一化

    Returns:
        Embedding 服务实例
    """
    global _embedding_service
    _embedding_service = EmbeddingService(
        model_name=model_name,
        device=device,
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
        normalize=normalize
    )
    return _embedding_service


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 测试 LLM 服务
    print("=" * 50)
    print("测试 LLM 服务")
    print("=" * 50)

    llm = init_llm_service(
        model_name="Qwen/Qwen2.5-72B-Instruct",
        base_url="http://localhost:8000/v1",
        max_retries=3
    )

    # 测试连接
    test_result = llm.test_connection()
    print(f"连接测试: {test_result}")

    # 打印统计
    print(f"统计信息: {llm.get_stats()}")

    # 测试 Embedding 服务（需要 CUDA）
    print("\n" + "=" * 50)
    print("测试 Embedding 服务")
    print("=" * 50)

    try:
        embedding = init_embedding_service(device="cuda:0")
        test_texts = ["这是一个测试句子。", "这是另一个测试句子。"]
        vectors = embedding.encode(test_texts)
        print(f"向量形状: {vectors.shape}")
        print(f"向量维度: {vectors.shape[1]}")
    except Exception as e:
        print(f"Embedding 测试跳过（可能没有 CUDA）: {e}")
