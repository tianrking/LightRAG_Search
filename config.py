"""
LightRAG 项目配置管理模块
管理所有路径、模型参数和系统配置
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PathConfig:
    """路径配置"""
    # 项目根目录
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).parent)

    # 输入数据目录（默认 HKIPO-中介公共资料）
    INPUT_DIR: Path = field(default_factory=lambda: Path("/home/tianrui/gotohk/private_ragflow/workspace/HKIPO_resource/HKIPO-中介公共资料"))

    # 工作目录（存储向量数据库等）
    WORKING_DIR: Path = field(default_factory=lambda: Path("./data/rag_storage"))

    # 日志目录
    LOG_DIR: Path = field(default_factory=lambda: Path("./logs"))

    # 配置文件目录
    CONFIG_DIR: Path = field(default_factory=lambda: Path("./config"))

    def __post_init__(self):
        """确保所有路径都是 Path 对象"""
        self.PROJECT_ROOT = Path(self.PROJECT_ROOT)
        self.INPUT_DIR = Path(self.INPUT_DIR)
        self.WORKING_DIR = Path(self.WORKING_DIR)
        self.LOG_DIR = Path(self.LOG_DIR)
        self.CONFIG_DIR = Path(self.CONFIG_DIR)

    def ensure_dirs(self):
        """确保所有必要的目录存在"""
        self.WORKING_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class LLMConfig:
    """LLM 模型配置"""
    # 模型名称（使用 vLLM 路径模式启动时的完整路径）
    MODEL_NAME: str = "/home/tianrui/gotohk/private_ragflow/Qwen2.5-3B"

    # API 基础地址
    BASE_URL: str = "http://localhost:38199/v1"

    # API 密钥（vLLM 通常不需要）
    API_KEY: str = "EMPTY"

    # 最大 token 数
    MAX_TOKENS: int = 4096

    # 温度参数
    TEMPERATURE: float = 0.7

    # 超时时间（秒）
    TIMEOUT: int = 120


@dataclass
class EmbeddingConfig:
    """Embedding 模型配置"""
    # 模型名称
    MODEL_NAME: str = "BAAI/bge-m3"

    # 设备
    DEVICE: str = "cuda:0"

    # 向量维度
    EMBEDDING_DIM: int = 1024

    # 最大 token 数
    MAX_TOKEN_SIZE: int = 8192

    # 是否归一化
    NORMALIZE: bool = True


@dataclass
class RAGConfig:
    """RAG 系统配置"""
    # 查询模式: naive, local, global, hybrid
    QUERY_MODE: str = "hybrid"

    # 批处理大小
    BATCH_SIZE: int = 16

    # 文档读取最大字符数
    MAX_DOC_SIZE: int = 1000000

    # 支持的文件格式
    SUPPORTED_FORMATS: tuple = field(default_factory=lambda: (".pdf", ".txt", ".docx", ".doc"))

    # 文本块大小（tokens）
    CHUNK_SIZE: int = 1024

    # 是否启用增量索引
    ENABLE_INCREMENTAL: bool = True


@dataclass
class GPUConfig:
    """GPU 资源配置"""
    # GPU 数量
    GPU_COUNT: int = 1

    # vLLM 张并行大小
    TENSOR_PARALLEL_SIZE: int = 1

    # GPU 显存利用率 (0.0-1.0)
    GPU_MEMORY_UTILIZATION: float = 0.80

    # vLLM 最大并发序列数
    MAX_NUM_SEQS: int = 256

    # vLLM 批处理 token 数
    MAX_NUM_BATCHED_TOKENS: int = 8192

    # vLLM 最大模型长度
    MAX_MODEL_LEN: int = 32768

    # Embedding 模型设备分配
    EMBEDDING_DEVICE: str = "cuda:0"  # 第一张 GPU

    # 是否启用量化
    ENABLE_QUANTIZATION: bool = False
    QUANTIZATION_METHOD: str = "bitsandbytes"  # bitsandbytes, awq, gptq

    def get_vllm_args(self) -> dict:
        """获取 vLLM 启动参数"""
        return {
            "tensor_parallel_size": self.TENSOR_PARALLEL_SIZE,
            "gpu_memory_utilization": self.GPU_MEMORY_UTILIZATION,
            "max_num_seqs": self.MAX_NUM_SEQS,
            "max_num_batched_tokens": self.MAX_NUM_BATCHED_TOKENS,
            "max_model_len": self.MAX_MODEL_LEN,
        }

    def get_embedding_device(self) -> str:
        """获取 Embedding 模型设备"""
        return self.EMBEDDING_DEVICE


@dataclass
class AppConfig:
    """应用主配置"""
    # 调试模式
    DEBUG: bool = True

    # 日志级别
    LOG_LEVEL: str = "INFO"

    # 是否启用缓存
    ENABLE_CACHE: bool = True

    def __post_init__(self):
        """初始化子配置"""
        self.paths = PathConfig()
        self.llm = LLMConfig()
        self.embedding = EmbeddingConfig()
        self.rag = RAGConfig()
        self.gpu = GPUConfig()

    def setup(self):
        """设置应用配置"""
        self.paths.ensure_dirs()


# 全局配置实例
config = AppConfig()


def get_config() -> AppConfig:
    """获取全局配置实例"""
    return config


def setup_paths(input_dir: Optional[str] = None,
                working_dir: Optional[str] = None) -> None:
    """
    设置自定义路径

    Args:
        input_dir: 输入文件目录
        working_dir: RAG 工作目录
    """
    if input_dir:
        config.paths.INPUT_DIR = Path(input_dir)
    if working_dir:
        config.paths.WORKING_DIR = Path(working_dir)
    config.paths.ensure_dirs()


if __name__ == "__main__":
    # 测试配置
    cfg = get_config()
    cfg.setup()

    print("=" * 50)
    print("LightRAG 项目配置")
    print("=" * 50)
    print(f"项目根目录: {cfg.paths.PROJECT_ROOT}")
    print(f"输入目录: {cfg.paths.INPUT_DIR}")
    print(f"工作目录: {cfg.paths.WORKING_DIR}")
    print(f"日志目录: {cfg.paths.LOG_DIR}")
    print("-" * 50)
    print(f"LLM 模型: {cfg.llm.MODEL_NAME}")
    print(f"LLM 地址: {cfg.llm.BASE_URL}")
    print("-" * 50)
    print(f"Embedding 模型: {cfg.embedding.MODEL_NAME}")
    print(f"Embedding 设备: {cfg.embedding.DEVICE}")
    print("-" * 50)
    print(f"查询模式: {cfg.rag.QUERY_MODE}")
    print(f"支持格式: {cfg.rag.SUPPORTED_FORMATS}")
    print("=" * 50)
