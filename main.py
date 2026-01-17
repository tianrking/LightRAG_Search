"""
LightRAG 主程序
用于处理和分析 HKIPO 资源文件
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, setup_paths
from src.extractors import get_extractor
from src.models import init_llm_service, init_embedding_service, create_embedding_function
from src.rag_engine import init_rag_engine
from src.index_manager import get_index_manager


def setup_logging(log_level: str = "INFO"):
    """配置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/lightrag.log', encoding='utf-8')
        ]
    )


async def index_documents(input_dir: str = None,
                         recursive: bool = True,
                         concurrent: bool = True,
                         max_workers: int = 4,
                         retry_failed: bool = False) -> None:
    """
    索引文档目录（支持并发、断点续传、去重）

    Args:
        input_dir: 输入目录路径（默认使用配置中的路径）
        recursive: 是否递归读取子目录
        concurrent: 是否启用并发处理
        max_workers: 最大并发数（默认4）
        retry_failed: 是否重试失败的文件
    """
    cfg = get_config()

    # 设置路径
    if input_dir:
        setup_paths(input_dir=input_dir)
    cfg.setup()

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("LightRAG 文档索引模式")
    logger.info("=" * 80)

    # 初始化服务
    logger.info("正在初始化 LLM 服务...")
    llm = init_llm_service(
        model_name=cfg.llm.MODEL_NAME,
        base_url=cfg.llm.BASE_URL,
        api_key=cfg.llm.API_KEY,
        max_tokens=cfg.llm.MAX_TOKENS,
        temperature=cfg.llm.TEMPERATURE
    )

    # 测试 LLM 连接
    if not llm.test_connection():
        logger.error("LLM 服务连接失败，请检查 vLLM 是否正常运行")
        return

    logger.info("正在初始化 Embedding 服务...")
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
        embedding_func=embedding_func,
        chunk_size=cfg.rag.CHUNK_SIZE
    )

    await engine.initialize()

    # 索引文档
    if concurrent:
        # 使用并发索引管理器
        logger.info("使用并发索引模式 (支持断点续传和去重)")
        # 根据配置启用 OCR 和指定引擎
        extractor = get_extractor(
            enable_ocr=cfg.rag.ENABLE_OCR,
            ocr_engine_type=cfg.rag.OCR_ENGINE
        )
        logger.info(f"OCR 状态: {'启用 (' + cfg.rag.OCR_ENGINE + ')' if cfg.rag.ENABLE_OCR else '禁用'}")
        progress_dir = cfg.paths.PROJECT_ROOT / "data" / "index_progress"

        index_manager = get_index_manager(
            rag_engine=engine,
            extractor=extractor,
            storage_dir=progress_dir,
            max_workers=max_workers
        )

        # 重试失败文件
        if retry_failed:
            logger.info("重试失败的文件...")
            index_manager.retry_failed()

        # 执行索引
        progress = await index_manager.index_directory(
            directory=cfg.paths.INPUT_DIR,
            recursive=recursive
        )

        # 保存统计报告
        report_path = cfg.paths.LOG_DIR / f"index_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        index_manager.save_statistics_report(report_path)

        # 打印失败文件
        failed = index_manager.get_failed_files()
        if failed:
            logger.info(f"\n有 {len(failed)} 个文件索引失败，详见统计报告: {report_path}")

    else:
        # 使用简单的顺序索引
        logger.info("使用顺序索引模式")
        extractor = get_extractor(
            enable_ocr=cfg.rag.ENABLE_OCR,
            ocr_engine_type=cfg.rag.OCR_ENGINE
        )

        logger.info(f"正在读取文档目录: {cfg.paths.INPUT_DIR}")
        documents = {}

        for file_path, result in extractor.extract_directory(
            cfg.paths.INPUT_DIR,
            recursive=recursive,
            min_length=10
        ):
            if result.success:
                documents[file_path] = result.content

        if not documents:
            logger.warning("没有找到可读取的文档")
            return

        logger.info(f"共读取 {len(documents)} 个文档")

        # 插入文档
        logger.info("正在建立索引...")
        await engine.insert_documents_from_dict(
            documents,
            batch_size=cfg.rag.BATCH_SIZE
        )

    logger.info("索引建立完成！")
    logger.info("=" * 80)


async def query_mode(query_text: str, mode: str = "hybrid") -> str:
    """
    查询模式

    Args:
        query_text: 查询文本
        mode: 查询模式 (naive, local, global, hybrid)

    Returns:
        查询结果
    """
    cfg = get_config()
    cfg.setup()

    logger = logging.getLogger(__name__)

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

    # 执行查询
    result = await engine.query(query_text, mode=mode)

    return result


async def interactive_mode() -> None:
    """交互式查询模式"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("LightRAG 交互式查询模式")
    logger.info("=" * 60)

    print("\n可用的查询模式:")
    print("  - naive:   简单检索")
    print("  - local:   局部知识图谱检索")
    print("  - global:  全局知识图谱检索")
    print("  - hybrid:  混合模式（推荐）")
    print("\n输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            query = input("\n请输入你的问题: ").strip()

            if query.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break

            if not query:
                continue

            mode_input = input("查询模式 (默认: hybrid): ").strip() or "hybrid"

            print(f"\n正在查询 (模式: {mode_input})...")
            result = await query_mode(query, mode=mode_input)

            print("\n" + "=" * 60)
            print("回答:")
            print("=" * 60)
            print(result)
            print("=" * 60)

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            logger.error(f"查询出错: {e}")
            print(f"查询出错: {e}")


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="LightRAG - 智能文档检索系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 并发索引（支持断点续传）
  python main.py index --input /path/to/docs --concurrent --workers 4

  # 索引失败后重试
  python main.py index --retry-failed

  # 查询文档
  python main.py query --query "专利申请流程" --mode hybrid

  # 交互模式
  python main.py interactive
        """
    )
    parser.add_argument("command", choices=["index", "query", "interactive"],
                       help="命令: index(索引文档), query(单次查询), interactive(交互模式)")
    parser.add_argument("--input", "-i", help="输入目录路径")
    parser.add_argument("--query", "-q", help="查询文本")
    parser.add_argument("--mode", "-m", default="hybrid",
                       choices=["naive", "local", "global", "hybrid"],
                       help="查询模式 (默认: hybrid)")
    parser.add_argument("--log-level", "-l", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别 (默认: INFO)")

    # 索引相关参数
    parser.add_argument("--recursive", "-r", action="store_true", default=True,
                       help="递归读取子目录 (默认: True)")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false",
                       help="不递归读取子目录")
    parser.add_argument("--concurrent", "-c", action="store_true", default=True,
                       help="启用并发处理 (默认: True)")
    parser.add_argument("--no-concurrent", dest="concurrent", action="store_false",
                       help="禁用并发处理")
    parser.add_argument("--workers", "-w", type=int, default=4,
                       help="并发处理数 (默认: 4)")
    parser.add_argument("--retry-failed", action="store_true",
                       help="重试之前索引失败的文件")

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_level)

    try:
        if args.command == "index":
            await index_documents(
                input_dir=args.input,
                recursive=args.recursive,
                concurrent=args.concurrent,
                max_workers=args.workers,
                retry_failed=args.retry_failed
            )
        elif args.command == "query":
            if not args.query:
                print("错误: 查询模式需要 --query 参数")
                sys.exit(1)
            result = await query_mode(args.query, args.mode)
            print("\n回答:\n")
            print(result)
        elif args.command == "interactive":
            await interactive_mode()

    except KeyboardInterrupt:
        print("\n\n操作已取消")
        sys.exit(0)
    except Exception as e:
        logging.getLogger(__name__).error(f"程序异常: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
