"""
LightRAG Backend 客户端示例
展示如何使用 backend 模块
"""

import asyncio
import logging
from pathlib import Path

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend import RAGService, QueryRequest, QueryMode, ServiceConfig
from backend.api import initialize, shutdown, query, query_simple, get_status, is_ready, RAGBackend


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_1_basic_usage():
    """示例 1: 基础使用 - 直接使用 API 函数"""
    logger.info("=" * 60)
    logger.info("示例 1: 基础使用")
    logger.info("=" * 60)

    try:
        # 1. 初始化服务
        logger.info("正在初始化服务...")
        await initialize()
        logger.info("服务初始化完成")

        # 2. 检查状态
        status = get_status()
        logger.info(f"服务状态: {status.to_dict()}")

        if not status.is_ready:
            logger.warning("服务未就绪，请先建立索引")
            return

        # 3. 执行查询
        query_text = "专利申请流程是什么？"
        logger.info(f"查询: {query_text}")

        result = await query_simple(query_text, mode="hybrid")
        logger.info(f"结果: {result[:200]}...")

    except Exception as e:
        logger.error(f"错误: {e}")

    finally:
        # 4. 关闭服务
        await shutdown()


async def example_2_service_object():
    """示例 2: 使用服务对象"""
    logger.info("=" * 60)
    logger.info("示例 2: 使用服务对象")
    logger.info("=" * 60)

    from config import get_config

    try:
        # 1. 创建配置
        cfg = get_config()

        service_config = ServiceConfig(
            llm_model_name=cfg.llm.MODEL_NAME,
            llm_base_url=cfg.llm.BASE_URL,
            llm_api_key=cfg.llm.API_KEY,
            embedding_model_name=cfg.embedding.MODEL_NAME,
            embedding_device=cfg.embedding.DEVICE,
            embedding_dim=cfg.embedding.EMBEDDING_DIM,
            working_dir=str(cfg.paths.WORKING_DIR),
            chunk_size=cfg.rag.CHUNK_SIZE,
        )

        # 2. 创建服务
        service = RAGService(service_config)

        # 3. 启动服务
        await service.start()
        logger.info("服务启动完成")

        # 4. 执行查询
        request = QueryRequest(
            query="如何申请专利？",
            mode=QueryMode.HYBRID,
        )

        response = await service.query(request)
        logger.info(f"查询结果: {response.answer[:200]}...")
        logger.info(f"查询耗时: {response.latency_ms:.2f}ms")

        # 5. 获取状态
        status = service.get_status()
        logger.info(f"服务状态: {status.to_dict()}")

        # 6. 停止服务
        await service.stop()

    except Exception as e:
        logger.error(f"错误: {e}")


async def example_3_context_manager():
    """示例 3: 使用上下文管理器"""
    logger.info("=" * 60)
    logger.info("示例 3: 使用上下文管理器")
    logger.info("=" * 60)

    async with RAGBackend() as backend:
        # 检查状态
        status = backend.status()
        logger.info(f"服务状态: {status.to_dict()}")

        if not status.is_ready:
            logger.warning("服务未就绪，请先建立索引")
            return

        # 执行查询
        result = await backend.query("专利申请需要什么材料？")
        logger.info(f"结果: {result[:200]}...")


async def example_4_batch_queries():
    """示例 4: 批量查询"""
    logger.info("=" * 60)
    logger.info("示例 4: 批量查询")
    logger.info("=" * 60)

    try:
        await initialize()

        queries = [
            "什么是专利？",
            "专利的类型有哪些？",
            "如何申请专利？",
        ]

        for q in queries:
            logger.info(f"\n查询: {q}")
            result = await query_simple(q, mode="hybrid")
            logger.info(f"结果: {result[:150]}...")

    except Exception as e:
        logger.error(f"错误: {e}")

    finally:
        await shutdown()


async def example_5_different_modes():
    """示例 5: 不同查询模式"""
    logger.info("=" * 60)
    logger.info("示例 5: 不同查询模式")
    logger.info("=" * 60)

    try:
        await initialize()

        query_text = "专利申请流程"
        modes = ["naive", "local", "global", "hybrid"]

        for mode in modes:
            logger.info(f"\n模式: {mode}")
            result = await query_simple(query_text, mode=mode)
            logger.info(f"结果: {result[:150]}...")

    except Exception as e:
        logger.error(f"错误: {e}")

    finally:
        await shutdown()


async def main():
    """主函数"""
    examples = [
        ("基础使用", example_1_basic_usage),
        ("服务对象", example_2_service_object),
        ("上下文管理器", example_3_context_manager),
        ("批量查询", example_4_batch_queries),
        ("不同模式", example_5_different_modes),
    ]

    print("可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print("  all. 运行所有示例")
    print("  0. 退出")

    choice = input("\n请选择示例 (1-5, all, 0): ").strip()

    if choice == "0":
        return
    elif choice == "all":
        for _, func in examples:
            try:
                await func()
                print()
            except Exception as e:
                logger.error(f"示例执行失败: {e}")
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                await examples[idx][1]()
            else:
                print("无效的选择")
        except ValueError:
            print("无效的选择")
        except Exception as e:
            logger.error(f"示例执行失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())
