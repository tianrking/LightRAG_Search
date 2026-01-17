"""
并发文档索引管理器
支持:
- 4 文件并发处理
- 进度持久化 (断点续传)
- 文件去重 (基于哈希)
- 实时进度显示
- 详细统计报告
"""

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Set, Dict, List
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class FileRecord:
    """文件处理记录"""
    path: str                    # 文件路径
    size: int                    # 文件大小
    mtime: float                 # 修改时间
    hash: str                    # 文件哈希 (MD5)
    indexed: bool = False        # 是否已索引
    index_time: Optional[str] = None  # 索引时间
    error: Optional[str] = None  # 错误信息
    doc_count: int = 0           # 提取的文档数
    file_type: str = ""          # 文件类型 (扩展名)

    def to_dict(self) -> dict:
        """转换为字典（确保 JSON 可序列化）"""
        data = asdict(self)
        # 确保 error 字段是字符串（防止意外传入非字符串类型）
        if data.get("error") is not None:
            data["error"] = str(data["error"])
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'FileRecord':
        return cls(**data)


@dataclass
class IndexProgress:
    """索引进度"""
    total_files: int = 0
    indexed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    start_time: Optional[str] = None
    update_time: Optional[str] = None
    current_file: Optional[str] = None

    @property
    def progress_percent(self) -> float:
        if self.total_files == 0:
            return 0.0
        return (self.indexed_files / self.total_files) * 100

    @property
    def remaining_files(self) -> int:
        return self.total_files - self.indexed_files - self.failed_files

    def to_dict(self) -> dict:
        return asdict(self)


class ProgressStorage:
    """进度持久化存储"""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.storage_path / "index_progress.json"
        self.records_file = self.storage_path / "file_records.json"
        self._lock = threading.Lock()

    def load_progress(self) -> IndexProgress:
        """加载进度"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return IndexProgress(**data)
            except Exception as e:
                logger.warning(f"加载进度失败: {e}")
        return IndexProgress()

    def save_progress(self, progress: IndexProgress) -> None:
        """保存进度"""
        with self._lock:
            try:
                progress.update_time = datetime.now().isoformat()
                with open(self.progress_file, 'w', encoding='utf-8') as f:
                    json.dump(progress.to_dict(), f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"保存进度失败: {e}")

    def load_records(self) -> Dict[str, FileRecord]:
        """加载文件记录"""
        if self.records_file.exists():
            try:
                with open(self.records_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {path: FileRecord.from_dict(rec) for path, rec in data.items()}
            except Exception as e:
                logger.warning(f"加载文件记录失败: {e}")
        return {}

    def save_records(self, records: Dict[str, FileRecord]) -> None:
        """保存文件记录"""
        with self._lock:
            try:
                data = {path: rec.to_dict() for path, rec in records.items()}
                with open(self.records_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"保存文件记录失败: {e}")

    def load(self) -> tuple[IndexProgress, Dict[str, FileRecord]]:
        """加载所有数据"""
        return self.load_progress(), self.load_records()

    def save(self, progress: IndexProgress, records: Dict[str, FileRecord]) -> None:
        """保存所有数据"""
        self.save_progress(progress)
        self.save_records(records)


class FileHasher:
    """文件哈希计算器 (用于去重)"""

    @staticmethod
    def compute_hash(file_path: Path, chunk_size: int = 8192) -> Optional[str]:
        """
        计算文件 MD5 哈希

        Args:
            file_path: 文件路径
            chunk_size: 读取块大小

        Returns:
            MD5 哈希值，失败返回 None
        """
        try:
            md5 = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(chunk_size), b''):
                    md5.update(chunk)
            return md5.hexdigest()
        except Exception as e:
            logger.error(f"计算哈希失败 {file_path.name}: {e}")
            return None


class ConcurrentIndexer:
    """
    并发文档索引器
    支持断点续传和去重
    """

    def __init__(self,
                 rag_engine,
                 extractor,
                 storage_dir: Path,
                 max_workers: int = 4,
                 supported_formats: tuple = (".pdf", ".txt", ".docx", ".doc")):
        """
        初始化并发索引器

        Args:
            rag_engine: RAG 引擎实例
            extractor: 文档提取器实例
            storage_dir: 进度存储目录
            max_workers: 最大并发数
            supported_formats: 支持的文件格式
        """
        self.rag_engine = rag_engine
        self.extractor = extractor
        self.storage = ProgressStorage(storage_dir)
        self.max_workers = max_workers
        self.supported_formats = supported_formats
        self.hasher = FileHasher()

        # 进度记录
        self.progress: IndexProgress
        self.file_records: Dict[str, FileRecord]
        self._seen_hashes: Set[str]

        # 线程锁
        self._index_lock = asyncio.Lock()
        self._print_lock = threading.Lock()

        # 加载历史进度
        self._load_history()

    def _load_history(self) -> None:
        """加载历史进度"""
        self.progress, self.file_records = self.storage.load()

        # 构建哈希集合 (用于去重)
        self._seen_hashes = {
            rec.hash for rec in self.file_records.values()
            if rec.hash and rec.indexed
        }

        logger.info(f"加载历史进度: {self.progress.indexed_files}/{self.progress.total_files} 已索引")
        if self._seen_hashes:
            logger.info(f"已识别 {len(self._seen_hashes)} 个唯一文件哈希")

    def _scan_directory(self, directory: Path, recursive: bool = True) -> List[Path]:
        """
        扫描目录获取文件列表

        Args:
            directory: 目录路径
            recursive: 是否递归

        Returns:
            文件路径列表
        """
        files = []
        pattern = "**/*" if recursive else "*"

        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                files.append(file_path)

        return sorted(files)

    def _should_process_file(self, file_path: Path) -> tuple[bool, Optional[str]]:
        """
        判断是否需要处理文件

        Args:
            file_path: 文件路径

        Returns:
            (是否处理, 原因)
        """
        path_str = str(file_path)

        # 获取文件信息
        try:
            stat = file_path.stat()
            size = stat.st_size
            mtime = stat.st_mtime
        except Exception as e:
            return False, f"无法获取文件信息: {e}"

        # 检查历史记录
        if path_str in self.file_records:
            record = self.file_records[path_str]

            # 已索引且文件未修改
            if record.indexed and record.mtime == mtime:
                return False, "已索引且未修改"

            # 已索引但文件已修改，需要重新索引
            if record.indexed and record.mtime != mtime:
                return True, "文件已修改，重新索引"

        # 计算哈希检查去重
        file_hash = self.hasher.compute_hash(file_path)
        if not file_hash:
            return True, "无法计算哈希，强制处理"

        # 检查是否与其他文件重复
        if file_hash in self._seen_hashes:
            # 查找重复文件
            for path, rec in self.file_records.items():
                if rec.hash == file_hash:
                    return False, f"与 {Path(path).name} 重复"

        return True, None

    def _update_record(self, file_path: Path, success: bool, error: Optional[str] = None, doc_count: int = 0) -> None:
        """更新文件记录"""
        path_str = str(file_path)

        try:
            stat = file_path.stat()
            file_hash = self.hasher.compute_hash(file_path) or ""

            record = FileRecord(
                path=path_str,
                size=stat.st_size,
                mtime=stat.st_mtime,
                hash=file_hash,
                indexed=success,
                index_time=datetime.now().isoformat() if success else None,
                error=error,
                doc_count=doc_count,
                file_type=file_path.suffix.lower()
            )

            self.file_records[path_str] = record

            if success and file_hash:
                self._seen_hashes.add(file_hash)

        except Exception as e:
            logger.error(f"更新记录失败 {file_path.name}: {e}")

    async def _process_file(self, file_path: Path) -> tuple[bool, int, Optional[str]]:
        """
        处理单个文件

        Args:
            file_path: 文件路径

        Returns:
            (是否成功, 文档数, 错误信息)
        """
        try:
            # 提取文档
            result = self.extractor.extract(file_path)

            if not result.success:
                # 将 ExtractionError 对象转换为字符串（修复 JSON 序列化错误）
                error_msg = result.error.message if result.error else "未知错误"
                return False, 0, error_msg

            # 索引入 RAG
            if result.content:
                async with self._index_lock:
                    await self.rag_engine.insert_documents([result.content])
                return True, 1, None
            else:
                return False, 0, "提取内容为空"

        except Exception as e:
            logger.error(f"处理文件失败 {file_path.name}: {e}")
            return False, 0, str(e)

    async def index_directory(self,
                             directory: Path,
                             recursive: bool = True,
                             batch_size: int = 4) -> IndexProgress:
        """
        索引目录

        Args:
            directory: 目录路径
            recursive: 是否递归
            batch_size: 批处理大小

        Returns:
            索引进度
        """
        logger.info(f"开始扫描目录: {directory}")

        # 扫描文件
        files = self._scan_directory(directory, recursive)
        self.progress.total_files = len(files)

        if not files:
            logger.warning("没有找到需要处理的文件")
            return self.progress

        # 设置开始时间
        if not self.progress.start_time:
            self.progress.start_time = datetime.now().isoformat()

        logger.info(f"找到 {len(files)} 个文件，开始处理...")

        # 过滤需要处理的文件
        pending_files = []
        skipped = 0

        for file_path in files:
            should_process, reason = self._should_process_file(file_path)
            if should_process:
                pending_files.append(file_path)
            else:
                skipped += 1
                logger.debug(f"跳过 {file_path.name}: {reason}")

        self.progress.skipped_files = skipped
        logger.info(f"需要处理 {len(pending_files)} 个文件，跳过 {skipped} 个")

        # 并发处理
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_with_semaphore(file_path: Path) -> tuple[Path, bool, int, Optional[str]]:
            async with semaphore:
                self.progress.current_file = str(file_path)
                self._print_progress()

                success, doc_count, error = await self._process_file(file_path)

                # 更新记录
                self._update_record(file_path, success, error, doc_count)

                # 更新进度
                if success:
                    self.progress.indexed_files += 1
                else:
                    self.progress.failed_files += 1

                # 定期保存
                if (self.progress.indexed_files + self.progress.failed_files) % 10 == 0:
                    self.storage.save(self.progress, self.file_records)

                return file_path, success, doc_count, error

        # 执行并发处理
        results = await asyncio.gather(
            *[process_with_semaphore(fp) for fp in pending_files],
            return_exceptions=True
        )

        # 最终保存
        self.storage.save(self.progress, self.file_records)

        # 打印总结
        self._print_summary()

        return self.progress

    def _print_progress(self) -> None:
        """打印进度"""
        with self._print_lock:
            percent = self.progress.progress_percent
            current = Path(self.progress.current_file).name if self.progress.current_file else ""

            print(f"\r[进度: {percent:.1f}%] "
                  f"已索引: {self.progress.indexed_files} | "
                  f"失败: {self.progress.failed_files} | "
                  f"跳过: {self.progress.skipped_files} | "
                  f"剩余: {self.progress.remaining_files} | "
                  f"当前: {current[:50]}", end="", flush=True)

    def _print_summary(self) -> None:
        """打印详细总结"""
        print("\n" + "=" * 80)
        print("索引完成总结")
        print("=" * 80)

        # 基本统计
        print(f"总文件数:     {self.progress.total_files}")
        print(f"成功索引:     {self.progress.indexed_files}")
        print(f"失败:         {self.progress.failed_files}")
        print(f"跳过:         {self.progress.skipped_files}")
        print(f"进度:         {self.progress.progress_percent:.1f}%")

        # 时间统计
        if self.progress.start_time and self.progress.update_time:
            start = datetime.fromisoformat(self.progress.start_time)
            end = datetime.fromisoformat(self.progress.update_time)
            duration = (end - start).total_seconds()
            print(f"耗时:         {duration:.1f} 秒 ({duration/60:.1f} 分钟)")

            # 速度统计
            if self.progress.indexed_files > 0:
                speed = self.progress.indexed_files / (duration / 60)
                print(f"速度:         {speed:.1f} 文件/分钟")

        print("-" * 80)

        # 按文件类型统计
        type_stats = defaultdict(lambda: {"total": 0, "success": 0, "failed": 0})
        for rec in self.file_records.values():
            file_type = rec.file_type or "unknown"
            type_stats[file_type]["total"] += 1
            if rec.indexed:
                type_stats[file_type]["success"] += 1
            if rec.error:
                type_stats[file_type]["failed"] += 1

        print("按文件类型统计:")
        print(f"{'类型':<12} {'总数':>8} {'成功':>8} {'失败':>8} {'成功率':>10}")
        print("-" * 50)

        for file_type in sorted(type_stats.keys()):
            stats = type_stats[file_type]
            success_rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"{file_type:<12} {stats['total']:>8} {stats['success']:>8} {stats['failed']:>8} {success_rate:>9.1f}%")

        print("=" * 80)

        # 失败文件列表
        failed_files = self.get_failed_files()
        if failed_files:
            print(f"\n失败文件列表 ({len(failed_files)} 个):")
            print("-" * 80)

            # 按错误类型分组
            error_groups = defaultdict(list)
            for rec in failed_files:
                error_msg = rec.error or "未知错误"
                error_groups[error_msg].append(rec)

            for error_msg, files in error_groups.items():
                print(f"\n错误: {error_msg} ({len(files)} 个文件)")
                for rec in files[:5]:  # 每种错误只显示前5个
                    file_path = Path(rec.path)
                    rel_path = file_path.relative_to(file_path.anchor)
                    print(f"  - {rel_path}")

                if len(files) > 5:
                    print(f"  ... 还有 {len(files) - 5} 个文件")

            print("\n" + "=" * 80)
            print(f"可以使用 --retry-failed 参数重试失败的文件")
            print("=" * 80)

    def get_statistics_report(self) -> str:
        """
        生成详细统计报告（文本格式）

        Returns:
            统计报告文本
        """
        lines = []
        lines.append("=" * 80)
        lines.append("LightRAG 索引统计报告")
        lines.append("=" * 80)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 基本统计
        lines.append("基本统计:")
        lines.append(f"  总文件数:   {self.progress.total_files}")
        lines.append(f"  成功索引:   {self.progress.indexed_files}")
        lines.append(f"  失败:       {self.progress.failed_files}")
        lines.append(f"  跳过:       {self.progress.skipped_files}")
        lines.append("")

        # 按文件类型统计
        type_stats = defaultdict(lambda: {"total": 0, "success": 0, "failed": 0})
        for rec in self.file_records.values():
            file_type = rec.file_type or "unknown"
            type_stats[file_type]["total"] += 1
            if rec.indexed:
                type_stats[file_type]["success"] += 1
            if rec.error:
                type_stats[file_type]["failed"] += 1

        lines.append("按文件类型统计:")
        for file_type in sorted(type_stats.keys()):
            stats = type_stats[file_type]
            success_rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
            lines.append(f"  {file_type:<12}: 总数={stats['total']:>4} 成功={stats['success']:>4} 失败={stats['failed']:>4} 成功率={success_rate:>6.1f}%")
        lines.append("")

        # 失败文件详情
        failed_files = self.get_failed_files()
        if failed_files:
            lines.append(f"失败文件详情 ({len(failed_files)} 个):")
            for rec in failed_files:
                file_path = Path(rec.path)
                lines.append(f"  - {file_path.name}: {rec.error}")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def save_statistics_report(self, report_path: Path) -> None:
        """
        保存统计报告到文件

        Args:
            report_path: 报告文件路径
        """
        report = self.get_statistics_report()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"统计报告已保存: {report_path}")

    def get_failed_files(self) -> List[FileRecord]:
        """获取失败的文件列表"""
        return [rec for rec in self.file_records.values() if not rec.indexed and rec.error]

    def retry_failed(self) -> None:
        """重置失败的文件状态，以便重试"""
        for rec in self.file_records.values():
            if not rec.indexed and rec.error:
                rec.error = None
        logger.info(f"已重置 {len(self.get_failed_files())} 个失败文件的状态")


def get_index_manager(rag_engine, extractor, storage_dir: Path, max_workers: int = 4) -> ConcurrentIndexer:
    """
    获取索引管理器实例

    Args:
        rag_engine: RAG 引擎
        extractor: 文档提取器
        storage_dir: 存储目录
        max_workers: 最大并发数

    Returns:
        索引管理器实例
    """
    return ConcurrentIndexer(
        rag_engine=rag_engine,
        extractor=extractor,
        storage_dir=storage_dir,
        max_workers=max_workers
    )


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    storage = ProgressStorage(Path("./data/progress"))

    # 测试保存/加载
    progress = IndexProgress(total_files=100, indexed_files=50)
    records = {
        "/test/file1.pdf": FileRecord(
            path="/test/file1.pdf",
            size=1024,
            mtime=1234567890.0,
            hash="abc123",
            indexed=True
        )
    }

    storage.save(progress, records)

    loaded_progress, loaded_records = storage.load()
    print(f"进度: {loaded_progress.indexed_files}/{loaded_progress.total_files}")
    print(f"记录: {len(loaded_records)}")
