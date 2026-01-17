"""
文件读取器模块
支持读取 PDF、TXT、DOCX、DOC 等多种格式
"""

import logging
from pathlib import Path
from typing import Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseReader(ABC):
    """文件读取器基类"""

    @abstractmethod
    def can_read(self, file_path: Path) -> bool:
        """判断是否支持读取该文件"""
        pass

    @abstractmethod
    def read(self, file_path: Path) -> str:
        """读取文件内容"""
        pass


class TextReader(BaseReader):
    """纯文本文件读取器 (.txt)"""

    def can_read(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".txt"

    def read(self, file_path: Path) -> str:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.error(f"读取文本文件失败 {file_path}: {e}")
            return ""


class PDFReader(BaseReader):
    """PDF 文件读取器 (.pdf)"""

    def can_read(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def read(self, file_path: Path) -> str:
        try:
            import fitz  # PyMuPDF
            with fitz.open(file_path) as doc:
                content = []
                for page in doc:
                    text = page.get_text()
                    if text.strip():
                        content.append(text)
                return "\n\n".join(content)
        except ImportError:
            logger.error("PyMuPDF 未安装，无法读取 PDF 文件")
            return ""
        except Exception as e:
            logger.error(f"读取 PDF 文件失败 {file_path}: {e}")
            return ""


class DocxReader(BaseReader):
    """Word 文档读取器 (.docx)"""

    def can_read(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".docx"

    def read(self, file_path: Path) -> str:
        try:
            from docx import Document
            doc = Document(file_path)
            content = []
            for para in doc.paragraphs:
                if para.text.strip():
                    content.append(para.text)
            return "\n\n".join(content)
        except ImportError:
            logger.error("python-docx 未安装，无法读取 DOCX 文件")
            return ""
        except Exception as e:
            logger.error(f"读取 DOCX 文件失败 {file_path}: {e}")
            return ""


class DocReader(BaseReader):
    """旧版 Word 文档读取器 (.doc)"""

    def can_read(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".doc"

    def read(self, file_path: Path) -> str:
        try:
            import docx2txt
            return docx2txt.process(file_path)
        except ImportError:
            logger.error("docx2txt 未安装，无法读取 DOC 文件")
            return ""
        except Exception as e:
            logger.error(f"读取 DOC 文件失败 {file_path}: {e}")
            return ""


class DocumentReader:
    """
    多格式文档读取器
    支持自动识别文件格式并使用对应的读取器
    """

    def __init__(self):
        """初始化读取器链"""
        self.readers: list[BaseReader] = [
            TextReader(),
            PDFReader(),
            DocxReader(),
            DocReader(),
        ]

    def add_reader(self, reader: BaseReader) -> None:
        """添加自定义读取器"""
        self.readers.insert(0, reader)

    def read(self, file_path: Union[str, Path]) -> str:
        """
        读取文件内容

        Args:
            file_path: 文件路径

        Returns:
            文件内容字符串
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return ""

        if not file_path.is_file():
            logger.warning(f"路径不是文件: {file_path}")
            return ""

        # 查找支持的读取器
        for reader in self.readers:
            if reader.can_read(file_path):
                logger.debug(f"使用 {reader.__class__.__name__} 读取: {file_path.name}")
                return reader.read(file_path)

        logger.warning(f"不支持的文件格式: {file_path.suffix}")
        return ""

    def read_directory(self,
                       directory: Union[str, Path],
                       recursive: bool = True,
                       min_length: int = 10) -> dict[str, str]:
        """
        读取目录下所有支持的文件

        Args:
            directory: 目录路径
            recursive: 是否递归读取子目录
            min_length: 文件最小字符数（过滤空文件）

        Returns:
            {文件路径: 文件内容} 字典
        """
        directory = Path(directory)

        if not directory.exists():
            logger.error(f"目录不存在: {directory}")
            return {}

        if not directory.is_dir():
            logger.error(f"路径不是目录: {directory}")
            return {}

        documents = {}
        pattern = "**/*" if recursive else "*"

        for file_path in directory.glob(pattern):
            if file_path.is_file():
                content = self.read(file_path)
                if len(content) >= min_length:
                    documents[str(file_path)] = content
                    logger.info(f"已读取: {file_path.name} ({len(content)} 字符)")
                elif content:
                    logger.debug(f"跳过过短文件: {file_path.name}")

        logger.info(f"共读取 {len(documents)} 个文件")
        return documents

    def read_directory_as_list(self,
                               directory: Union[str, Path],
                               recursive: bool = True,
                               min_length: int = 10) -> list[str]:
        """
        读取目录下所有文件，返回内容列表

        Args:
            directory: 目录路径
            recursive: 是否递归读取子目录
            min_length: 文件最小字符数

        Returns:
            文件内容列表
        """
        docs_dict = self.read_directory(directory, recursive, min_length)
        return list(docs_dict.values())


# 全局单例
_reader_instance: Optional[DocumentReader] = None


def get_reader() -> DocumentReader:
    """获取全局文档读取器实例"""
    global _reader_instance
    if _reader_instance is None:
        _reader_instance = DocumentReader()
    return _reader_instance


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    reader = get_reader()

    # 测试读取单个文件
    test_file = Path(__file__).parent.parent / "config.py"
    if test_file.exists():
        content = reader.read(test_file)
        print(f"读取 {test_file.name}: {len(content)} 字符")

    # 测试读取目录
    test_dir = Path(__file__).parent.parent / "src"
    if test_dir.exists():
        docs = reader.read_directory_as_list(test_dir, recursive=False)
        print(f"从 {test_dir} 读取了 {len(docs)} 个文件")
