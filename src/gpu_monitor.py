"""
GPU 资源监控工具
监控 8x RTX 4090 的使用情况
"""

import subprocess
import logging
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """单个 GPU 信息"""
    index: int
    name: str
    memory_total: int       # MB
    memory_used: int        # MB
    memory_free: int        # MB
    utilization: int        # %
    temperature: int        # °C
    power_usage: int        # W
    power_limit: int        # W

    @property
    def memory_percent(self) -> float:
        """显存使用率"""
        return (self.memory_used / self.memory_total) * 100 if self.memory_total > 0 else 0


@dataclass
class GPUStats:
    """GPU 集群统计"""
    gpus: list[GPUInfo]
    timestamp: datetime

    @property
    def total_memory(self) -> int:
        """总显存 (MB)"""
        return sum(gpu.memory_total for gpu in self.gpus)

    @property
    def total_memory_used(self) -> int:
        """已用显存 (MB)"""
        return sum(gpu.memory_used for gpu in self.gpus)

    @property
    def total_memory_free(self) -> int:
        """空闲显存 (MB)"""
        return sum(gpu.memory_free for gpu in self.gpus)

    @property
    def average_utilization(self) -> float:
        """平均 GPU 利用率"""
        if not self.gpus:
            return 0.0
        return sum(gpu.utilization for gpu in self.gpus) / len(self.gpus)

    @property
    def total_power_usage(self) -> int:
        """总功耗 (W)"""
        return sum(gpu.power_usage for gpu in self.gpus)


class GPUMonitor:
    """GPU 监控器"""

    def __init__(self):
        """初始化 GPU 监控器"""
        self._check_nvidia_smi()

    def _check_nvidia_smi(self) -> None:
        """检查 nvidia-smi 是否可用"""
        try:
            subprocess.run(
                ["nvidia-smi", "--version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("nvidia-smi 不可用，请检查 NVIDIA 驱动是否安装")

    def get_gpu_info(self, gpu_index: int) -> Optional[GPUInfo]:
        """
        获取单个 GPU 信息

        Args:
            gpu_index: GPU 索引

        Returns:
            GPU 信息
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={gpu_index}",
                    "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                check=True
            )

            values = result.stdout.strip().split(", ")
            return GPUInfo(
                index=int(values[0]),
                name=values[1],
                memory_total=int(values[2]),
                memory_used=int(values[3]),
                memory_free=int(values[4]),
                utilization=int(values[5]),
                temperature=int(values[6]),
                power_usage=int(float(values[7])),
                power_limit=int(float(values[8]))
            )

        except (subprocess.CalledProcessError, IndexError, ValueError) as e:
            logger.error(f"获取 GPU {gpu_index} 信息失败: {e}")
            return None

    def get_all_gpus(self) -> list[GPUInfo]:
        """
        获取所有 GPU 信息

        Returns:
            GPU 信息列表
        """
        gpus = []

        # 先获取 GPU 数量
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            gpu_count = int(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            gpu_count = 8  # 默认 8 张

        for i in range(gpu_count):
            if gpu_info := self.get_gpu_info(i):
                gpus.append(gpu_info)

        return gpus

    def get_stats(self) -> GPUStats:
        """
        获取 GPU 集群统计

        Returns:
            GPU 统计信息
        """
        return GPUStats(
            gpus=self.get_all_gpus(),
            timestamp=datetime.now()
        )

    def print_stats(self, stats: Optional[GPUStats] = None) -> None:
        """
        打印 GPU 状态

        Args:
            stats: GPU 统计信息（默认获取最新）
        """
        if stats is None:
            stats = self.get_stats()

        print("\n" + "=" * 80)
        print("GPU 集群状态".center(80))
        print("=" * 80)

        for gpu in stats.gpus:
            print(f"\nGPU {gpu.index}: {gpu.name}")
            print(f"  显存: {gpu.memory_used:5}/{gpu.memory_total:5} MB ({gpu.memory_percent:5.1f}%)")
            print(f"  利用率: {gpu.utilization:3}%")
            print(f"  温度: {gpu.temperature:3}°C")
            print(f"  功耗: {gpu.power_usage:3}/{gpu.power_limit:3} W")

        print("\n" + "-" * 80)
        print("汇总:")
        print(f"  总显存: {stats.total_memory_used}/{stats.total_memory} MB")
        print(f"  平均利用率: {stats.average_utilization:.1f}%")
        print(f"  总功耗: {stats.total_power_usage} W")
        print("=" * 80 + "\n")

    def check_availability(self, required_memory_mb: int, gpu_count: int = 1) -> bool:
        """
        检查 GPU 资源是否可用

        Args:
            required_memory_mb: 每张 GPU 需要的显存 (MB)
            gpu_count: 需要的 GPU 数量

        Returns:
            是否可用
        """
        gpus = self.get_all_gpus()

        available_gpus = [
            gpu for gpu in gpus
            if gpu.memory_free >= required_memory_mb
        ]

        return len(available_gpus) >= gpu_count

    def get_optimal_gpu_for_embedding(self) -> str:
        """
        获取最适合运行 Embedding 模型的 GPU

        策略: 选择显存使用率最低的 GPU
        """
        gpus = self.get_all_gpus()

        if not gpus:
            return "cpu"

        # 选择显存使用率最低的
        optimal_gpu = min(gpus, key=lambda g: g.memory_percent)

        return f"cuda:{optimal_gpu.index}"


# 全局单例
_gpu_monitor: Optional[GPUMonitor] = None


def get_gpu_monitor() -> GPUMonitor:
    """获取全局 GPU 监控器实例"""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor()
    return _gpu_monitor


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    monitor = get_gpu_monitor()

    # 打印状态
    monitor.print_stats()

    # 检查可用性
    required_memory = 20000  # 20GB
    if monitor.check_availability(required_memory, gpu_count=8):
        print(f"✓ 有足够的 GPU 资源（需要 {gpu_count}x {required_memory//1024}GB）")
    else:
        print(f"✗ GPU 资源不足（需要 {gpu_count}x {required_memory//1024}GB）")

    # 获取最优 Embedding GPU
    optimal_gpu = monitor.get_optimal_gpu_for_embedding()
    print(f"推荐 Embedding 设备: {optimal_gpu}")
