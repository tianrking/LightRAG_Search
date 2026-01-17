# 快速启动指南

## 一键启动 (推荐) ⭐

```bash
cd /home/tianrui/gotohk/private_ragflow/workspace/LightRAG_Project

# 安装依赖
pip install -r requirements.txt

# 一键启动（自动启动 vLLM + 索引 + 测试）
./scripts/start_all.sh
```

## 分步启动

### 步骤 1: 安装依赖
```bash
cd /home/tianrui/gotohk/private_ragflow/workspace/LightRAG_Project
pip install -r requirements.txt
```

### 步骤 2: 启动 vLLM 服务 ⭐ **必需**

vLLM 是 LLM 推理服务，**必须先启动**才能进行索引和查询。

```bash
# 终端 1 - 启动 8x4090 vLLM 服务（端口 38199）
./scripts/start_vllm.sh


Qwen2.5-32B-Instruct

# 或手动启动
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --host 0.0.0.0 \
  --port 38199 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768


vllm serve Qwen/Qwen2.5-32B-Instruct \
  --host 0.0.0.0 \
  --port 38199 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768

vllm serve Qwen2.5-0.5B \
  --host 0.0.0.0 \
  --port 38199 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.50 \
  --max-model-len 

vllm serve /home/tianrui/gotohk/private_ragflow/Qwen2.5-3B \
  --host 0.0.0.0 \
  --port 38199 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.50 \
  --max-model-len 32768

modelscope download --model Qwen/Qwen2.5-3B --local_dir ./Qwen2.5-3B

```

**验证 vLLM 服务**（新终端）：
```bash
# 检查服务是否运行
curl http://localhost:38199/v1/models

# 查看 GPU 使用情况
nvidia-smi
```

### 步骤 3: 并发索引 HKIPO 文档
```bash
# 终端 2 - 并发索引 1894 个文件 (4个文件同时处理)
python main.py index \
  --input /home/tianrui/gotohk/private_ragflow/workspace/HKIPO_resource/HKIPO-中介公共资料 \
  --concurrent \
  --workers 4
```

**新特性**:
- ✓ **并发处理**: 4个文件同时处理，速度提升4倍
- ✓ **断点续传**: 中断后重新运行会自动跳过已处理的文件
- ✓ **文件去重**: 自动检测并跳过重复文件（基于MD5哈希）
- ✓ **实时进度**: 显示当前进度和处理的文件名
- ✓ **失败重试**: 支持单独重试失败的文件

### 步骤 4: 查询文档
```bash
# 单次查询
python main.py query --query "香港专利申请流程" --mode hybrid

# 或进入交互模式
python main.py interactive
```

## 并发索引详解

### 基本用法

```bash
# 默认并发索引（4个文件同时处理）
python main.py index --concurrent

# 自定义并发数（6个文件同时处理）
python main.py index --concurrent --workers 6

# 只索引当前目录（不递归）
python main.py index --no-recursive
```

### 断点续传

如果索引过程中中断（Ctrl+C 或错误），重新运行相同命令会自动：

1. **跳过已索引的文件** - 基于文件修改时间和哈希
2. **跳过重复文件** - 相同内容的文件只索引一次
3. **继续处理剩余文件** - 从中断处继续

```bash
# 第一次运行（处理 1000 个文件后中断）
python main.py index --concurrent --workers 4

# 第二次运行（只处理剩余的 894 个文件）
python main.py index --concurrent --workers 4
```

### 失败重试

如果部分文件索引失败，可以单独重试：

```bash
# 重试所有失败的文件
python main.py index --retry-failed
```

### 进度显示

实时显示处理进度：

```
[进度: 45.2%] 已索引: 856 | 失败: 3 | 跳过: 120 | 剩余: 915 | 当前: 尽调资料/公司A.pdf
```

进度数据保存在 `data/index_progress/` 目录：
- `index_progress.json` - 索引进度
- `file_records.json` - 文件记录（哈希、状态等）

### 去重机制

自动检测重复文件（基于 MD5 哈希）：

```
找到文件: 尽调资料/公司A.pdf (新文件)
找到文件: 招股书验证/公司A_副本.pdf (跳过: 与 尽调资料/公司A.pdf 重复)
```

## 性能对比

| 模式 | 并发数 | 1894文件预计耗时 |
|------|--------|------------------|
| 顺序模式 | 1 | ~60 分钟 |
| 并发模式 | 4 | ~15 分钟 |
| 并发模式 | 8 | ~8 分钟 |

## 常用命令

### GPU 监控
```bash
# 查看 GPU 状态
python -c "from src.gpu_monitor import get_gpu_monitor; get_gpu_monitor().print_stats()"
```

### 查看索引进度
```bash
# 查看进度文件
cat data/index_progress/index_progress.json

# 查看文件记录
cat data/index_progress/file_records.json
```

### 查看日志
```bash
# vLLM 日志
tail -f logs/vllm_server.log

# LightRAG 日志
tail -f logs/lightrag.log
```

## 故障排查

### vLLM 启动失败？
```bash
# 检查 GPU
nvidia-smi

# 减少并行度
# 编辑 scripts/start_vllm.sh，修改 TENSOR_PARALLEL_SIZE=4
```

### 索引太慢？
```bash
# 增加并发数（注意GPU显存）
python main.py index --concurrent --workers 8
```

### 内存不足？
```bash
# 减少并发数
python main.py index --concurrent --workers 2
```

### 部分文件失败？
```bash
# 查看失败文件
cat data/index_progress/file_records.json | grep '"error":'

# 重试失败的文件
python main.py index --retry-failed
```

## HKIPO 数据统计

```
总文件数:     1,894
├── PDF:      ~1,700
├── DOCX:     ~180
└── TXT:      ~14

预计索引时间: ~15 分钟 (4并发)
```

## 更多信息

- [完整部署指南](docs/DEPLOYMENT.md)
- [最佳实践](docs/BEST_PRACTICES.md)
- [主 README](README.md)
