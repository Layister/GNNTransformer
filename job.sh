#!/bin/bash

# 设置 LD_PRELOAD 环境变量
export LD_PRELOAD=/lib64/libgomp.so.1

#SBATCH --job-name=human_DLPFC  # 作业名称，可自定义
#SBATCH --partition=gpu-low  # 指定 GPU 分区，根据超算实际情况修改
#SBATCH --gres=gpu:1  # 申请 1 块 GPU，可按需修改数量
#SBATCH --nodes=1  # 使用 1 个节点
#SBATCH --ntasks=1  # 任务数量为 1
#SBATCH --cpus-per-task=4  # 为每个任务分配 4 个 CPU 核心，可按需调整
#SBATCH --time=00:10:00  # 作业最长运行时间为 10 分钟，可按需修改
#SBATCH --output=log/job_output.log  # 标准输出日志文件
#SBATCH --error=log/job_error.log  # 错误输出日志文件

# 加载 cuda 环境
module load cuda/11.4

# 手动初始化 Conda
source ~/.bashrc

# 激活 Conda 环境
# 根据实际的环境名称修改
conda activate py39tor112

# 打印当前环境信息
echo "Current working directory: $(pwd)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# 运行 Python 文件
# 根据实际的 Python 文件路径和名称修改
python /gs/home/dakezhang/GNNTransformer/Human_DLPFC_histology.py

# 检查 Python 脚本是否成功运行
if [ $? -eq 0 ]; then
    echo "Python script executed successfully."
else
    echo "Python script failed to execute."
fi
