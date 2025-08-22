#!/bin/bash

# SimCEN 运行脚本
echo "SimCEN 论文代码运行指南"
echo "======================="

# 设置基本参数
CONFIG_DIR="./config/"
GPU_ID=0  # 设置为 -1 使用 CPU

echo "可用的运行选项："
echo "1. 运行 SimCEN (默认 Avazu 数据集)"
echo "2. 运行 SimCEN with KKBox 数据集"
echo "3. 运行其他模型变体"

# 选项1: 默认 SimCEN
echo "运行命令示例："
echo "1. 基本 SimCEN (Avazu 数据集):"
echo "   python run_expid.py --config $CONFIG_DIR --expid SimCEN --gpu $GPU_ID"
echo ""

echo "2. SimCEN with KKBox 数据集:"
echo "   python run_expid.py --config $CONFIG_DIR --expid SimCEN_KKBox --gpu $GPU_ID"
echo ""

echo "3. 其他可用的实验 ID:"
echo "   - SimCEN (主模型)"
echo "   - SimCEN_KKBox (KKBox数据集版本)"
echo ""

echo "参数说明："
echo "  --config: 配置文件目录 (默认: ./config/)"
echo "  --expid:  实验ID (在 model_config.yaml 中定义)"
echo "  --gpu:    GPU编号 (0,1,2... 或 -1 表示CPU)"
echo ""

echo "开始运行? 请选择一个选项 (1-3) 或直接运行对应的命令"
