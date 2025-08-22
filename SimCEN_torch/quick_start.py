#!/usr/bin/env python3
"""
SimCEN 快速启动脚本
用法: python quick_start.py [选项]
"""

import os
import sys
import argparse
import subprocess

def main():
    print("=" * 50)
    print("SimCEN 论文代码快速启动")
    print("=" * 50)

    parser = argparse.ArgumentParser(description='SimCEN 快速启动脚本')
    parser.add_argument('--model', type=str, default='SimCEN',
                        choices=['SimCEN', 'SimCEN_KKBox'],
                        help='选择模型配置')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU编号 (-1表示使用CPU)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='训练轮数 (用于快速测试)')

    args = parser.parse_args()

    print(f"配置信息:")
    print(f"- 模型: {args.model}")
    print(f"- GPU: {args.gpu}")
    print(f"- 训练轮数: {args.epochs}")
    print("")

    # 构建运行命令
    cmd = [
        'python', 'run_expid.py',
        '--config', './config/',
        '--expid', args.model,
        '--gpu', str(args.gpu)
    ]

    print("执行命令:")
    print(" ".join(cmd))
    print("")

    try:
        # 执行命令
        result = subprocess.run(cmd, cwd='.', check=True)
        print("运行成功!")
    except subprocess.CalledProcessError as e:
        print(f"运行失败: {e}")
        return 1
    except FileNotFoundError:
        print("错误: 找不到 run_expid.py 文件")
        print("请确保在 SimCEN_torch 目录下运行此脚本")
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
