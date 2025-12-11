#!/usr/bin/env python3
"""
訓練腳本入口點
使用方式: python train.py --config train.yaml
"""

import sys
import os

# 將 src 目錄添加到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 導入並執行主訓練腳本
if __name__ == "__main__":
    import argparse

    # 直接從 src.train 模組導入 main 函數
    import train as train_module

    parser = argparse.ArgumentParser(description='VAE 訓練腳本')
    parser.add_argument(
        '--config',
        type=str,
        default='train.yaml',
        help='訓練配置檔案路徑 (default: train.yaml)'
    )
    args = parser.parse_args()

    train_module.main(config_path=args.config)
