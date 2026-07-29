#!/usr/bin/env python
"""
GRU/Transformer VAE 訓練入口點

使用方式:
    python train.py --config configs/train.yaml
"""

import argparse

from gruvae.training import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GRU-VAE 訓練腳本')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train.yaml',
        help='訓練配置檔案路徑 (default: configs/train.yaml)'
    )
    args = parser.parse_args()

    main(config_path=args.config)
