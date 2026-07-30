#!/usr/bin/env python
"""
REINVENT 風格 GRU decoder-only 語言模型 訓練入口點

使用方式:
    python train_reinvent.py --config configs/train_reinvent.yaml
"""

import argparse

from reinvent_lm.training import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='REINVENT 風格 SMILES LM 訓練腳本')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_reinvent.yaml',
        help='訓練配置檔案路徑 (default: configs/train_reinvent.yaml)'
    )
    args = parser.parse_args()

    main(config_path=args.config)
