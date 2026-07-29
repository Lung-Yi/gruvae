#!/bin/bash

# 執行訓練，並把輸出寫入 log
# (這裡不需要 nohup，因為我們會在執行這個腳本時加 nohup)
python train.py --config configs/train.yaml > outputs/logs/my_log.com
