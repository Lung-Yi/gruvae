#!/bin/bash

# 1. 執行 Python，並把輸出寫入 log
# (這裡不需要 nohup，因為我們會在執行這個腳本時加 nohup)
python train.py > my_log.com

# 2. 執行 Git 推送
git add my_log.com
git commit -m "Auto update: log file after execution"
git push