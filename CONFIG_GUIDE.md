# 配置檔案使用指南

## 概述

GRU-VAE 訓練系統現在支援使用 YAML 配置檔案來管理所有超參數和訓練設定。這使得實驗配置更加靈活和可追溯。

## 快速開始

### 1. 使用預設配置訓練

```bash
python train.py
```

這會自動使用 `train.yaml` 作為配置檔案。

### 2. 使用自訂配置檔案

```bash
python train.py --config my_config.yaml
```

## 配置檔案結構

### 完整配置範例 (`train.yaml`)

```yaml
# GRU-VAE 訓練配置檔案

# 資料設定
data:
  train_csv: './data/train_processed.csv'
  train_split: 0.9  # 訓練集比例
  max_length: 100   # 最大序列長度

# 模型架構
model:
  embedding_dim: 256
  hidden_dim: 512
  latent_dim: 128
  num_layers: 2
  dropout: 0.1

# 訓練參數
training:
  num_epochs: 20
  batch_size: 64
  learning_rate: 0.001
  num_workers: 0

  # KL Annealing 設定
  kl_weight_start: 0.0
  kl_weight_end: 0.1
  kl_anneal_epochs: 10

  # 梯度裁剪
  grad_clip_max_norm: 1.0

# 驗證設定
validation:
  num_reconstruct: 10  # 顯示多少個重建範例
  num_sample: 10       # 顯示多少個採樣分子

# 檢查點設定
checkpoint:
  save_dir: './checkpoints'
  save_interval: 5     # 每幾個 epoch 保存一次

# 設備設定
device:
  use_cuda: true       # 如果可用，使用 CUDA

# 隨機種子
seed: 42
```

## 配置參數詳解

### 資料設定 (data)

| 參數 | 類型 | 說明 | 預設值 |
|------|------|------|--------|
| `train_csv` | string | 訓練資料 CSV 檔案路徑 | `'./data/train_processed.csv'` |
| `train_split` | float | 訓練集比例 (0-1) | `0.9` |
| `max_length` | int | SMILES 最大序列長度 | `100` |

### 模型架構 (model)

| 參數 | 類型 | 說明 | 預設值 |
|------|------|------|--------|
| `embedding_dim` | int | Embedding 層維度 | `256` |
| `hidden_dim` | int | GRU 隱藏層維度 | `512` |
| `latent_dim` | int | 潛在空間維度 | `128` |
| `num_layers` | int | GRU 層數 | `2` |
| `dropout` | float | Dropout 比率 | `0.1` |

**建議設定:**
- 小模型: `embedding_dim=128, hidden_dim=256, latent_dim=64`
- 中模型: `embedding_dim=256, hidden_dim=512, latent_dim=128` (預設)
- 大模型: `embedding_dim=512, hidden_dim=1024, latent_dim=256`

### 訓練參數 (training)

| 參數 | 類型 | 說明 | 預設值 |
|------|------|------|--------|
| `num_epochs` | int | 訓練輪數 | `20` |
| `batch_size` | int | Batch 大小 | `64` |
| `learning_rate` | float | 學習率 | `0.001` |
| `num_workers` | int | DataLoader 工作進程數 | `0` |
| `kl_weight_start` | float | KL weight 起始值 | `0.0` |
| `kl_weight_end` | float | KL weight 結束值 | `0.1` |
| `kl_anneal_epochs` | int | KL annealing 持續輪數 | `10` |
| `grad_clip_max_norm` | float | 梯度裁剪最大範數 | `1.0` |

**KL Annealing 建議:**
- 避免後驗坍塌: 從 `0.0` 開始，逐漸增加到 `0.1` 或更小
- 快速收斂: 從 `0.05` 開始，增加到 `0.3`
- 標準設定: `start=0.0, end=0.1, epochs=10`

### 驗證設定 (validation)

| 參數 | 類型 | 說明 | 預設值 |
|------|------|------|--------|
| `num_reconstruct` | int | 顯示重建範例數 | `10` |
| `num_sample` | int | 顯示採樣分子數 | `10` |

### 檢查點設定 (checkpoint)

| 參數 | 類型 | 說明 | 預設值 |
|------|------|------|--------|
| `save_dir` | string | 檢查點保存目錄 | `'./checkpoints'` |
| `save_interval` | int | 定期保存間隔 (epochs) | `5` |

### 設備設定 (device)

| 參數 | 類型 | 說明 | 預設值 |
|------|------|------|--------|
| `use_cuda` | bool | 是否使用 CUDA | `true` |

### 隨機種子 (seed)

| 參數 | 類型 | 說明 | 預設值 |
|------|------|------|--------|
| `seed` | int | 隨機種子 | `42` |

## 常見配置範例

### 1. 快速實驗配置 (small_model.yaml)

```yaml
data:
  train_csv: './data/train_processed.csv'
  train_split: 0.9
  max_length: 80

model:
  embedding_dim: 128
  hidden_dim: 256
  latent_dim: 64
  num_layers: 1
  dropout: 0.1

training:
  num_epochs: 10
  batch_size: 128
  learning_rate: 0.002
  num_workers: 0
  kl_weight_start: 0.0
  kl_weight_end: 0.05
  kl_anneal_epochs: 5
  grad_clip_max_norm: 1.0

validation:
  num_reconstruct: 5
  num_sample: 5

checkpoint:
  save_dir: './checkpoints/small'
  save_interval: 3

device:
  use_cuda: true

seed: 42
```

使用方式:
```bash
python train.py --config small_model.yaml
```

### 2. 大型模型配置 (large_model.yaml)

```yaml
data:
  train_csv: './data/train_processed.csv'
  train_split: 0.95
  max_length: 150

model:
  embedding_dim: 512
  hidden_dim: 1024
  latent_dim: 256
  num_layers: 3
  dropout: 0.2

training:
  num_epochs: 50
  batch_size: 32
  learning_rate: 0.0005
  num_workers: 4
  kl_weight_start: 0.0
  kl_weight_end: 0.1
  kl_anneal_epochs: 20
  grad_clip_max_norm: 2.0

validation:
  num_reconstruct: 20
  num_sample: 20

checkpoint:
  save_dir: './checkpoints/large'
  save_interval: 10

device:
  use_cuda: true

seed: 42
```

### 3. 無 KL Annealing 配置

```yaml
training:
  kl_weight_start: 0.1
  kl_weight_end: 0.1
  kl_anneal_epochs: 1
  # ... 其他參數
```

## 最佳實踐

### 1. 版本控制

將不同的實驗配置保存為不同的 YAML 檔案:

```
configs/
├── baseline.yaml
├── experiment_01_high_kl.yaml
├── experiment_02_large_latent.yaml
└── experiment_03_deep_model.yaml
```

### 2. 命名規範

建議使用描述性的檔案名稱:
- `baseline.yaml` - 基準配置
- `exp_high_kl.yaml` - 高 KL weight 實驗
- `exp_large_batch.yaml` - 大 batch size 實驗

### 3. 記錄實驗

在配置檔案頂部添加註釋:

```yaml
# 實驗: 測試較大的潛在空間維度
# 日期: 2024-12-09
# 目的: 檢查 latent_dim=256 是否能改善重建品質
# 結果: (待填寫)

model:
  latent_dim: 256
  # ...
```

### 4. GPU 記憶體管理

如果遇到 GPU 記憶體不足:

```yaml
training:
  batch_size: 32  # 減小 batch size
  num_workers: 0   # 減少 worker 數量

model:
  hidden_dim: 256  # 減小模型大小
```

## 故障排除

### 問題 1: 配置檔案找不到

**錯誤訊息:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'my_config.yaml'
```

**解決方法:**
確保配置檔案在當前目錄，或提供完整路徑:
```bash
python train.py --config /path/to/my_config.yaml
```

### 問題 2: YAML 語法錯誤

**錯誤訊息:**
```
yaml.scanner.ScannerError: mapping values are not allowed here
```

**解決方法:**
檢查 YAML 語法:
- 確保冒號後有空格: `key: value` (正確) vs `key:value` (錯誤)
- 檢查縮排一致性 (使用空格，不要用 Tab)

### 問題 3: 參數值錯誤

**錯誤訊息:**
```
KeyError: 'learning_rate'
```

**解決方法:**
確保配置檔案包含所有必要的參數。參考 `train.yaml` 模板。

## 進階用法

### 1. 結合命令列參數

未來可以擴展為支援命令列覆蓋配置:

```bash
# 計劃功能 (未實作)
python train.py --config train.yaml --lr 0.0005 --batch_size 128
```

### 2. 配置繼承

可以創建基礎配置和特定實驗配置:

```yaml
# base_config.yaml
defaults: &defaults
  seed: 42
  device:
    use_cuda: true

# experiment_config.yaml
<<: *defaults
model:
  latent_dim: 256
```

## 總結

使用配置檔案的優點:
- ✅ 易於管理不同實驗
- ✅ 可追溯的實驗設定
- ✅ 避免修改代碼
- ✅ 便於分享和重現實驗
- ✅ 支援版本控制

建議流程:
1. 從 `train.yaml` 開始
2. 複製並修改為自己的配置
3. 使用 `--config` 參數指定配置
4. 記錄實驗結果
5. 迭代優化
