# 專案結構說明

本專案已經重構為模組化結構，所有源代碼位於 `src/` 目錄中。

## 目錄結構

```
gruvae/
├── src/                          # 源代碼目錄
│   ├── __init__.py
│   ├── train.py                  # 訓練腳本主程序
│   ├── models/                   # 模型模組
│   │   ├── __init__.py
│   │   ├── gru_vae.py           # GRU-VAE 模型
│   │   └── transformer_vae.py   # Transformer-VAE 模型
│   ├── data/                     # 數據處理模組
│   │   ├── __init__.py
│   │   ├── tokenizer.py         # SMILES 分詞器
│   │   ├── dataset.py           # 數據集類
│   │   └── process_smiles.py    # SMILES 預處理腳本
│   └── utils/                    # 工具模組
│       ├── __init__.py
│       └── molecule_generator.py # 分子生成器
├── train.py                      # 訓練入口腳本（包裝器）
├── train.yaml                    # 訓練配置文件
├── data/                         # 數據目錄
│   ├── train_processed.csv
│   └── train_small_processed.csv
├── checkpoints*/                 # 模型檢查點目錄
├── README.md                     # 專案說明
├── CONFIG_GUIDE.md              # 配置指南
├── GENERATOR_GUIDE.md           # 生成器使用指南
└── CHANGELOG.md                 # 更新日誌
```

## 使用方法

### 1. 訓練模型

從專案根目錄運行：

```bash
python train.py --config train.yaml
```

### 2. 模組導入

如果需要在其他腳本中導入模組：

```python
import sys
sys.path.insert(0, 'src')

# 導入模型
from models import GRUVAE, TransformerVAE

# 導入數據處理
from data import SmilesTokenizer, SmilesVAEDataset

# 導入工具
from utils.molecule_generator import VAEMoleculeGenerator
```

### 3. 模型選擇

在 `train.yaml` 中設置 `model_type` 來選擇模型：

#### 使用 GRU-VAE：
```yaml
model:
  model_type: 'gru'
  embedding_dim: 256
  hidden_dim: 512
  latent_dim: 128
  num_layers: 3
  dropout: 0.1
  bidirectional: true
```

#### 使用 Transformer-VAE：
```yaml
model:
  model_type: 'transformer'
  d_model: 512
  nhead: 8
  num_encoder_layers: 8
  num_decoder_layers: 6
  dim_feedforward: 2048
  latent_dim: 128
  dropout: 0.1
  max_len: 128
```

## 模組說明

### models/ - 模型模組

- `gru_vae.py`: 包含 GRU-based VAE 的實現
  - `GRUVAE`: 完整的 GRU-VAE 模型
  - `GRUEncoder`: GRU 編碼器
  - `GRUDecoder`: GRU 解碼器
  - `compute_loss`: VAE 損失函數

- `transformer_vae.py`: 包含 Transformer-based VAE 的實現
  - `TransformerVAE`: 完整的 Transformer-VAE 模型
  - `PositionalEncoding`: 位置編碼層

### data/ - 數據處理模組

- `tokenizer.py`: SMILES 分詞器
  - `SmilesTokenizer`: 字符級分詞器
  - `canonicalize_smiles`: SMILES 規範化
  - `randomize_smiles`: SMILES 隨機化

- `dataset.py`: PyTorch 數據集
  - `SmilesVAEDataset`: VAE 訓練數據集
  - `collate_fn`: 批次數據整理函數
  - `get_dataloader`: DataLoader 創建函數

- `process_smiles.py`: 數據預處理腳本

### utils/ - 工具模組

- `molecule_generator.py`: 分子生成工具
  - `VAEMoleculeGenerator`: VAE 分子生成器類

## Teacher Forcing 功能

兩種模型都支援 teacher forcing 控制：

- **訓練階段**: `teacher_forcing=True` - 使用真實目標序列加速訓練
- **驗證階段**: `teacher_forcing=False` - 自回歸生成，公平評估模型性能

訓練腳本會自動處理這個切換，無需手動配置。

## 測試

### 測試 TransformerVAE
```bash
cd src
python models/transformer_vae.py
```

### 測試 GRU-VAE
```bash
cd src
python models/gru_vae.py
```

## 注意事項

1. 所有導入都使用相對導入（在 src 內部）或絕對導入（從外部）
2. 配置文件路徑相對於專案根目錄
3. 數據文件路徑相對於專案根目錄
4. 檢查點保存在配置文件指定的目錄

## 遷移說明

從舊版本遷移：

1. ✓ 所有 `.py` 文件已移至 `src/` 目錄並重新組織
2. ✓ 導入路徑已更新
3. ✓ 配置文件路徑保持不變
4. ✓ 舊的檢查點仍然兼容
5. ✓ 使用方法保持一致（仍然是 `python train.py`）

舊的原始文件仍然保留在根目錄，但建議使用新的 `src/` 結構。
