# GRU-VAE for SMILES Generation

基於 GRU 的變分自編碼器（Variational Autoencoder, VAE），用於學習和生成 SMILES 分子表示。

## 專案簡介

本專案實現了一個用於分子生成的 GRU-VAE 模型，能夠：
- 學習 SMILES 分子的潛在表示（Latent Representation）
- 重建輸入的 SMILES 分子
- 從潛在空間採樣生成新的分子結構

## 主要特性

- ✅ **GRU 架構**：使用 GRU（Gated Recurrent Unit）作為編碼器和解碼器
- ✅ **Teacher Forcing**：訓練時使用 teacher forcing 加速收斂
- ✅ **資料增強**：輸入使用隨機化 SMILES，輸出使用規範化 SMILES
- ✅ **完整的驗證**：顯示分子重建和潛在空間採樣結果
- ✅ **模組化設計**：易於擴展和修改

## 專案結構

```
gruvae/
├── README.md                   # 專案說明文檔
├── tokenizer.py               # SMILES 分詞器
├── dataset.py                 # Dataset 和 DataLoader
├── model.py                   # GRU-VAE 模型架構
├── train.py                   # 訓練腳本
├── process_smiles.py          # 資料預處理腳本
├── data/
│   ├── train.txt              # 原始訓練資料
│   └── train_processed.csv    # 處理後的資料
└── checkpoints/
    ├── tokenizer.json         # 保存的 tokenizer
    └── best_model.pt          # 最佳模型檢查點
```

## 環境需求

```bash
python >= 3.7
torch >= 1.9.0
rdkit >= 2021.09.1
pandas >= 1.3.0
numpy >= 1.21.0
tqdm >= 4.62.0
```

## 安裝

1. 克隆專案

```bash
git clone <repository-url>
cd gruvae
```

2. 安裝依賴套件

```bash
pip install torch rdkit pandas numpy tqdm
```

或使用 conda：

```bash
conda install pytorch rdkit pandas numpy tqdm -c pytorch -c conda-forge
```

## 快速開始

### 1. 資料預處理

將原始 SMILES 資料處理為訓練格式：

```bash
python process_smiles.py
```

這個腳本會：
- 讀取 `./data/train.txt` 中的 SMILES
- 移除手性和順反異構信息
- 過濾掉重原子數 > 20 的分子
- 輸出到 `./data/train_processed.csv`

### 2. 訓練模型

```bash
python train.py
```

訓練過程中會：
- 自動建立並保存 tokenizer
- 每個 epoch 結束後進行驗證
- 顯示 10 個分子重建結果
- 顯示 10 個從潛在空間採樣的新分子
- 自動保存最佳模型到 `./checkpoints/best_model.pt`

### 3. 監控訓練

訓練日誌會實時顯示：
- 訓練/驗證的 Loss、重建 Loss 和 KL Loss
- 分子重建的匹配結果（✓ 或 ✗）
- 從潛在空間採樣的新分子

## 模型架構

### 整體架構

```
Input SMILES (Randomized)
        ↓
    Embedding
        ↓
   GRU Encoder
        ↓
    μ, log(σ²)  ← Latent Distribution
        ↓
 Reparameterization (z = μ + σ * ε)
        ↓
        z (Latent Vector)
        ↓
   GRU Decoder (with Teacher Forcing)
        ↓
    Output SMILES (Canonical)
```

### GRU Encoder

- **輸入**: Tokenized SMILES 序列
- **結構**:
  - Embedding Layer (vocab_size → embedding_dim=256)
  - GRU Layers (2 層, hidden_dim=512)
  - Linear Layers (hidden_dim → latent_dim=128) for μ and log(σ²)
- **輸出**: 潛在分布參數 (μ, log σ²)

### GRU Decoder

- **輸入**: 潛在向量 z + 目標序列（teacher forcing）
- **結構**:
  - Linear Layer (latent_dim → hidden_dim * num_layers)
  - Embedding Layer
  - GRU Layers (2 層, hidden_dim=512)
  - Output Linear Layer (hidden_dim → vocab_size)
- **輸出**: 重建的 SMILES 序列

### 重參數化技巧

使用重參數化技巧使梯度能夠反向傳播：

```
z = μ + σ * ε, where ε ~ N(0, I)
```

### Loss 函數

```
Total Loss = Reconstruction Loss + β * KL Divergence

Reconstruction Loss = CrossEntropy(predicted, target)
KL Divergence = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
```

其中 β = 0.1（KL weight，避免後驗坍塌）

## 訓練配置

預設超參數：

```python
vocab_size = 26              # 根據資料集自動計算
embedding_dim = 256          # Embedding 維度
hidden_dim = 512             # GRU 隱藏層維度
latent_dim = 128             # 潛在空間維度
num_layers = 2               # GRU 層數
dropout = 0.1                # Dropout 比率
batch_size = 64              # Batch 大小
learning_rate = 1e-3         # 學習率
kl_weight = 0.1              # KL 散度權重
num_epochs = 10              # 訓練輪數
max_length = 100             # 最大序列長度
```

## 模型使用

### 載入訓練好的模型

```python
import torch
from model import GRUVAE
from tokenizer import SmilesTokenizer

# 載入 tokenizer
tokenizer = SmilesTokenizer()
tokenizer.load('./checkpoints/tokenizer.json')

# 建立模型
model = GRUVAE(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=256,
    hidden_dim=512,
    latent_dim=128,
    num_layers=2
)

# 載入權重
checkpoint = torch.load('./checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### 編碼 SMILES

```python
# 將 SMILES 編碼為潛在向量
smiles = "CCO"
indices = tokenizer.encode(smiles)
input_tensor = torch.tensor([indices])

with torch.no_grad():
    mu, logvar = model.encoder(input_tensor)
    z = model.reparameterize(mu, logvar)

print(f"Latent vector shape: {z.shape}")  # [1, 128]
```

### 從潛在空間採樣生成新分子

```python
# 從潛在空間採樣生成新分子
num_samples = 10
max_length = 100

samples = model.sample(
    num_samples=num_samples,
    max_length=max_length,
    start_idx=tokenizer.start_idx,
    device='cpu'
)

# 解碼為 SMILES
for i, sample in enumerate(samples):
    smiles = tokenizer.decode(sample.tolist())
    print(f"Sample {i+1}: {smiles}")
```

### 重建分子

```python
# 重建輸入的 SMILES
smiles = "c1ccccc1"  # 苯
encoder_input = torch.tensor([tokenizer.encode(smiles)])
decoder_input = encoder_input.clone()

with torch.no_grad():
    output, mu, logvar = model(
        encoder_input,
        decoder_input,
        teacher_forcing=False
    )

    # 取最大概率的 token
    predicted = output.argmax(dim=-1)
    reconstructed = tokenizer.decode(predicted[0].tolist())

print(f"Original:      {smiles}")
print(f"Reconstructed: {reconstructed}")
```

## 訓練結果示例

### Epoch 3 驗證結果

**重建示例**:
```
[1] Target:        CCn1ncc(CNc2ccc(F)cc2F)c1C
    Reconstructed: CCn1cc(CNc2ccc(F)cc2F)c(C)n1
    Match: ✗

[2] Target:        O=C(COc1ccc(Cl)cc1)Nc1ccc(O)cc1
    Reconstructed: O=C(COc1ccc(Cl)cc1)Nc1ccc(O)cc1
    Match: ✓
```

**從潛在空間採樣的新分子**:
```
[1] Cc1cc(C(=O)NCc2cccs2)no1
[2] COC(=O)Nc1ccc(C(=O)N2CCOCC2)cc1
[3] COc1ccc(-c2nc(C)c(C(=O)NC(C)C)o2)cc1
[4] CC(C)Oc1ccc(C(=O)Nc2ccc(C)o2)cc1
[5] COc1ccc(C(=O)Nc2nnc(C3CC3)s2)cc1
```

**訓練統計**:
```
Epoch 3 Summary:
  Train - Loss: 0.3475, Recon: 0.3068, KL: 0.4065
  Val   - Loss: 5.3381, Recon: 5.2963, KL: 0.4186
```

## 模型參數

預設模型配置下的參數統計：
- **總參數量**: 5,807,386
- **詞彙表大小**: 26 tokens（包含 4 個特殊 tokens）
- **訓練集大小**: 34,111 分子
- **驗證集大小**: 3,791 分子

## 資料格式

### 輸入格式 (train.txt)

每行一個 SMILES 字串：
```
CCO
c1ccccc1
CC(=O)OC1=CC=CC=C1C(=O)O
...
```

### 處理後格式 (train_processed.csv)

包含兩個欄位：
```csv
smiles,heavy_atoms
CCO,3
c1ccccc1,6
CC(=O)O,4
...
```

## 進階使用

### 修改模型架構

編輯 `model.py` 中的模型類別：

```python
# 增加潛在空間維度
model = GRUVAE(
    vocab_size=tokenizer.vocab_size,
    latent_dim=256,  # 從 128 改為 256
    ...
)
```

### 調整訓練參數

編輯 `train.py` 中的訓練配置：

```python
# 調整 KL 權重
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    ...
    kl_weight=0.05,  # 從 0.1 改為 0.05
)
```

### 使用 GPU 訓練

模型會自動檢測並使用可用的 GPU：

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

若要指定特定 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

## 常見問題

### Q: 驗證 Loss 比訓練 Loss 高很多？

A: 這是正常的，因為驗證時不使用 teacher forcing，模型需要完全依靠自己生成的序列，難度更高。

### Q: 如何避免後驗坍塌（Posterior Collapse）？

A: 本專案使用較小的 KL weight (0.1) 來避免後驗坍塌。您也可以：
- 進一步降低 KL weight
- 使用 KL annealing（逐漸增加 KL weight）
- 使用 free bits 技術

### Q: 生成的分子不合法？

A: 可以：
- 增加訓練時間
- 使用更大的資料集
- 添加分子有效性約束
- 使用束搜尋（Beam Search）代替 Greedy Decoding

### Q: 如何加速訓練？

A: 可以：
- 使用更大的 batch size（如果 GPU 記憶體允許）
- 減少 max_length
- 使用多 GPU 訓練
- 使用混合精度訓練（AMP）

## 參考文獻

1. Gómez-Bombarelli, R., et al. (2018). Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules. ACS Central Science.
2. Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. ICLR.
3. Bowman, S. R., et al. (2015). Generating Sentences from a Continuous Space. CoNLL.

## 授權

MIT License

## 聯繫方式

如有問題或建議，歡迎提交 Issue 或 Pull Request。
