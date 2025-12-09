# GRU-VAE Training Updates

## 最新功能 (Latest Features)

### 1. **Reconstruction Accuracy (重建準確度)**
在驗證階段計算模型重建分子的準確度，使用規範化 SMILES 進行比較。

**實作細節:**
- 在 `validate()` 方法中新增準確度計算
- 將預測和目標 SMILES 都轉換為規範化形式 (canonical form)
- 比較規範化後的 SMILES 是否完全匹配
- 計算匹配百分比作為重建準確度

**程式碼位置:** `train.py:193-203`

```python
for i in range(encoder_input.size(0)):
    target_smiles = self.tokenizer.decode(decoder_target[i].cpu().tolist())
    pred_smiles = self.tokenizer.decode(predictions[i].cpu().tolist())

    # Canonicalize 兩者進行比較
    target_canonical = canonicalize_smiles(target_smiles)
    pred_canonical = canonicalize_smiles(pred_smiles)

    if target_canonical == pred_canonical:
        total_correct += 1
    total_samples += 1
```

### 2. **Learning Rate 和 KL Weight 顯示**
每個 epoch 結束時顯示當前的學習率和 KL weight。

**實作細節:**
- 在訓練歷史中追蹤 `kl_weight` 和 `lr`
- 在每個 epoch 的摘要中顯示這些數值

**程式碼位置:** `train.py:288-306, 312-313`

```python
# 獲取當前學習率
current_lr = self.optimizer.param_groups[0]['lr']

# 記錄 KL weight 和 learning rate
self.history['kl_weight'].append(self.current_kl_weight)
self.history['lr'].append(current_lr)

# 輸出統計
print(f"  Learning Rate:     {current_lr:.6f}")
print(f"  KL Weight:         {self.current_kl_weight:.4f}")
```

### 3. **KL Weight Annealing (KL 權重退火)**
KL weight 從較小的值開始，在訓練過程中線性增加，以避免後驗坍塌 (posterior collapse)。

**實作細節:**
- 新增三個參數: `kl_weight_start`, `kl_weight_end`, `kl_anneal_epochs`
- 實作 `get_kl_weight()` 方法進行線性退火
- 在每個 epoch 開始時更新 `current_kl_weight`

**程式碼位置:** `train.py:32-34, 46-50, 70-78, 286`

```python
def get_kl_weight(self, epoch: int) -> float:
    """計算當前 epoch 的 KL weight（線性 annealing）"""
    if epoch > self.kl_anneal_epochs:
        return self.kl_weight_end

    # 線性增長
    progress = epoch / self.kl_anneal_epochs
    kl_weight = self.kl_weight_start + (self.kl_weight_end - self.kl_weight_start) * progress
    return kl_weight
```

**預設配置:**
```python
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    lr=1e-3,
    kl_weight_start=0.0,    # KL weight 從 0 開始
    kl_weight_end=0.1,       # 逐漸增加到 0.1
    kl_anneal_epochs=10,     # 在 10 個 epochs 內線性增加
    save_dir='./checkpoints'
)
```

## 訓練輸出範例

```
開始訓練 20 epochs...
訓練集大小: 34111
驗證集大小: 3791
設備: cuda
初始學習率: 0.001
KL Weight 範圍: 0.0 -> 0.1 (over 10 epochs)

================================================================================
Epoch 1 Summary:
================================================================================
  Learning Rate:     0.001000
  KL Weight:         0.0000
  Train - Loss: X.XXXX, Recon: X.XXXX, KL: X.XXXX
  Val   - Loss: X.XXXX, Recon: X.XXXX, KL: X.XXXX
  Val Recon Acc:     XX.XX% (XXXX/3791)
```

## KL Annealing 的優點

1. **避免後驗坍塌**: KL weight 從 0 開始，讓模型先專注於重建損失
2. **穩定訓練**: 逐漸增加 KL 項的影響，避免訓練初期的不穩定
3. **更好的潛在表示**: 模型有更多時間學習有意義的潛在空間表示

## 模型配置

- **詞彙表大小**: 26 tokens (包含 4 個特殊 tokens)
- **Embedding 維度**: 256
- **隱藏層維度**: 512
- **潛在空間維度**: 128
- **GRU 層數**: 2
- **Dropout**: 0.1
- **Batch Size**: 64
- **學習率**: 0.001
- **KL Annealing**: 0.0 → 0.1 (over 10 epochs)
- **訓練 Epochs**: 20

## 訓練技巧

1. **Teacher Forcing**: 訓練時使用，驗證時不使用
2. **梯度裁剪**: max_norm=1.0
3. **資料增強**: 輸入使用隨機化 SMILES，輸出使用規範化 SMILES
4. **Early Stopping**: 保存驗證損失最低的模型

## 檔案修改

- ✅ `train.py`: 新增所有三個功能
- ✅ `README.md`: 已更新文檔（之前的版本）
- ✅ `model.py`: 已實作（無修改）
- ✅ `dataset.py`: 已實作（無修改）
- ✅ `tokenizer.py`: 已實作（無修改）
