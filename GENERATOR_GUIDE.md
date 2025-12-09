# VAE 分子生成器使用指南

## 概述

`VAEMoleculeGenerator` 是一個便於使用的類別，提供訓練好的 GRU-VAE 模型的推理功能，包括：
- 隨機分子生成
- 分子重建
- 分子插值
- 分子編碼

## 快速開始

### 1. 導入和初始化

```python
from molecule_generator import VAEMoleculeGenerator

# 初始化生成器
generator = VAEMoleculeGenerator(
    config_path='train.yaml',
    tokenizer_path='./checkpoints/tokenizer.json',
    checkpoint_path='./checkpoints/best_model.pt'
)
```

### 2. 基本使用

```python
# 生成新分子
molecules = generator.sample_molecules(num_samples=10)

# 重建分子
results = generator.reconstruct_molecules(['CCO', 'c1ccccc1'])

# 分子插值
interpolated = generator.interpolate_molecules('CCO', 'c1ccccc1', num_steps=5)

# 編碼分子
latent_vectors = generator.encode_molecules(['CCO', 'c1ccccc1'])
```

## API 文檔

### 初始化

```python
VAEMoleculeGenerator(
    config_path: str,
    tokenizer_path: str,
    checkpoint_path: str,
    device: str = None
)
```

**參數:**
- `config_path`: 訓練配置檔案路徑 (train.yaml)
- `tokenizer_path`: tokenizer 檔案路徑 (tokenizer.json)
- `checkpoint_path`: 模型檢查點路徑 (.pt 檔案)
- `device`: 計算設備 ('cuda' 或 'cpu'，預設自動偵測)

**範例:**
```python
generator = VAEMoleculeGenerator(
    config_path='train.yaml',
    tokenizer_path='./checkpoints/tokenizer.json',
    checkpoint_path='./checkpoints/best_model.pt',
    device='cuda'  # 可選，自動偵測
)
```

---

### 1. sample_molecules()

從潛在空間隨機採樣生成新分子。

```python
sample_molecules(num_samples: int) -> List[str]
```

**參數:**
- `num_samples`: 要生成的分子數量

**返回:**
- `List[str]`: 生成的 SMILES 列表

**範例:**
```python
# 生成 10 個新分子
molecules = generator.sample_molecules(10)

for i, smiles in enumerate(molecules, 1):
    print(f"{i}. {smiles}")

# 輸出:
# 1. COc1ccc(C(F)(F)F)cc1
# 2. CC(=O)Nc1ccccc1
# 3. ...
```

**使用場景:**
- 探索化學空間
- 生成候選分子
- 創建分子庫

---

### 2. reconstruct_molecules()

重建給定的 SMILES 分子，並比較重建結果。

```python
reconstruct_molecules(smiles_list: List[str]) -> List[Dict[str, str]]
```

**參數:**
- `smiles_list`: 輸入的 SMILES 列表

**返回:**
- `List[Dict]`: 重建結果列表，每個字典包含:
  - `input`: 原始輸入 SMILES
  - `input_canonical`: 規範化的輸入 SMILES
  - `reconstructed`: 重建的 SMILES
  - `reconstructed_canonical`: 規範化的重建 SMILES
  - `match`: 是否匹配 (True/False)

**範例:**
```python
# 重建分子
test_smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
results = generator.reconstruct_molecules(test_smiles)

for result in results:
    match_symbol = '✓' if result['match'] else '✗'
    print(f"{match_symbol} Input: {result['input']}")
    print(f"  Reconstructed: {result['reconstructed']}")
    print(f"  Match: {result['match']}\n")

# 計算重建準確率
accuracy = sum(r['match'] for r in results) / len(results)
print(f"Reconstruction Accuracy: {accuracy:.2%}")
```

**使用場景:**
- 評估模型重建能力
- 檢查模型是否學會了分子表示
- 驗證特定分子的編碼/解碼品質

---

### 3. interpolate_molecules()

在兩個分子之間進行潛在空間線性插值。

```python
interpolate_molecules(
    smiles1: str,
    smiles2: str,
    num_steps: int
) -> List[str]
```

**參數:**
- `smiles1`: 第一個 SMILES（起點）
- `smiles2`: 第二個 SMILES（終點）
- `num_steps`: 插值步數（包含起點和終點）

**返回:**
- `List[str]`: 插值生成的 SMILES 列表

**範例:**
```python
# 在乙醇和苯之間插值
smiles1 = 'CCO'      # 乙醇
smiles2 = 'c1ccccc1'  # 苯
num_steps = 7

interpolated = generator.interpolate_molecules(smiles1, smiles2, num_steps)

print(f"從 {smiles1} 到 {smiles2}:")
for i, smiles in enumerate(interpolated):
    alpha = i / (num_steps - 1)
    print(f"  α={alpha:.2f}: {smiles}")

# 輸出:
# 從 CCO 到 c1ccccc1:
#   α=0.00: CCO
#   α=0.17: CCOc1ccccc1
#   α=0.33: COc1ccccc1
#   ...
#   α=1.00: c1ccccc1
```

**使用場景:**
- 分子優化（在已知活性分子間尋找中間結構）
- 理解分子結構轉換
- 生成具有漸進性質的分子系列
- 探索化學空間的連續性

**進階應用:**
```python
# 生成更多插值點以平滑過渡
fine_interpolation = generator.interpolate_molecules(
    'CCO',
    'c1ccccc1',
    num_steps=20
)

# 可視化插值路徑
from rdkit import Chem
from rdkit.Chem import Draw

mols = [Chem.MolFromSmiles(s) for s in fine_interpolation if Chem.MolFromSmiles(s)]
img = Draw.MolsToGridImage(mols, molsPerRow=5)
img.save('interpolation.png')
```

---

### 4. encode_molecules()

將 SMILES 列表編碼為潛在空間向量。

```python
encode_molecules(smiles_list: List[str]) -> torch.Tensor
```

**參數:**
- `smiles_list`: 輸入的 SMILES 列表

**返回:**
- `torch.Tensor`: 潛在向量張量 `[num_molecules, latent_dim]`

**範例:**
```python
# 編碼分子
molecules = ['CCO', 'c1ccccc1', 'CC(=O)O']
latent_vectors = generator.encode_molecules(molecules)

print(f"維度: {latent_vectors.shape}")  # 輸出: torch.Size([3, 128])
print(f"第一個分子的潛在向量: {latent_vectors[0, :5]}")

# 計算分子間的距離
import torch
dist = torch.norm(latent_vectors[0] - latent_vectors[1])
print(f"CCO 和 c1ccccc1 的潛在空間距離: {dist:.4f}")
```

**使用場景:**
- 分子相似性分析
- 聚類分析
- 降維可視化（t-SNE, UMAP）
- 構建分子搜索系統
- 機器學習特徵提取

**進階應用 - 分子聚類:**
```python
from sklearn.cluster import KMeans
import numpy as np

# 編碼大量分子
smiles_list = [...]  # 你的分子列表
latent_vectors = generator.encode_molecules(smiles_list)

# 轉換為 numpy 並進行聚類
latent_np = latent_vectors.cpu().numpy()
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(latent_np)

# 查看每個聚類的分子
for cluster_id in range(5):
    print(f"\nCluster {cluster_id}:")
    cluster_molecules = [smiles_list[i] for i in range(len(smiles_list)) if clusters[i] == cluster_id]
    for mol in cluster_molecules[:5]:  # 顯示前 5 個
        print(f"  - {mol}")
```

**進階應用 - 分子相似性搜索:**
```python
# 找出與查詢分子最相似的分子
query_smiles = 'CCO'
database_smiles = ['c1ccccc1', 'CCCC', 'CC(=O)O', 'CCN', ...]

# 編碼
query_vec = generator.encode_molecules([query_smiles])
db_vecs = generator.encode_molecules(database_smiles)

# 計算相似度（餘弦相似度或歐氏距離）
import torch.nn.functional as F

# 使用餘弦相似度
similarities = F.cosine_similarity(query_vec, db_vecs)

# 找出最相似的 top-k 個分子
top_k = 5
top_indices = similarities.argsort(descending=True)[:top_k]

print(f"與 {query_smiles} 最相似的分子:")
for i, idx in enumerate(top_indices, 1):
    print(f"{i}. {database_smiles[idx]} (similarity: {similarities[idx]:.4f})")
```

---

## 完整使用範例

### 範例 1: 生成和篩選分子

```python
from molecule_generator import VAEMoleculeGenerator
from rdkit import Chem
from rdkit.Chem import Descriptors

# 初始化
generator = VAEMoleculeGenerator(
    config_path='train.yaml',
    tokenizer_path='./checkpoints/tokenizer.json',
    checkpoint_path='./checkpoints/best_model.pt'
)

# 生成 100 個分子
molecules = generator.sample_molecules(100)

# 篩選有效且符合條件的分子
valid_molecules = []
for smiles in molecules:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # 計算分子量
        mw = Descriptors.MolWt(mol)
        # 計算 LogP
        logp = Descriptors.MolLogP(mol)

        # 篩選條件：200 < MW < 500, LogP < 5
        if 200 < mw < 500 and logp < 5:
            valid_molecules.append({
                'smiles': smiles,
                'mw': mw,
                'logp': logp
            })

print(f"生成了 {len(valid_molecules)} 個符合條件的分子:")
for i, mol_info in enumerate(valid_molecules[:10], 1):
    print(f"{i}. {mol_info['smiles']}")
    print(f"   MW: {mol_info['mw']:.2f}, LogP: {mol_info['logp']:.2f}")
```

### 範例 2: 分子優化

```python
# 在兩個已知活性分子之間尋找中間結構
active_molecule_1 = 'CC(=O)Nc1ccccc1'  # 已知活性
active_molecule_2 = 'COc1ccc(NC(=O)C)cc1'  # 已知活性

# 生成 20 個中間結構
candidates = generator.interpolate_molecules(
    active_molecule_1,
    active_molecule_2,
    num_steps=20
)

# 評估每個候選分子（這裡用分子量作為例子）
from rdkit import Chem
from rdkit.Chem import Descriptors

print("候選優化分子:")
for i, smiles in enumerate(candidates[1:-1], 1):  # 跳過起點和終點
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mw = Descriptors.MolWt(mol)
        print(f"{i}. {smiles} (MW: {mw:.2f})")
```

### 範例 3: 分子空間可視化

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 編碼一批分子
molecules = [...]  # 你的分子列表
latent_vectors = generator.encode_molecules(molecules)

# 使用 t-SNE 降維到 2D
tsne = TSNE(n_components=2, random_state=42)
latent_2d = tsne.fit_transform(latent_vectors.cpu().numpy())

# 可視化
plt.figure(figsize=(10, 8))
plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('Molecular Latent Space Visualization')
plt.savefig('latent_space_visualization.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 測試

運行內建測試:

```bash
python molecule_generator.py
```

這會執行所有功能的測試並顯示結果。

## 注意事項

1. **模型品質**: 生成的分子品質取決於訓練模型的品質
2. **有效性**: 並非所有生成的 SMILES 都是化學有效的，需要用 RDKit 驗證
3. **GPU 加速**: 如果有 GPU 可用，會自動使用以提升速度
4. **記憶體管理**: 編碼大量分子時注意記憶體使用

## 故障排除

### 問題 1: 找不到檔案

```
FileNotFoundError: [Errno 2] No such file or directory: 'train.yaml'
```

**解決方法**: 確保配置檔案、tokenizer 和模型檔案路徑正確。

### 問題 2: CUDA 記憶體不足

```
RuntimeError: CUDA out of memory
```

**解決方法**:
```python
# 使用 CPU
generator = VAEMoleculeGenerator(
    config_path='train.yaml',
    tokenizer_path='./checkpoints/tokenizer.json',
    checkpoint_path='./checkpoints/best_model.pt',
    device='cpu'
)

# 或減少批次大小
molecules = generator.sample_molecules(10)  # 而不是 1000
```

### 問題 3: 生成的 SMILES 無效

**解決方法**: 這是正常的，可以用 RDKit 過濾:

```python
from rdkit import Chem

molecules = generator.sample_molecules(100)
valid_molecules = [s for s in molecules if Chem.MolFromSmiles(s) is not None]
print(f"有效分子: {len(valid_molecules)}/{len(molecules)}")
```

## 總結

`VAEMoleculeGenerator` 提供了便捷的接口來使用訓練好的 VAE 模型進行分子生成和分析。主要功能包括:

- ✅ **隨機生成**: 探索新的化學空間
- ✅ **分子重建**: 驗證模型表示能力
- ✅ **分子插值**: 在已知分子間尋找新結構
- ✅ **分子編碼**: 用於下游分析和機器學習

建議使用流程:
1. 生成大量候選分子
2. 使用 RDKit 驗證有效性
3. 篩選符合條件的分子
4. 使用插值優化有潛力的分子
5. 用編碼向量進行相似性搜索
