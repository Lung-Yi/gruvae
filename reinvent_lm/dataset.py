"""
SMILES LM Dataset 模組
沒有 encoder/decoder 之分，訓練資料就是單純的 token 序列（shifted 一格當 target）
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional

from .tokenizer import SmilesTokenizer, canonicalize_smiles, randomize_smiles


def pad_to_len(seq, max_len, pad_id):
    """Pad or truncate sequence to max_len. Accepts both list and tensor."""
    if isinstance(seq, torch.Tensor):
        seq = seq.tolist()

    if len(seq) >= max_len:
        return seq[:max_len]

    return seq + [pad_id] * (max_len - len(seq))


class SmilesLMDataset(Dataset):
    """
    SMILES 語言模型 Dataset。

    randomize=True 時，每次 __getitem__ 都會用 RDKit 對該分子重新隨機化一次 SMILES
    書寫順序（SMILES enumeration），而不是每次都回傳同一個 canonical 寫法。因為
    DataLoader 每個 epoch 都會重新呼叫 __getitem__，這樣同一個分子在不同 epoch
    （甚至同一個 epoch 內不同次取用）看到的書寫法都可能不一樣，等同免費做到
    REINVENT 系列論文（Bjerrum 2017; Arús-Pous et al. 2019）驗證過有效的
    「每個 epoch 用不同隨機 SMILES 訓練」資料增強，且成本跟原本 canonicalize 一次
    的開銷相當，不需要像 REINVENT 官方實作那樣預先產生多份枚舉檔案。
    """

    def __init__(
        self,
        csv_file: Optional[str] = None,
        tokenizer: Optional[SmilesTokenizer] = None,
        max_length: Optional[int] = None,
        smiles_list: Optional[List[str]] = None,
        randomize: bool = False,
    ):
        """
        Args:
            csv_file: CSV 檔案路徑（需有 'smiles' 欄），跟 smiles_list 二選一
            smiles_list: 直接給一份已經讀好的 SMILES 列表，跟 csv_file 二選一
                （train/val 需要各自不同的 randomize 設定時，由呼叫端先切好 list 再傳進來）
            tokenizer / max_length: 保留參數，不影響這個類別本身的行為（實際編碼在 collate_fn 做）
            randomize: True 時每次取用都重新隨機化 SMILES 書寫法；False 時固定回傳 canonical 寫法
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.randomize = randomize

        if smiles_list is not None:
            self.smiles_list = list(smiles_list)
        elif csv_file is not None:
            df = pd.read_csv(csv_file)
            self.smiles_list = df['smiles'].tolist()
        else:
            raise ValueError("必須提供 csv_file 或 smiles_list 其中一種")

        print(f"載入 {len(self.smiles_list)} 個 SMILES" + ("（訓練時隨機化 SMILES）" if randomize else ""))

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> str:
        """回傳規範化或隨機化的 SMILES（在 collate_fn 中才編碼）"""
        smiles = self.smiles_list[idx]
        return randomize_smiles(smiles) if self.randomize else canonicalize_smiles(smiles)


class DynamicSmilesLMDataset(Dataset):
    """
    訓練過程中內容可以動態調整的 SMILES Dataset（給 property-guided RL 微調用）。

    - base_smiles：固定不變的真實訓練資料
    - dynamic_smiles：由外部（PropertyGuidedLMTrainer）整批替換的動態資料，
      用來讓「訓練過程中新發現的優質分子」持續存在於訓練資料中。
    - randomize：跟 SmilesLMDataset 意義相同，對 base_smiles 跟 dynamic_smiles 都適用。
    """

    def __init__(self, base_smiles: List[str], randomize: bool = False):
        self.base_smiles = list(base_smiles)
        self.dynamic_smiles: List[str] = []
        self.randomize = randomize

    def set_dynamic_smiles(self, smiles_list: List[str]) -> None:
        """整批覆蓋目前的動態資料（呼叫端已經決定好要保留哪些分子）"""
        self.dynamic_smiles = list(smiles_list)

    def __len__(self) -> int:
        return len(self.base_smiles) + len(self.dynamic_smiles)

    def __getitem__(self, idx: int) -> str:
        if idx < len(self.base_smiles):
            smiles = self.base_smiles[idx]
        else:
            smiles = self.dynamic_smiles[idx - len(self.base_smiles)]
        return randomize_smiles(smiles) if self.randomize else canonicalize_smiles(smiles)


def collate_fn(
    batch: List[str],
    tokenizer: SmilesTokenizer,
    max_length: Optional[int] = 50,
):
    """
    Collate function for DataLoader

    Args:
        batch: canonical SMILES 字串列表
        tokenizer: SMILES tokenizer
        max_length: 最大序列長度

    Returns:
        input_seq: [batch_size, seq_len] - [START] + tokens（沒有 END），供模型輸入
        target_seq: [batch_size, seq_len] - tokens + [END]，供計算 loss 的目標
    """
    input_seqs = []
    target_seqs = []

    for canonical_smiles in batch:
        indices = tokenizer.encode(canonical_smiles, add_special_tokens=False)

        input_seq = [tokenizer.start_idx] + indices
        target_seq = indices + [tokenizer.end_idx]

        input_seq = pad_to_len(input_seq, max_length, tokenizer.pad_idx)
        target_seq = pad_to_len(target_seq, max_length, tokenizer.pad_idx)

        input_seqs.append(torch.tensor(input_seq, dtype=torch.long))
        target_seqs.append(torch.tensor(target_seq, dtype=torch.long))

    input_seq = pad_sequence(input_seqs, batch_first=True, padding_value=tokenizer.pad_idx)
    target_seq = pad_sequence(target_seqs, batch_first=True, padding_value=tokenizer.pad_idx)

    return input_seq, target_seq


def get_dataloader(
    csv_file: str,
    tokenizer: SmilesTokenizer,
    batch_size: int = 32,
    max_length: Optional[int] = 50,
    shuffle: bool = True,
    num_workers: int = 0,
    randomize: bool = False,
) -> DataLoader:
    """建立 DataLoader"""
    dataset = SmilesLMDataset(csv_file=csv_file, tokenizer=tokenizer, max_length=max_length, randomize=randomize)
    collate = lambda batch: collate_fn(batch, tokenizer, max_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
    )


if __name__ == "__main__":
    tokenizer = SmilesTokenizer()
    smiles_samples = ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]
    tokenizer.build_vocab(smiles_samples)

    batch = [canonicalize_smiles(s) for s in smiles_samples]
    input_seq, target_seq = collate_fn(batch, tokenizer, max_length=20)
    print(f"input_seq shape: {input_seq.shape}")
    print(f"target_seq shape: {target_seq.shape}")
    assert input_seq.shape == target_seq.shape == (4, 20)

    dynamic_dataset = DynamicSmilesLMDataset(smiles_samples)
    assert len(dynamic_dataset) == 4
    dynamic_dataset.set_dynamic_smiles(["CCCC", "CCCCC"])
    assert len(dynamic_dataset) == 6
    print(f"dynamic_dataset[5] = {dynamic_dataset[5]}")

    # randomize=True：每次取用都應該是合法、可被 RDKit 解析的同一個分子的某種書寫法
    from rdkit import Chem
    seed = "CCOc1ccccc1CC(=O)Nc1ccc(Cl)cc1"
    seed_canonical = canonicalize_smiles(seed)
    randomized_dataset = SmilesLMDataset(smiles_list=[seed] * 20, randomize=True)
    renderings = {randomized_dataset[i] for i in range(20)}
    for rendering in renderings:
        assert Chem.MolFromSmiles(rendering) is not None
        assert canonicalize_smiles(rendering) == seed_canonical
    print(f"randomize=True 20 次取用產生了 {len(renderings)} 種不同書寫法（同一個分子）")

    non_randomized_dataset = SmilesLMDataset(smiles_list=[seed], randomize=False)
    assert non_randomized_dataset[0] == seed_canonical
    print("randomize=False 固定回傳 canonical 寫法 ✓")

    print("✓ Dataset 基本測試通過")
