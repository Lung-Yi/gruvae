"""
SMILES LM Dataset 模組
沒有 encoder/decoder 之分，訓練資料就是單純的 token 序列（shifted 一格當 target）
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional

from .tokenizer import SmilesTokenizer, canonicalize_smiles


def pad_to_len(seq, max_len, pad_id):
    """Pad or truncate sequence to max_len. Accepts both list and tensor."""
    if isinstance(seq, torch.Tensor):
        seq = seq.tolist()

    if len(seq) >= max_len:
        return seq[:max_len]

    return seq + [pad_id] * (max_len - len(seq))


class SmilesLMDataset(Dataset):
    """SMILES 語言模型 Dataset"""

    def __init__(self, csv_file: str, tokenizer: SmilesTokenizer, max_length: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        df = pd.read_csv(csv_file)
        self.smiles_list = df['smiles'].tolist()

        print(f"載入 {len(self.smiles_list)} 個 SMILES")

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> str:
        """回傳規範化的 SMILES（在 collate_fn 中才編碼）"""
        smiles = self.smiles_list[idx]
        return canonicalize_smiles(smiles)


class DynamicSmilesLMDataset(Dataset):
    """
    訓練過程中內容可以動態調整的 SMILES Dataset（給 property-guided RL 微調用）。

    - base_smiles：固定不變的真實訓練資料
    - dynamic_smiles：由外部（PropertyGuidedLMTrainer）整批替換的動態資料，
      用來讓「訓練過程中新發現的優質分子」持續存在於訓練資料中。
    """

    def __init__(self, base_smiles: List[str]):
        self.base_smiles = list(base_smiles)
        self.dynamic_smiles: List[str] = []

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
        return canonicalize_smiles(smiles)


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
) -> DataLoader:
    """建立 DataLoader"""
    dataset = SmilesLMDataset(csv_file, tokenizer, max_length)
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

    print("✓ Dataset 基本測試通過")
