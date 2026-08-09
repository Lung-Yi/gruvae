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
from .seed_utils import derive_seed


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
        seed: int = 0,
    ):
        """
        Args:
            csv_file: CSV 檔案路徑（需有 'smiles' 欄），跟 smiles_list 二選一
            smiles_list: 直接給一份已經讀好的 SMILES 列表，跟 csv_file 二選一
                （train/val 需要各自不同的 randomize 設定時，由呼叫端先切好 list 再傳進來）
            tokenizer / max_length: 保留參數，不影響這個類別本身的行為（實際編碼在 collate_fn 做）
            randomize: True 時每次取用都重新隨機化 SMILES 書寫法；False 時固定回傳 canonical 寫法
            seed: randomize=True 時，`(seed, epoch, idx)` 會被混合成該 item 的專屬
                隨機化 seed（見 __getitem__），讓結果不依賴 DataLoader worker 數量/
                行程排程，只要三者相同就一定重現同一個結果。搭配 set_epoch() 使用，
                讓同一個分子在不同 epoch 仍然拿到不同的隨機書寫法。
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.randomize = randomize
        self.seed = seed
        self._epoch = 0

        if smiles_list is not None:
            self.smiles_list = list(smiles_list)
        elif csv_file is not None:
            df = pd.read_csv(csv_file)
            self.smiles_list = df['smiles'].tolist()
        else:
            raise ValueError("必須提供 csv_file 或 smiles_list 其中一種")

        print(f"載入 {len(self.smiles_list)} 個 SMILES" + ("（訓練時隨機化 SMILES）" if randomize else ""))

    def set_epoch(self, epoch: int) -> None:
        """
        訓練迴圈每個 epoch 開始（建立/迭代該 epoch 的 DataLoader 之前）都要呼叫一次，
        讓 __getitem__ 算出的 per-item seed 隨 epoch 變化（同一個分子每個 epoch 仍會
        拿到不同的隨機書寫法），也讓 num_workers>0 時新 fork 出來的 worker 能拿到
        正確的 _epoch 值（沿用 PyTorch DistributedSampler.set_epoch() 的慣例寫法）。
        """
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> str:
        """回傳規範化或隨機化的 SMILES（在 collate_fn 中才編碼）"""
        smiles = self.smiles_list[idx]
        if not self.randomize:
            return canonicalize_smiles(smiles)
        item_seed = derive_seed(self.seed, self._epoch, idx)
        return randomize_smiles(smiles, seed=item_seed)


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

    # 可復現性：相同 (seed, epoch, idx) 兩次重跑（不同的 Dataset 實例）必須得到同一個結果，
    # 不同的 epoch 則應該（極高機率）得到不同的結果
    ds_a = SmilesLMDataset(smiles_list=[seed] * 5, randomize=True, seed=123)
    ds_b = SmilesLMDataset(smiles_list=[seed] * 5, randomize=True, seed=123)
    ds_a.set_epoch(3)
    ds_b.set_epoch(3)
    epoch3_a = [ds_a[i] for i in range(5)]
    epoch3_b = [ds_b[i] for i in range(5)]
    assert epoch3_a == epoch3_b, "同一個 (seed, epoch) 兩個 Dataset 實例結果不一致"
    ds_a.set_epoch(4)
    epoch4_a = [ds_a[i] for i in range(5)]
    assert epoch4_a != epoch3_a, "不同 epoch 應該（極高機率）產生不同的隨機書寫法"
    print("randomize=True 在相同 (seed, epoch, idx) 下可重現、不同 epoch 結果不同 ✓")

    print("✓ Dataset 基本測試通過")
