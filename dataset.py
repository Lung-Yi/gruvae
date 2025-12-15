"""
VAE Dataset 模組
處理 SMILES 資料，支援隨機化輸入和規範化輸出
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Optional
from tokenizer import SmilesTokenizer, randomize_smiles, canonicalize_smiles

def pad_to_len(seq, max_len, pad_id):
    if len(seq) >= max_len:
        return seq[:max_len]
    return torch.cat([
        seq,
        torch.full((max_len - len(seq),), pad_id, dtype=seq.dtype)
    ])

class SmilesVAEDataset(Dataset):
    """SMILES VAE Dataset"""

    def __init__(
        self,
        csv_file: str,
        tokenizer: SmilesTokenizer,
        max_length: Optional[int] = None
    ):
        """
        Args:
            csv_file: CSV 文件路徑
            tokenizer: SMILES tokenizer
            max_length: 最大序列長度（包含特殊 tokens）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 讀取資料
        df = pd.read_csv(csv_file)
        self.smiles_list = df['smiles'].tolist()

        print(f"載入 {len(self.smiles_list)} 個 SMILES")

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """
        返回一對 (randomized_smiles, canonical_smiles)
        在 collate_fn 中會進行編碼
        """
        smiles = self.smiles_list[idx]

        # 規範化 SMILES（作為 target）
        canonical_smiles = canonicalize_smiles(smiles)

        # 隨機化 SMILES（作為 input）
        # randomized_smiles = randomize_smiles(canonical_smiles)

        return canonical_smiles, canonical_smiles


def collate_fn(
    batch: List[Tuple[str, str]],
    tokenizer: SmilesTokenizer,
    max_length: Optional[int] = 50
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader

    Args:
        batch: List of (canonical_smiles, canonical_smiles) tuples
        tokenizer: SMILES tokenizer
        max_length: 最大序列長度

    Returns:
        encoder_input: [batch_size, seq_len] - 編碼器輸入（canonical + START + END）
        decoder_input: [batch_size, seq_len] - 解碼器輸入（canonical + START）
        decoder_target: [batch_size, seq_len] - 解碼器目標（canonical + END）
    """
    encoder_inputs = []
    decoder_inputs = []
    decoder_targets = []

    for randomized_smiles, canonical_smiles in batch:
        # 編碼器輸入：隨機化的 SMILES（帶 START 和 END）
        encoder_indices = tokenizer.encode(randomized_smiles, add_special_tokens=True)

        # 解碼器輸入：規範化的 SMILES（帶 START，不帶 END）
        canonical_indices = tokenizer.encode(canonical_smiles, add_special_tokens=False)
        decoder_input = [tokenizer.start_idx] + canonical_indices

        # 解碼器目標：規範化的 SMILES（不帶 START，帶 END）
        decoder_target = canonical_indices + [tokenizer.end_idx]

        # 如果設置了最大長度，進行截斷
        encoder_indices = pad_to_len(encoder_indices, max_length, tokenizer.pad_idx)
        decoder_input = pad_to_len(decoder_input, max_length, tokenizer.pad_idx)
        decoder_target = pad_to_len(decoder_target, max_length, tokenizer.pad_idx)

        encoder_inputs.append(torch.tensor(encoder_indices, dtype=torch.long))
        decoder_inputs.append(torch.tensor(decoder_input, dtype=torch.long))
        decoder_targets.append(torch.tensor(decoder_target, dtype=torch.long))

    # Padding
    encoder_input = pad_sequence(
        encoder_inputs,
        batch_first=True,
        padding_value=tokenizer.pad_idx
    )
    decoder_input = pad_sequence(
        decoder_inputs,
        batch_first=True,
        padding_value=tokenizer.pad_idx
    )
    decoder_target = pad_sequence(
        decoder_targets,
        batch_first=True,
        padding_value=tokenizer.pad_idx
    )

    return encoder_input, decoder_input, decoder_target


def get_dataloader(
    csv_file: str,
    tokenizer: SmilesTokenizer,
    batch_size: int = 32,
    max_length: Optional[int] = 50,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """建立 DataLoader"""

    dataset = SmilesVAEDataset(csv_file, tokenizer, max_length)

    # 使用 lambda 來傳遞 tokenizer 和 max_length
    collate = lambda batch: collate_fn(batch, tokenizer, max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate
    )

    return dataloader


if __name__ == "__main__":
    # 測試 Dataset
    from tokenizer import SmilesTokenizer

    # 建立 tokenizer
    tokenizer = SmilesTokenizer()

    # 從 CSV 讀取並建立詞彙表
    df = pd.read_csv('./data/train_processed.csv')
    smiles_list = df['smiles'].tolist()
    tokenizer.build_vocab(smiles_list)

    # 測試 Dataset
    dataset = SmilesVAEDataset('./data/train_processed.csv', tokenizer)
    print(f"Dataset 大小: {len(dataset)}")

    # 測試 collate_fn
    batch = [dataset[i] for i in range(3)]
    encoder_input, decoder_input, decoder_target = collate_fn(batch, tokenizer)

    print(f"\nEncoder input shape: {encoder_input.shape}")
    print(f"Decoder input shape: {decoder_input.shape}")
    print(f"Decoder target shape: {decoder_target.shape}")

    # 測試 DataLoader
    dataloader = get_dataloader('./data/train_processed.csv', tokenizer, batch_size=4)
    encoder_input, decoder_input, decoder_target = next(iter(dataloader))
    print(f"\nDataLoader batch:")
    print(f"Encoder input: {encoder_input.shape}")
    print(f"Decoder input: {decoder_input.shape}")
    print(f"Decoder target: {decoder_target.shape}")
