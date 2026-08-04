"""
SmilesLM：REINVENT 風格的純 autoregressive GRU 語言模型
沒有 encoder、沒有潛在空間 z，訓練規則（teacher forcing）跟生成規則（自回歸）是同一件事
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .sampling import sample_next_token
from ..seed_utils import set_seed


class SmilesLM(nn.Module):
    """GRU decoder-only 的 SMILES 語言模型"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_tokens: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        整段 teacher forcing 前向傳播

        Args:
            input_tokens: [batch_size, seq_len]
            hidden: (可選) 初始 hidden state，[num_layers, batch_size, hidden_dim]

        Returns:
            logits: [batch_size, seq_len, vocab_size]
            hidden: 最後一步的 hidden state
        """
        embedded = self.embedding(input_tokens)
        output, hidden = self.gru(embedded, hidden)
        logits = self.fc_out(output)
        return logits, hidden

    def sample(
        self,
        num_samples: int,
        max_length: int,
        start_idx: int,
        device: torch.device,
        sampling_mode: str = 'multinomial',
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        從 BOS token 開始逐步自回歸生成（REINVENT Agent 的採樣方式）

        seed 有給值時，函式最開頭會呼叫 seed_utils.set_seed(seed) 種好全域 RNG，
        讓「同 seed、同輸入參數 -> 同輸出」成立（給 MoleculeGenerator 等獨立於
        訓練流程之外呼叫的場合用；seed=None 時完全不影響現有行為）。

        Returns:
            tokens: [num_samples, max_length]
        """
        if seed is not None:
            set_seed(seed)
        self.eval()
        with torch.no_grad():
            input_token = torch.full((num_samples, 1), start_idx, dtype=torch.long, device=device)
            hidden = None
            tokens = []

            for _ in range(max_length):
                embedded = self.embedding(input_token)
                output, hidden = self.gru(embedded, hidden)
                logits = self.fc_out(output)  # [num_samples, 1, vocab_size]

                next_token = sample_next_token(logits, sampling_mode=sampling_mode, temperature=temperature)
                tokens.append(next_token)
                input_token = next_token

        return torch.cat(tokens, dim=1)

    def generate_from_prefix(
        self,
        prefix_tokens: torch.Tensor,
        max_length: int,
        sampling_mode: str = 'multinomial',
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        把 prefix_tokens teacher-force 過 GRU 取得該點的 hidden state，
        之後從那個 hidden state 繼續自回歸生成剩餘長度（sample_from_prefix 的核心）

        Args:
            prefix_tokens: [num_samples, prefix_len]，已含 START token，不含 END/PAD
            max_length: 總長度上限（含 prefix）
            seed: (可選) 見 sample() 的說明

        Returns:
            tokens: [num_samples, max_length - prefix_len]，接續 prefix 之後生成的部分
        """
        if seed is not None:
            set_seed(seed)
        self.eval()
        num_samples = prefix_tokens.size(0)
        device = prefix_tokens.device
        remaining = max_length - prefix_tokens.size(1)
        tokens = []

        with torch.no_grad():
            # teacher-force 整段 prefix，取「最後一個位置」的輸出來預測 prefix 之後的第一個新 token
            # （不能只留 hidden 就把 prefix 最後一個 token 又當成新輸入丟一次，那樣會讓
            #  GRU 把最後一個 token 處理兩遍，等於憑空多塞了一步、之後全部生成都會錯位）
            embedded = self.embedding(prefix_tokens)
            output, hidden = self.gru(embedded, None)
            logits = self.fc_out(output[:, -1:, :])

            if remaining > 0:
                next_token = sample_next_token(logits, sampling_mode=sampling_mode, temperature=temperature)
                tokens.append(next_token)
                input_token = next_token

                for _ in range(remaining - 1):
                    embedded = self.embedding(input_token)
                    output, hidden = self.gru(embedded, hidden)
                    logits = self.fc_out(output)

                    next_token = sample_next_token(logits, sampling_mode=sampling_mode, temperature=temperature)
                    tokens.append(next_token)
                    input_token = next_token

        if not tokens:
            return torch.zeros((num_samples, 0), dtype=torch.long, device=device)
        return torch.cat(tokens, dim=1)


if __name__ == "__main__":
    vocab_size = 30
    batch_size = 4
    seq_len = 20

    model = SmilesLM(vocab_size=vocab_size, embedding_dim=32, hidden_dim=64, num_layers=2)

    input_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, hidden = model(input_tokens)
    print(f"forward logits shape: {logits.shape}, hidden shape: {hidden.shape}")
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert hidden.shape == (2, batch_size, 64)

    samples = model.sample(num_samples=3, max_length=seq_len, start_idx=1, device='cpu')
    print(f"sample tokens shape: {samples.shape}")
    assert samples.shape == (3, seq_len)

    prefix = torch.randint(0, vocab_size, (3, 8))
    cont = model.generate_from_prefix(prefix, max_length=seq_len)
    print(f"generate_from_prefix tokens shape: {cont.shape}")
    assert cont.shape == (3, seq_len - 8)

    print("✓ SmilesLM 基本測試通過")
