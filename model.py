"""
GRU-VAE 模型架構
包含 Encoder、Decoder 和完整的 VAE 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GRUEncoder(nn.Module):
    """GRU Encoder - 將輸入序列編碼為潛在向量"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        """
        Args:
            vocab_size: 詞彙表大小
            embedding_dim: Embedding 維度
            hidden_dim: GRU 隱藏層維度
            latent_dim: 潛在空間維度
            num_layers: GRU 層數
            dropout: Dropout 比率
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding 層
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # GRU 層
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 潛在空間映射
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len] - 輸入序列

        Returns:
            mu: [batch_size, latent_dim] - 潛在分布均值
            logvar: [batch_size, latent_dim] - 潛在分布對數方差
        """
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]

        # GRU encoding
        _, hidden = self.gru(embedded)  # hidden: [num_layers, batch_size, hidden_dim]

        # 使用最後一層的隱藏狀態
        h = hidden[-1]  # [batch_size, hidden_dim]

        # 映射到潛在空間
        mu = self.fc_mu(h)  # [batch_size, latent_dim]
        logvar = self.fc_logvar(h)  # [batch_size, latent_dim]

        return mu, logvar


class GRUDecoder(nn.Module):
    """GRU Decoder - 從潛在向量解碼為序列"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        """
        Args:
            vocab_size: 詞彙表大小
            embedding_dim: Embedding 維度
            hidden_dim: GRU 隱藏層維度
            latent_dim: 潛在空間維度
            num_layers: GRU 層數
            dropout: Dropout 比率
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # Embedding 層
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 潛在向量到初始隱藏狀態的映射
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)

        # GRU 層
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 輸出層
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        z: torch.Tensor,
        target: torch.Tensor,
        teacher_forcing: bool = True
    ) -> torch.Tensor:
        """
        Args:
            z: [batch_size, latent_dim] - 潛在向量
            target: [batch_size, seq_len] - 目標序列（用於 teacher forcing）
            teacher_forcing: 是否使用 teacher forcing

        Returns:
            output: [batch_size, seq_len, vocab_size] - 輸出 logits
        """
        batch_size = z.size(0)
        seq_len = target.size(1)

        # 將潛在向量映射到初始隱藏狀態
        h = self.latent_to_hidden(z)  # [batch_size, hidden_dim * num_layers]
        h = h.view(batch_size, self.num_layers, self.hidden_dim)  # [batch_size, num_layers, hidden_dim]
        h = h.permute(1, 0, 2).contiguous()  # [num_layers, batch_size, hidden_dim]

        if teacher_forcing:
            # Teacher forcing: 使用真實的目標序列作為輸入
            embedded = self.embedding(target)  # [batch_size, seq_len, embedding_dim]
            output, _ = self.gru(embedded, h)  # [batch_size, seq_len, hidden_dim]
            output = self.fc_out(output)  # [batch_size, seq_len, vocab_size]
        else:
            # 自回歸生成：一步一步生成
            outputs = []
            input_token = target[:, 0:1]  # [batch_size, 1] - START token

            for t in range(seq_len):
                embedded = self.embedding(input_token)  # [batch_size, 1, embedding_dim]
                output, h = self.gru(embedded, h)  # output: [batch_size, 1, hidden_dim]
                output = self.fc_out(output)  # [batch_size, 1, vocab_size]

                outputs.append(output)

                # 使用預測的 token 作為下一個輸入
                input_token = output.argmax(dim=-1)  # [batch_size, 1]

            output = torch.cat(outputs, dim=1)  # [batch_size, seq_len, vocab_size]

        return output


class GRUVAE(nn.Module):
    """完整的 GRU-VAE 模型"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        latent_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            vocab_size: 詞彙表大小
            embedding_dim: Embedding 維度
            hidden_dim: GRU 隱藏層維度
            latent_dim: 潛在空間維度
            num_layers: GRU 層數
            dropout: Dropout 比率
        """
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.encoder = GRUEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # Decoder
        self.decoder = GRUDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重參數化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        teacher_forcing: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_input: [batch_size, seq_len] - 編碼器輸入
            decoder_input: [batch_size, seq_len] - 解碼器輸入
            teacher_forcing: 是否使用 teacher forcing

        Returns:
            output: [batch_size, seq_len, vocab_size] - 輸出 logits
            mu: [batch_size, latent_dim] - 潛在分布均值
            logvar: [batch_size, latent_dim] - 潛在分布對數方差
        """
        # Encode
        mu, logvar = self.encoder(encoder_input)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        output = self.decoder(z, decoder_input, teacher_forcing=teacher_forcing)

        return output, mu, logvar

    def sample(
        self,
        num_samples: int,
        max_length: int,
        start_idx: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        從潛在空間採樣生成新分子

        Args:
            num_samples: 要生成的樣本數
            max_length: 最大序列長度
            start_idx: START token 的索引
            device: 設備

        Returns:
            samples: [num_samples, max_length] - 生成的序列
        """
        self.eval()
        with torch.no_grad():
            # 從標準正態分布採樣潛在向量
            z = torch.randn(num_samples, self.latent_dim).to(device)

            # 創建 decoder 輸入（全是 START token）
            decoder_input = torch.full((num_samples, max_length), start_idx, dtype=torch.long).to(device)

            # 不使用 teacher forcing 進行生成
            output = self.decoder(z, decoder_input, teacher_forcing=False)

            # 取最大概率的 token
            samples = output.argmax(dim=-1)

        return samples


def compute_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    pad_idx: int,
    kl_weight: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    計算 VAE 損失

    Args:
        output: [batch_size, seq_len, vocab_size] - 模型輸出
        target: [batch_size, seq_len] - 目標序列
        mu: [batch_size, latent_dim] - 潛在分布均值
        logvar: [batch_size, latent_dim] - 潛在分布對數方差
        pad_idx: PAD token 的索引
        kl_weight: KL 散度的權重

    Returns:
        total_loss: 總損失
        recon_loss: 重建損失
        kl_loss: KL 散度損失
    """
    # 重建損失（交叉熵）
    output = output.view(-1, output.size(-1))  # [batch_size * seq_len, vocab_size]
    target = target.view(-1)  # [batch_size * seq_len]

    recon_loss = F.cross_entropy(
        output,
        target,
        ignore_index=pad_idx,
        reduction='mean'
    )

    # KL 散度損失
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # 總損失
    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss


if __name__ == "__main__":
    # 測試模型
    vocab_size = 100
    batch_size = 4
    seq_len = 50

    model = GRUVAE(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        latent_dim=64,
        num_layers=2
    )

    # 創建測試數據
    encoder_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    decoder_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    target = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Teacher forcing
    print("Testing with teacher forcing...")
    output, mu, logvar = model(encoder_input, decoder_input, teacher_forcing=True)
    print(f"Output shape: {output.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")

    # 計算損失
    loss, recon_loss, kl_loss = compute_loss(output, target, mu, logvar, pad_idx=0)
    print(f"Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}")

    # Without teacher forcing
    print("\nTesting without teacher forcing...")
    output, mu, logvar = model(encoder_input, decoder_input, teacher_forcing=False)
    print(f"Output shape: {output.shape}")

    # Sampling
    print("\nTesting sampling...")
    samples = model.sample(num_samples=3, max_length=seq_len, start_idx=1, device='cpu')
    print(f"Samples shape: {samples.shape}")
