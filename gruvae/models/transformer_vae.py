import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    標準 Transformer 位置編碼
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # [Seq_Len, 1, Dim] for batch_first=False default in older PyTorch, 
        # but here we usually use batch_first=True. Let's adjust in forward.
        self.register_buffer('pe', pe.unsqueeze(0)) # [1, Max_Len, Dim]

    def forward(self, x):
        # x: [Batch, Seq_Len, Dim]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerVAE(nn.Module):
    """
    基於 mizuno-group/TransformerVAE 的分子生成模型
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,        # 論文預設
        nhead: int = 8,            # 論文預設
        num_encoder_layers: int = 8, # 論文特點: 加深 Encoder
        num_decoder_layers: int = 6, # 通常 Decoder 維持 6 層或與 Encoder 相同
        dim_feedforward: int = 2048,
        latent_dim: int = 128,     # 潛在空間維度
        dropout: float = 0.1,
        max_len: int = 128,        # 分子最大長度
        pad_idx: int = 0,
        start_idx: int = 1,
        end_idx: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.pad_idx = pad_idx

        # --- Embedding & Positional Encoding ---
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # --- Encoder ---
        # "Pre-LN structure was adopted" -> norm_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True,
            norm_first=True  # 關鍵設置：Pre-Layer Norm
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # --- Latent Space (Bottleneck) ---
        # 將 Encoder 輸出壓縮為 mean 和 logvar
        # 我們使用 Global Average Pooling 或取第一個 token (CLS) 的方式
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)
        
        # 將 Latent 映射回 Decoder 輸入大小
        self.latent_to_hidden = nn.Linear(latent_dim, d_model)

        # --- Decoder ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # --- Output Head ---
        self.fc_out = nn.Linear(d_model, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None, teacher_forcing=True):
        """
        src: [Batch, Seq_Len] (SMILES tokens)
        tgt: [Batch, Seq_Len] (Target tokens for teacher forcing)
        teacher_forcing: 是否使用 teacher forcing
        """
        batch_size = src.size(0)
        seq_len = tgt.size(1)

        # 1. Encode
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        # Transformer Encoder
        # memory: [Batch, Seq_Len, d_model]
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        # 2. Bottleneck (Pooling)
        # 這裡使用 Mean Pooling 將序列壓縮為單一向量
        # Mask 掉 padding 部分以獲得準確的平均
        if src_key_padding_mask is not None:
            mask = ~src_key_padding_mask.unsqueeze(-1) # [Batch, Seq, 1]
            memory_pooled = (memory * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            memory_pooled = memory.mean(dim=1)

        mu = self.fc_mu(memory_pooled)
        logvar = self.fc_logvar(memory_pooled)
        z = self.reparameterize(mu, logvar)

        # 3. Decode
        # 將 z 映射回 d_model 並作為 Decoder 的 "Memory" (Context)
        # Reshape to [Batch, 1, d_model] so decoder can attend to it
        z_memory = self.latent_to_hidden(z).unsqueeze(1)

        if teacher_forcing:
            # Teacher forcing: 使用真實的目標序列作為輸入
            tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)

            # Transformer Decoder
            output = self.transformer_decoder(
                tgt=tgt_emb,
                memory=z_memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
                # memory_key_padding_mask is not needed as memory is length 1
            )

            logits = self.fc_out(output)
        else:
            # 自回歸生成：一步一步生成
            # 從 START token 開始
            outputs = []
            input_token = tgt[:, 0:1]  # [Batch, 1] - START token

            for t in range(seq_len):
                # Embedding
                tgt_emb = self.embedding(input_token) * math.sqrt(self.d_model)
                tgt_emb = self.pos_encoder(tgt_emb)

                # 生成 causal mask (只看到當前和之前的位置)
                current_len = t + 1
                mask = self.generate_square_subsequent_mask(current_len).to(src.device)

                # Decoder
                output = self.transformer_decoder(
                    tgt=tgt_emb,
                    memory=z_memory,
                    tgt_mask=mask
                )

                # Output projection
                step_logits = self.fc_out(output[:, -1:, :])  # [Batch, 1, vocab_size]
                outputs.append(step_logits)

                # 使用預測的 token 作為下一個輸入
                next_token = step_logits.argmax(dim=-1)  # [Batch, 1]
                input_token = torch.cat([input_token, next_token], dim=1)  # [Batch, t+2]

            logits = torch.cat(outputs, dim=1)  # [Batch, Seq_Len, vocab_size]

        return logits, mu, logvar

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

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

            # 將 z 映射回 d_model 並作為 Decoder 的 "Memory"
            z_memory = self.latent_to_hidden(z).unsqueeze(1)  # [num_samples, 1, d_model]

            # 自回歸生成
            outputs = []
            input_token = torch.full((num_samples, 1), start_idx, dtype=torch.long).to(device)

            for t in range(max_length):
                # Embedding
                tgt_emb = self.embedding(input_token) * math.sqrt(self.d_model)
                tgt_emb = self.pos_encoder(tgt_emb)

                # 生成 causal mask
                current_len = t + 1
                mask = self.generate_square_subsequent_mask(current_len).to(device)

                # Decoder
                output = self.transformer_decoder(
                    tgt=tgt_emb,
                    memory=z_memory,
                    tgt_mask=mask
                )

                # Output projection
                step_logits = self.fc_out(output[:, -1:, :])  # [num_samples, 1, vocab_size]

                # 使用預測的 token
                next_token = step_logits.argmax(dim=-1)  # [num_samples, 1]
                outputs.append(next_token)
                input_token = torch.cat([input_token, next_token], dim=1)  # [num_samples, t+2]

            samples = torch.cat(outputs, dim=1)  # [num_samples, max_length]

        return samples


if __name__ == "__main__":
    # 測試 TransformerVAE
    print("=" * 80)
    print("Testing TransformerVAE")
    print("=" * 80)

    vocab_size = 100
    batch_size = 4
    seq_len = 50

    model = TransformerVAE(
        vocab_size=vocab_size,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        latent_dim=64,
        dropout=0.1,
        max_len=128,
        pad_idx=0,
        start_idx=1,
        end_idx=2
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 創建測試數據
    src = torch.randint(3, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(3, vocab_size, (batch_size, seq_len))

    # Test 1: Teacher forcing
    print("\n" + "=" * 80)
    print("Test 1: Forward with teacher_forcing=True")
    print("=" * 80)

    # 創建 masks
    src_key_padding_mask = (src == 0)
    tgt_key_padding_mask = (tgt == 0)
    tgt_mask = model.generate_square_subsequent_mask(seq_len)

    output, mu, logvar = model(
        src, tgt,
        src_key_padding_mask=src_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        tgt_mask=tgt_mask,
        teacher_forcing=True
    )

    print(f"Output shape: {output.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    print(f"✓ Teacher forcing test passed!")

    # Test 2: Without teacher forcing (autoregressive)
    print("\n" + "=" * 80)
    print("Test 2: Forward with teacher_forcing=False (autoregressive)")
    print("=" * 80)

    output_no_tf, mu_no_tf, logvar_no_tf = model(
        src, tgt,
        src_key_padding_mask=src_key_padding_mask,
        teacher_forcing=False
    )

    print(f"Output shape: {output_no_tf.shape}")
    print(f"Mu shape: {mu_no_tf.shape}")
    print(f"Logvar shape: {logvar_no_tf.shape}")
    print(f"✓ Autoregressive generation test passed!")

    # Test 3: Sampling from latent space
    print("\n" + "=" * 80)
    print("Test 3: Sampling from latent space")
    print("=" * 80)

    num_samples = 3
    samples = model.sample(
        num_samples=num_samples,
        max_length=seq_len,
        start_idx=1,
        device='cpu'
    )

    print(f"Samples shape: {samples.shape}")
    print(f"✓ Sampling test passed!")

    # Test 4: Compare outputs with and without teacher forcing
    print("\n" + "=" * 80)
    print("Test 4: Comparing outputs")
    print("=" * 80)

    print(f"With teacher forcing - Output mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")
    print(f"Without teacher forcing - Output mean: {output_no_tf.mean().item():.4f}, std: {output_no_tf.std().item():.4f}")
    print(f"Outputs are different (as expected): {not torch.allclose(output, output_no_tf)}")

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
