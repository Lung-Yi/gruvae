"""
VAE 分子生成器
提供分子生成、重建、插值和編碼功能
"""

import os
import yaml
import torch
import numpy as np
from typing import List, Tuple, Dict
# 嘗試相對導入，如果失敗則使用絕對導入
try:
    from ..data.tokenizer import SmilesTokenizer, canonicalize_smiles
    from ..models.gru_vae import GRUVAE
    from ..models.transformer_vae import TransformerVAE
except ImportError:
    from data.tokenizer import SmilesTokenizer, canonicalize_smiles
    from models.gru_vae import GRUVAE
    from models.transformer_vae import TransformerVAE

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class VAEMoleculeGenerator:
    """VAE 分子生成器類別"""

    def __init__(
        self,
        config_path: str,
        tokenizer_path: str,
        checkpoint_path: str,
        device: str = None
    ):
        """
        初始化 VAE 分子生成器

        Args:
            config_path: 配置檔案路徑 (train.yaml)
            tokenizer_path: tokenizer 檔案路徑 (tokenizer.json)
            checkpoint_path: 模型檢查點路徑 (.pt)
            device: 運算設備 ('cuda' 或 'cpu'，預設自動偵測)
        """
        # 設置設備
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"使用設備: {self.device}")

        # 載入配置
        print(f"載入配置檔案: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 載入 tokenizer
        print(f"載入 tokenizer: {tokenizer_path}")
        self.tokenizer = SmilesTokenizer()
        self.tokenizer.load(tokenizer_path)

        # 建立模型
        print("建立模型...")
        model_config = self.config['model']
        model_type = model_config.get('model_type', 'gru').lower()

        if model_type == 'transformer':
            print("使用 Transformer VAE 模型")
            self.model = TransformerVAE(
                vocab_size=self.tokenizer.vocab_size,
                d_model=model_config.get('d_model', 512),
                nhead=model_config.get('nhead', 8),
                num_encoder_layers=model_config.get('num_encoder_layers', 8),
                num_decoder_layers=model_config.get('num_decoder_layers', 6),
                dim_feedforward=model_config.get('dim_feedforward', 2048),
                latent_dim=model_config.get('latent_dim', 128),
                dropout=model_config.get('dropout', 0.1),
                max_len=model_config.get('max_len', 128),
                pad_idx=self.tokenizer.pad_idx,
                start_idx=self.tokenizer.start_idx,
                end_idx=self.tokenizer.end_idx
            )
        elif model_type == 'gru':
            print("使用 GRU VAE 模型")
            self.model = GRUVAE(
                vocab_size=self.tokenizer.vocab_size,
                embedding_dim=model_config['embedding_dim'],
                hidden_dim=model_config['hidden_dim'],
                latent_dim=model_config['latent_dim'],
                num_layers=model_config['num_layers'],
                dropout=model_config['dropout'],
                bidirectional=model_config.get('bidirectional', True)
            )
        else:
            raise ValueError(f"不支援的模型類型: {model_type}")

        # 載入模型權重
        print(f"載入模型權重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 獲取配置參數
        self.max_length = self.config['data']['max_length']

        print("✓ VAE 分子生成器初始化完成!\n")

    def sample_molecules(self, num_samples: int) -> List[str]:
        """
        從潛在空間隨機採樣生成分子

        Args:
            num_samples: 要生成的分子數量

        Returns:
            生成的 SMILES 列表
        """
        print(f"正在生成 {num_samples} 個分子...")

        with torch.no_grad():
            # 從潛在空間採樣
            samples = self.model.sample(
                num_samples=num_samples,
                max_length=self.max_length,
                start_idx=self.tokenizer.start_idx,
                device=self.device
            )

            # 解碼為 SMILES
            smiles_list = []
            for i in range(num_samples):
                smiles = self.tokenizer.decode(samples[i].cpu().tolist())
                smiles_list.append(smiles)

        print(f"✓ 成功生成 {num_samples} 個分子\n")
        return smiles_list

    def reconstruct_molecules(self, smiles_list: List[str]) -> List[Dict[str, str]]:
        """
        重建給定的 SMILES 分子

        Args:
            smiles_list: 輸入的 SMILES 列表

        Returns:
            重建結果列表，每個元素包含:
            - input: 原始輸入 SMILES
            - input_canonical: 規範化的輸入 SMILES
            - reconstructed: 重建的 SMILES
            - reconstructed_canonical: 規範化的重建 SMILES
            - match: 是否匹配 (True/False)
        """
        print(f"正在重建 {len(smiles_list)} 個分子...")

        results = []
        self.model.eval()

        with torch.no_grad():
            for smiles in smiles_list:
                # 編碼輸入
                input_indices = self.tokenizer.encode(smiles, add_special_tokens=True)
                input_tensor = torch.tensor([input_indices], dtype=torch.long).to(self.device)

                # 創建 decoder 輸入（僅 START token）
                decoder_input = torch.full(
                    (1, self.max_length),
                    self.tokenizer.start_idx,
                    dtype=torch.long
                ).to(self.device)

                # 前向傳播（不使用 teacher forcing）
                output, mu, logvar = self.model(
                    input_tensor,
                    decoder_input,
                    teacher_forcing=False
                )

                # 解碼輸出
                predicted = output.argmax(dim=-1)
                reconstructed = self.tokenizer.decode(predicted[0].cpu().tolist())

                # 規範化
                input_canonical = canonicalize_smiles(smiles)
                reconstructed_canonical = canonicalize_smiles(reconstructed)

                # 檢查是否匹配
                match = (input_canonical == reconstructed_canonical)

                results.append({
                    'input': smiles,
                    'input_canonical': input_canonical,
                    'reconstructed': reconstructed,
                    'reconstructed_canonical': reconstructed_canonical,
                    'match': match
                })

        print(f"✓ 成功重建 {len(smiles_list)} 個分子\n")
        return results

    def interpolate_molecules(
        self,
        smiles1: str,
        smiles2: str,
        num_steps: int
    ) -> List[str]:
        """
        在兩個分子之間進行潛在空間線性插值

        Args:
            smiles1: 第一個 SMILES
            smiles2: 第二個 SMILES
            num_steps: 插值步數（包含起點和終點）

        Returns:
            插值生成的 SMILES 列表
        """
        print(f"正在對兩個分子進行 {num_steps} 步插值...")

        self.model.eval()

        with torch.no_grad():
            # 編碼兩個分子到潛在空間
            # 分子 1
            indices1 = self.tokenizer.encode(smiles1, add_special_tokens=True)
            tensor1 = torch.tensor([indices1], dtype=torch.long).to(self.device)
            mu1, logvar1 = self.model.encoder(tensor1)
            z1 = mu1  # 使用均值（不加噪聲）

            # 分子 2
            indices2 = self.tokenizer.encode(smiles2, add_special_tokens=True)
            tensor2 = torch.tensor([indices2], dtype=torch.long).to(self.device)
            mu2, logvar2 = self.model.encoder(tensor2)
            z2 = mu2  # 使用均值（不加噪聲）

            # 線性插值
            interpolated_smiles = []
            for i in range(num_steps):
                # 計算插值係數
                alpha = i / (num_steps - 1) if num_steps > 1 else 0
                z_interp = (1 - alpha) * z1 + alpha * z2

                # 解碼插值的潛在向量
                decoder_input = torch.full(
                    (1, self.max_length),
                    self.tokenizer.start_idx,
                    dtype=torch.long
                ).to(self.device)

                output = self.model.decoder(z_interp, decoder_input, teacher_forcing=False)
                predicted = output.argmax(dim=-1)
                smiles = self.tokenizer.decode(predicted[0].cpu().tolist())
                interpolated_smiles.append(smiles)

        print(f"✓ 成功生成 {num_steps} 個插值分子\n")
        return interpolated_smiles

    def encode_molecules(self, smiles_list: List[str]) -> torch.Tensor:
        """
        將 SMILES 列表編碼為潛在向量

        Args:
            smiles_list: 輸入的 SMILES 列表

        Returns:
            潛在向量張量 [num_molecules, latent_dim]
        """
        print(f"正在編碼 {len(smiles_list)} 個分子...")

        self.model.eval()
        latent_vectors = []

        with torch.no_grad():
            for smiles in smiles_list:
                # 編碼
                indices = self.tokenizer.encode(smiles, add_special_tokens=True)
                tensor = torch.tensor([indices], dtype=torch.long).to(self.device)

                # 獲取潛在向量（使用均值）
                mu, logvar = self.model.encoder(tensor)
                latent_vectors.append(mu)

            # 合併為一個張量
            latent_tensor = torch.cat(latent_vectors, dim=0)

        print(f"✓ 成功編碼 {len(smiles_list)} 個分子")
        print(f"  潛在向量維度: {latent_tensor.shape}\n")

        return latent_tensor


def test_generator():
    """測試 VAEMoleculeGenerator 的所有功能"""
    print("="*80)
    print("開始測試 VAEMoleculeGenerator")
    print("="*80 + "\n")

    # 初始化生成器
    generator = VAEMoleculeGenerator(
        config_path='train.yaml',
        tokenizer_path='./checkpoints/tokenizer.json',
        checkpoint_path='./checkpoints/checkpoint_epoch_20.pt'
    )

    # 測試 1: 隨機採樣生成分子
    print("="*80)
    print("測試 1: 隨機採樣生成分子")
    print("="*80)
    num_samples = 5
    sampled_molecules = generator.sample_molecules(num_samples)

    print("生成的分子:")
    for i, smiles in enumerate(sampled_molecules, 1):
        canonical = canonicalize_smiles(smiles)
        print(f"  [{i}] {smiles}")
        print(f"      (Canonical: {canonical})")
    print()

    # 測試 2: 分子重建
    print("="*80)
    print("測試 2: 分子重建")
    print("="*80)
    test_smiles = [
        "CCO",  # 乙醇
        "c1ccccc1",  # 苯
        "CC(=O)O",  # 乙酸
    ]

    reconstruction_results = generator.reconstruct_molecules(test_smiles)

    print("重建結果:")
    for i, result in enumerate(reconstruction_results, 1):
        print(f"\n  [{i}] {'✓' if result['match'] else '✗'} Match: {result['match']}")
        print(f"      Input:         {result['input']}")
        print(f"      Input (Can):   {result['input_canonical']}")
        print(f"      Reconstructed: {result['reconstructed']}")
        print(f"      Recon (Can):   {result['reconstructed_canonical']}")

    # 計算重建準確率
    accuracy = sum(r['match'] for r in reconstruction_results) / len(reconstruction_results)
    print(f"\n  重建準確率: {accuracy:.2%}")
    print()

    # 測試 3: 分子插值
    print("="*80)
    print("測試 3: 分子插值")
    print("="*80)
    smiles1 = "COCCNC(=O)COc1cc(C)c(Br)c(C)c1"  # 1
    smiles2 = "O=C(COc1ccc(Cl)cc1)Nc1ccc(O)cc1"  # 2
    num_steps = 5

    print(f"  起點: {smiles1} ({canonicalize_smiles(smiles1)})")
    print(f"  終點: {smiles2} ({canonicalize_smiles(smiles2)})")
    print(f"  步數: {num_steps}\n")

    interpolated = generator.interpolate_molecules(smiles1, smiles2, num_steps)

    print("插值結果:")
    for i, smiles in enumerate(interpolated):
        canonical = canonicalize_smiles(smiles)
        alpha = i / (num_steps - 1) if num_steps > 1 else 0
        print(f"  [{i}] α={alpha:.2f}: {smiles}")
        print(f"              (Canonical: {canonical})")
    print()

    # 測試 4: 編碼分子
    print("="*80)
    print("測試 4: 編碼分子到潛在空間")
    print("="*80)
    encode_smiles = [
        "CCO",
        "Cc1ccc(NS(=O)(=O)c2cc(C)n(C)c2C)nc1",
        "c1ccccc1",
        "CC(=O)O",
        "CCCC",
    ]

    latent_vectors = generator.encode_molecules(encode_smiles)

    print("編碼結果:")
    print(f"  輸入分子數: {len(encode_smiles)}")
    print(f"  潛在向量維度: {latent_vectors.shape}")
    print(f"  數據類型: {latent_vectors.dtype}")
    print(f"  設備: {latent_vectors.device}")
    print(f"\n  前 5 個潛在向量的值 (每個分子):")
    for i, smiles in enumerate(encode_smiles):
        print(f"    [{i}] {smiles}: {latent_vectors[i, :5].cpu().numpy()}")
    print()

    # 額外測試：計算兩個分子的潛在空間距離
    print("="*80)
    print("額外測試: 計算分子間的潛在空間距離")
    print("="*80)
    print("計算前兩個分子在潛在空間中的歐氏距離:")
    dist = torch.norm(latent_vectors[0] - latent_vectors[1]).item()
    print(f"  {encode_smiles[0]} <-> {encode_smiles[1]}")
    print(f"  距離: {dist:.4f}\n")

    print("="*80)
    print("✓ 所有測試完成!")
    print("="*80)


if __name__ == "__main__":
    # 執行測試
    test_generator()
