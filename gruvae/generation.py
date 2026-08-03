"""
VAE 分子生成器
提供分子生成、重建、插值和編碼功能
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from .tokenizer import SmilesTokenizer, canonicalize_smiles
from .dataset import collate_fn, pad_to_len
from .models.gru_vae import GRUVAE
from .pareto import PropertySpec, assign_pareto_fronts
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class VAEMoleculeGenerator:
    """VAE 分子生成器類別"""

    def __init__(
        self,
        tokenizer: Optional[SmilesTokenizer] = None,
        max_length: Optional[int] = None,
        model=None,
        tokenizer_path: str = None,
        config_path: str = None,
        checkpoint_path: str = None,
        device: str = None
    ):
        """
        初始化 VAE 分子生成器

        支援兩種初始化模式:
        1. 使用已有的模型實例 (用於訓練過程中)：給 model + tokenizer + max_length
        2. 從檢查點獨立載入 (方便在任何地方直接用)：只需給
           tokenizer_path + config_path + checkpoint_path 三個路徑就好

        Args:
            tokenizer: SmilesTokenizer 實例 (模式1 用)
            max_length: 最大序列長度 (模式1 用；模式2 會從 config 讀取)
            model: (可選) 已有的模型實例。如果提供,則不需要 config_path 和 checkpoint_path
            tokenizer_path: tokenizer 檔案路徑 (tokenizer.json)，模式2 必填
            config_path: 配置檔案路徑 (train.yaml)，模式2 必填
            checkpoint_path: 模型檢查點路徑 (.pt)，模式2 必填
            device: 運算設備 ('cuda' 或 'cpu'，預設自動偵測)
        """
        # 設置設備
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 模式1: 使用已有的模型實例
        if model is not None:
            print(f"使用已有的模型實例")
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.model = model
            self.model.to(self.device)
            self.config = None
            print("✓ VAE 分子生成器初始化完成 (使用已有模型)!\n")

        # 模式2: 從檢查點獨立載入
        elif config_path is not None and checkpoint_path is not None:
            if tokenizer_path is None:
                raise ValueError("從檢查點載入時必須提供 tokenizer_path")

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
            self.model = GRUVAE(
                vocab_size=self.tokenizer.vocab_size,
                embedding_dim=model_config['embedding_dim'],
                hidden_dim=model_config['hidden_dim'],
                latent_dim=model_config['latent_dim'],
                num_layers=model_config['num_layers'],
                dropout=model_config['dropout'],
                bidirectional=model_config.get('bidirectional', False)
            )

            # 載入模型權重
            print(f"載入模型權重: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            # 獲取配置參數
            self.max_length = self.config['data']['max_length']

            print("✓ VAE 分子生成器初始化完成 (從檢查點載入)!\n")

        else:
            raise ValueError(
                "必須提供以下其中一種:\n"
                "  1. model + tokenizer + max_length (已有的模型實例)\n"
                "  2. tokenizer_path + config_path + checkpoint_path (從檢查點獨立載入)"
            )

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
                encoder_input, _decoder_input, decoder_target = collate_fn([(smiles, smiles)], tokenizer=self.tokenizer, max_length=self.max_length)

                # 編碼輸入
                input_indices = self.tokenizer.encode(smiles, add_special_tokens=True)
                input_tensor = torch.tensor(input_indices, dtype=torch.long).to(self.device)
                input_tensor = pad_to_len(input_tensor, self.max_length, self.tokenizer.pad_idx)
                # print("input_tensor")
                # print(input_tensor)
                input_tensor = torch.tensor([input_tensor], dtype=torch.long).to(self.device)

                encoder_input = torch.tensor(encoder_input, dtype=torch.long).to(self.device)
                # decoder_input = torch.tensor(decoder_input, dtype=torch.long).to(self.device)
                # print("encoder_input")
                # print(encoder_input)
                # print("input_tensor")
                # print(input_tensor)

                # 創建 decoder 輸入（僅 START token）
                decoder_input = torch.full(
                    (1, self.max_length),
                    self.tokenizer.start_idx,
                    dtype=torch.long
                ).to(self.device)

                # 前向傳播（不使用 teacher forcing）
                output, mu, logvar = self.model(
                    input_tensor, # input_tensor
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
            # tensor1 = torch.tensor([indices1], dtype=torch.long).to(self.device)
            tensor1 = pad_to_len(indices1, self.max_length, self.tokenizer.pad_idx)
            tensor1 = torch.tensor([tensor1], dtype=torch.long).to(self.device)
            mu1, logvar1 = self.model.encoder(tensor1)
            z1 = mu1  # 使用均值（不加噪聲）

            # 分子 2
            indices2 = self.tokenizer.encode(smiles2, add_special_tokens=True)
            # tensor2 = torch.tensor([indices2], dtype=torch.long).to(self.device)
            tensor2 = pad_to_len(indices2, self.max_length, self.tokenizer.pad_idx)
            tensor2 = torch.tensor([tensor2], dtype=torch.long).to(self.device)
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

    def sample_around(
        self,
        smiles: str,
        num_samples: int,
        noise_scale: float = 0.5,
        sampling_mode: str = 'multinomial',
        temperature: float = 1.0,
    ) -> List[str]:
        """
        以某個分子在潛在空間的位置為中心，加高斯雜訊後解碼，探索附近的結構。

        Args:
            smiles: 種子分子的 SMILES
            num_samples: 要生成幾個鄰近分子
            noise_scale: 高斯雜訊的標準差，越大探索範圍越廣、跟原分子差異越大
            sampling_mode: 'greedy' 或 'multinomial'（建議用 multinomial 才有多樣性，
                不然雜訊還是一樣但每次都 argmax，很容易生出重複的分子）
            temperature: multinomial 模式下的取樣溫度

        Returns:
            鄰近分子的 SMILES 列表 (長度為 num_samples，可能含重複/無效分子)
        """
        self.model.eval()
        with torch.no_grad():
            indices = self.tokenizer.encode(smiles, add_special_tokens=True)
            tensor = pad_to_len(indices, self.max_length, self.tokenizer.pad_idx)
            tensor = torch.tensor([tensor], dtype=torch.long).to(self.device)
            mu, _ = self.model.encoder(tensor)  # [1, latent_dim]

            z_center = mu.repeat(num_samples, 1)
            z = z_center + torch.randn_like(z_center) * noise_scale

            tokens = self.model.decoder.generate(
                z, self.tokenizer.start_idx, self.max_length,
                sampling_mode=sampling_mode, temperature=temperature
            )

        return [self.tokenizer.decode(tokens[i].cpu().tolist()) for i in range(num_samples)]

    def generate_analogs(
        self,
        smiles: str,
        num_candidates: int = 100,
        noise_scale: float = 0.5,
        sampling_mode: str = 'multinomial',
        temperature: float = 1.0,
        filter_api=None,
        inference_api=None,
        target_spec: Optional[Dict[str, PropertySpec]] = None,
        dedupe: bool = True,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        針對一個種子分子做「鄰近結構搜索」：在潛在空間附近取樣一批候選分子，
        依序套用結構規則過濾 (filter_api) 與性質目標的 pareto front 排序 (target_spec)，
        回傳排序後最好的候選分子。等於是把 RL 訓練用的 filter/pareto 機制搬來做
        推論階段的應用：手上已經有一個分子，想找附近結構更好的替代品。

        Args:
            smiles: 種子分子的 SMILES
            num_candidates: 潛在空間附近取樣的候選分子數量
            noise_scale / sampling_mode / temperature: 同 sample_around
            filter_api: 結構規則過濾器，簽名為 filter_api(smiles_list) -> List[str]；
                不填則不做結構過濾
            inference_api: 性質推論介面，需有 inference_pipeline(smiles_list, properties) -> DataFrame；
                有提供 target_spec 時必填
            target_spec: {性質名稱: PropertySpec}，用來做 pareto front 排序；
                不填則不排序，只回傳過濾/去重後的候選
            dedupe: 是否對候選分子做 canonical 去重複，並排除跟種子分子本身相同的結果
            top_k: 只回傳排序後前 k 筆 (需搭配 target_spec)；不填則回傳全部

        Returns:
            DataFrame。欄位為 ['smiles']，有給 target_spec 時額外附上各性質欄位與
            'front_rank'（越小代表越好，同一層 front 內不分優劣，順序依來源決定）
        """
        candidates = self.sample_around(
            smiles, num_candidates,
            noise_scale=noise_scale, sampling_mode=sampling_mode, temperature=temperature
        )

        valid_candidates = [s for s in candidates if Chem.MolFromSmiles(s) is not None]

        if dedupe:
            seed_canonical = canonicalize_smiles(smiles)
            seen = set()
            deduped = []
            for s in valid_candidates:
                canonical = canonicalize_smiles(s)
                if canonical == seed_canonical or canonical in seen:
                    continue
                seen.add(canonical)
                deduped.append(s)
            valid_candidates = deduped

        if filter_api is not None:
            valid_candidates = filter_api(valid_candidates)

        if not valid_candidates:
            return pd.DataFrame(columns=['smiles'])

        if target_spec is None:
            return pd.DataFrame({'smiles': valid_candidates})

        if inference_api is None:
            raise ValueError("提供 target_spec 時必須同時提供 inference_api")

        prop_df = inference_api.inference_pipeline(valid_candidates, properties=list(target_spec.keys()))
        objective_matrix = np.array([
            [target_spec[prop].to_objective(row[prop]) for prop in target_spec]
            for _, row in prop_df.iterrows()
        ])
        prop_df['front_rank'] = assign_pareto_fronts(objective_matrix)
        prop_df = prop_df.sort_values('front_rank').reset_index(drop=True)

        if top_k is not None:
            prop_df = prop_df.head(top_k)

        return prop_df


def test_generator():
    """測試 VAEMoleculeGenerator 的所有功能"""
    print("="*80)
    print("開始測試 VAEMoleculeGenerator")
    print("="*80 + "\n")

    # 初始化生成器 (從檢查點獨立載入模式，只需給三個路徑)
    generator = VAEMoleculeGenerator(
        tokenizer_path='./checkpoints/gru/tokenizer.json',
        config_path='configs/train.yaml',
        checkpoint_path='./checkpoints/gru/checkpoint_epoch_20.pt'
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

    # 測試 5: 鄰近結構採樣
    print("="*80)
    print("測試 5: 鄰近結構採樣 (sample_around)")
    print("="*80)
    seed_smiles = "CCO"
    neighbors = generator.sample_around(seed_smiles, num_samples=5, noise_scale=0.5)
    print(f"  種子分子: {seed_smiles}")
    print("  鄰近分子:")
    for i, smiles in enumerate(neighbors, 1):
        print(f"    [{i}] {smiles}")
    print()

    # 測試 6: 鄰近分子搜索 (結構過濾 + 性質 pareto 排序)
    print("="*80)
    print("測試 6: 鄰近分子搜索 (generate_analogs)")
    print("="*80)
    from .filters import StructureFilter
    from .properties import PropertyInferenceAPI

    analogs_df = generator.generate_analogs(
        seed_smiles,
        num_candidates=50,
        noise_scale=0.5,
        filter_api=StructureFilter(),
        inference_api=PropertyInferenceAPI(),
        target_spec={
            "ClogP": PropertySpec(goal="range", low=1.0, high=3.0),
            "SAScore": PropertySpec(goal="minimize"),
        },
        top_k=5,
    )
    print(f"  種子分子: {seed_smiles}")
    print(analogs_df)
    print()

    print("="*80)
    print("✓ 所有測試完成!")
    print("="*80)


if __name__ == "__main__":
    # 執行測試
    test_generator()
