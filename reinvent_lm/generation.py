"""
SmilesLM 分子生成器
提供分子隨機採樣、prefix 截斷接續生成（鄰近結構探索）、以及結合 filter/pareto 的鄰近分子搜索
"""

import os
from typing import Dict, List, Optional

import yaml
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from .tokenizer import SmilesTokenizer, canonicalize_smiles
from .models.lm import SmilesLM
from .pareto import PropertySpec, assign_pareto_fronts


class MoleculeGenerator:
    """SmilesLM 分子生成器類別"""

    def __init__(
        self,
        tokenizer: Optional[SmilesTokenizer] = None,
        max_length: Optional[int] = None,
        model: Optional[SmilesLM] = None,
        tokenizer_path: str = None,
        config_path: str = None,
        checkpoint_path: str = None,
        device: str = None,
    ):
        """
        支援兩種初始化模式：
        1. 使用已有的模型實例（用於訓練過程中）：給 model + tokenizer + max_length
        2. 從檢查點獨立載入（方便在任何地方直接用）：給
           tokenizer_path + config_path + checkpoint_path 三個路徑

        Args:
            tokenizer: SmilesTokenizer 實例（模式1 用）
            max_length: 最大序列長度（模式1 用；模式2 會從 config 讀取）
            model: (可選) 已有的模型實例
            tokenizer_path: tokenizer 檔案路徑 (tokenizer.json)，模式2 必填
            config_path: 配置檔案路徑 (train_reinvent.yaml)，模式2 必填
            checkpoint_path: 模型檢查點路徑 (.pt)，模式2 必填
            device: 運算設備 ('cuda' 或 'cpu'，預設自動偵測)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if model is not None:
            print("使用已有的模型實例")
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.model = model
            self.model.to(self.device)
            self.config = None
            print("✓ MoleculeGenerator 初始化完成 (使用已有模型)!\n")

        elif config_path is not None and checkpoint_path is not None:
            if tokenizer_path is None:
                raise ValueError("從檢查點載入時必須提供 tokenizer_path")

            print(f"使用設備: {self.device}")

            print(f"載入配置檔案: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)

            print(f"載入 tokenizer: {tokenizer_path}")
            self.tokenizer = SmilesTokenizer()
            self.tokenizer.load(tokenizer_path)

            print("建立模型...")
            model_config = self.config['model']
            self.model = SmilesLM(
                vocab_size=self.tokenizer.vocab_size,
                embedding_dim=model_config['embedding_dim'],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                dropout=model_config['dropout'],
                pad_idx=self.tokenizer.pad_idx,
            )

            print(f"載入模型權重: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            self.max_length = self.config['data']['max_length']

            print("✓ MoleculeGenerator 初始化完成 (從檢查點載入)!\n")

        else:
            raise ValueError(
                "必須提供以下其中一種:\n"
                "  1. model + tokenizer + max_length (已有的模型實例)\n"
                "  2. tokenizer_path + config_path + checkpoint_path (從檢查點獨立載入)"
            )

    def sample(
        self,
        num_samples: int,
        sampling_mode: str = 'multinomial',
        temperature: float = 1.0,
    ) -> List[str]:
        """從 BOS token 開始隨機採樣生成分子"""
        self.model.eval()
        with torch.no_grad():
            tokens = self.model.sample(
                num_samples=num_samples,
                max_length=self.max_length,
                start_idx=self.tokenizer.start_idx,
                device=self.device,
                sampling_mode=sampling_mode,
                temperature=temperature,
            )
        return [self.tokenizer.decode(tokens[i].cpu().tolist()) for i in range(num_samples)]

    def sample_from_prefix(
        self,
        smiles: str,
        num_samples: int,
        truncate_fraction: Optional[float] = None,
        truncate_fraction_range: tuple = (0.3, 0.7),
        sampling_mode: str = 'multinomial',
        temperature: float = 1.0,
    ) -> List[str]:
        """
        把種子分子的 SMILES 截斷到某個比例當 prefix，讓模型接續自回歸生成剩下的部分，
        藉此在沒有連續潛在空間的情況下做「鄰近結構探索」。

        Args:
            smiles: 種子分子的 SMILES
            num_samples: 要生成幾個鄰近分子
            truncate_fraction: 截斷比例（0~1），不填則每個樣本各自在
                truncate_fraction_range 範圍內隨機挑一個比例（更多樣）
            truncate_fraction_range: truncate_fraction 為 None 時的隨機範圍
            sampling_mode: 'greedy' 或 'multinomial'
            temperature: multinomial 模式下的取樣溫度

        Returns:
            鄰近分子的 SMILES 列表（prefix 部分保持不變，之後是模型接續生成的內容）
        """
        self.model.eval()
        base_indices = self.tokenizer.encode(smiles, add_special_tokens=False)
        if len(base_indices) < 2:
            raise ValueError(f"分子太短，無法截斷探索: {smiles}")

        # 同一次呼叫用同一個截斷長度：這樣整個 batch 的 prefix 長度一致，
        # teacher forcing 取得的 hidden state 不需要靠 padding 湊齊，
        # 避免補的 token 汙染 hidden state（每個樣本各自截斷長度不同時，
        # 短的 prefix 得額外塞假 token 才能跟長的一起 batch，那樣算出來的 hidden state
        # 就不是「剛好處理完真正 prefix」那個狀態了）
        low, high = truncate_fraction_range
        fraction = truncate_fraction if truncate_fraction is not None else np.random.uniform(low, high)
        cut = max(1, min(len(base_indices), round(len(base_indices) * fraction)))
        prefix_tokens = [self.tokenizer.start_idx] + base_indices[:cut]

        prefix_batch = torch.tensor(
            [prefix_tokens] * num_samples, dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            continuations = self.model.generate_from_prefix(
                prefix_batch,
                max_length=self.max_length,
                sampling_mode=sampling_mode,
                temperature=temperature,
            )

        results = []
        for i in range(num_samples):
            full_tokens = prefix_tokens[1:] + continuations[i].cpu().tolist()  # 去掉 START
            results.append(self.tokenizer.decode(full_tokens))
        return results

    def generate_analogs(
        self,
        smiles: str,
        num_candidates: int = 100,
        truncate_fraction: Optional[float] = None,
        truncate_fraction_range: tuple = (0.3, 0.7),
        sampling_mode: str = 'multinomial',
        temperature: float = 1.0,
        filter_api=None,
        inference_api=None,
        target_spec: Optional[Dict[str, PropertySpec]] = None,
        dedupe: bool = True,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        針對一個種子分子做「鄰近結構搜索」：用 sample_from_prefix 在附近取樣一批候選分子，
        依序套用結構規則過濾 (filter_api) 與性質目標的 pareto front 排序 (target_spec)，
        回傳排序後最好的候選分子。

        Args:
            smiles: 種子分子的 SMILES
            num_candidates: 取樣的候選分子數量
            truncate_fraction / truncate_fraction_range / sampling_mode / temperature: 同 sample_from_prefix
            filter_api: 結構規則過濾器，簽名為 filter_api(smiles_list) -> List[str]；不填則不過濾
            inference_api: 性質推論介面，需有 inference_pipeline(smiles_list, properties) -> DataFrame；
                有提供 target_spec 時必填
            target_spec: {性質名稱: PropertySpec}，用來做 pareto front 排序；不填則不排序
            dedupe: 是否對候選分子做 canonical 去重複，並排除跟種子分子本身相同的結果
            top_k: 只回傳排序後前 k 筆 (需搭配 target_spec)

        Returns:
            DataFrame，欄位為 ['smiles']，有給 target_spec 時額外附上各性質欄位與 'front_rank'
        """
        candidates = self.sample_from_prefix(
            smiles, num_candidates,
            truncate_fraction=truncate_fraction,
            truncate_fraction_range=truncate_fraction_range,
            sampling_mode=sampling_mode, temperature=temperature,
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
    """測試 MoleculeGenerator 的所有功能"""
    print("=" * 80)
    print("開始測試 MoleculeGenerator")
    print("=" * 80 + "\n")

    generator = MoleculeGenerator(
        tokenizer_path='./checkpoints/reinvent_small/tokenizer.json',
        config_path='configs/train_reinvent.yaml',
        checkpoint_path='./checkpoints/reinvent_small/best_model.pt',
    )

    print("=" * 80)
    print("測試 1: 隨機採樣生成分子")
    print("=" * 80)
    sampled = generator.sample(5)
    for i, smiles in enumerate(sampled, 1):
        valid = Chem.MolFromSmiles(smiles) is not None
        print(f"  [{i}] {'✓' if valid else '✗'} {smiles}")
    print()

    print("=" * 80)
    print("測試 2: prefix 截斷接續生成 (sample_from_prefix)")
    print("=" * 80)
    seed_smiles = "CCOc1ccccc1"
    neighbors = generator.sample_from_prefix(seed_smiles, num_samples=5)
    print(f"  種子分子: {seed_smiles}")
    for i, smiles in enumerate(neighbors, 1):
        print(f"    [{i}] {smiles}")
    print()

    print("=" * 80)
    print("測試 3: 鄰近分子搜索 (generate_analogs)")
    print("=" * 80)
    from .filters import StructureFilter
    from .properties import PropertyInferenceAPI

    analogs_df = generator.generate_analogs(
        seed_smiles,
        num_candidates=50,
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

    print("=" * 80)
    print("✓ 所有測試完成!")
    print("=" * 80)


if __name__ == "__main__":
    test_generator()
