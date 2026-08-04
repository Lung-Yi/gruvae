"""
SmilesLM 分子生成器
提供分子隨機採樣、prefix 截斷接續生成（鄰近結構探索）、以及結合 filter/pareto 的鄰近分子搜索
"""

import os
import tempfile
from typing import Dict, List, Optional, Union

import yaml
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from .tokenizer import SmilesTokenizer, canonicalize_smiles, randomize_smiles
from .models.lm import SmilesLM
from .pareto import PropertySpec, assign_pareto_fronts
from .seed_utils import derive_seed, set_seed


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

    def _display_molecule_grid(
        self,
        smiles_list: List[str],
        legends: Optional[List[str]] = None,
        mols_per_row: int = 10,
        max_mols_per_image: int = 100,
    ) -> None:
        """
        把一批 SMILES 畫成分子結構網格圖顯示，每列 mols_per_row 個，每張圖最多
        max_mols_per_image 個，超過會自動切成多張圖依序顯示。

        用 useSVG=True 渲染，而不是 RDKit 預設的 raster (PNG) 模式：部分環境下用
        raster 模式畫 legend 文字時，底層 FreeType 字型渲染會直接讓 process segfault
        （這是 RDKit 在特定環境下的已知問題，不是這裡的邏輯錯誤），SVG 渲染完全不會
        走到那段文字光柵化的路徑，穩定很多，在 Jupyter 裡也能直接內嵌顯示。

        是否呼叫 IPython 的 display()：只在**確定已經有一個活著的 Jupyter kernel 在跑**時才用，
        用 `'ipykernel' in sys.modules` 判斷，而不是自己動手 `import IPython`。這裡刻意不自己
        匯入 IPython——在這台機器的環境下，即使只是單純 `from IPython import get_ipython`，
        只要在那之前已經呼叫過 RDKit 的 `Chem`/`Draw`，就會讓 process 直接 segfault
        （確認過是環境層級的 shared library 衝突，重現穩定但跟這裡的程式邏輯無關）。
        用 `sys.modules` 判斷完全不會觸發新的 import：真的在 Jupyter kernel 裡執行時，
        `ipykernel` 一定早就被 kernel 本身載入好了，這時候 `import IPython.display`
        只是查 cache，不會重新載入任何東西，所以是安全的；不在 kernel 裡執行時，
        就完全不去碰 IPython，一律存成 SVG 檔案。
        """
        import sys
        from rdkit.Chem import Draw

        if legends is None:
            legends = list(smiles_list)
        elif len(legends) != len(smiles_list):
            raise ValueError("legends 長度必須跟 smiles_list 一致")

        valid_entries = [
            (mol, legend)
            for smiles, legend in zip(smiles_list, legends)
            for mol in [Chem.MolFromSmiles(smiles)]
            if mol is not None
        ]
        num_invalid = len(smiles_list) - len(valid_entries)
        if num_invalid > 0:
            print(f"（{num_invalid} 個無效分子已略過，不會顯示在圖片中）")
        if not valid_entries:
            print("沒有可顯示的合法分子")
            return

        display = None
        SVG = None
        if 'ipykernel' in sys.modules:
            from IPython.display import display, SVG

        for page, start in enumerate(range(0, len(valid_entries), max_mols_per_image), start=1):
            chunk = valid_entries[start:start + max_mols_per_image]
            svg = Draw.MolsToGridImage(
                [mol for mol, _ in chunk],
                molsPerRow=mols_per_row,
                subImgSize=(200, 200),
                legends=[legend for _, legend in chunk],
                useSVG=True,
            )
            # RDKit 自己也會偵測是否在 IPython/notebook 環境下：偵測到的話
            # MolsToGridImage 會直接回傳一個包好的 IPython.display.SVG 物件，
            # 不是純文字；偵測不到才會回傳原始 SVG 字串。用型別判斷兩種情況，
            # 不要對已經是 SVG 物件的結果再包一層 SVG(...)（會直接壞掉）。
            if display is not None:
                display(svg if not isinstance(svg, str) else SVG(svg))
            else:
                svg_text = svg if isinstance(svg, str) else getattr(svg, 'data', str(svg))
                fd, path = tempfile.mkstemp(prefix=f'molecules_page{page}_', suffix='.svg')
                with os.fdopen(fd, 'w') as f:
                    f.write(svg_text)
                print(f"（非 Jupyter 環境，已將第 {page} 張分子結構圖存成 SVG: {path}）")

    def sample(
        self,
        num_samples: int,
        sampling_mode: str = 'multinomial',
        temperature: float = 1.0,
        seed: Optional[int] = None,
        display_molecules: bool = False,
        mols_per_row: int = 10,
        max_mols_per_image: int = 100,
    ) -> List[str]:
        """從 BOS token 開始隨機採樣生成分子

        seed 有給值時，同一組參數重複呼叫會得到完全一樣的結果；不填則維持原本
        每次呼叫都不同的隨機行為。

        display_molecules=True 時會額外畫出分子結構網格圖（每列 mols_per_row 個，
        每張圖最多 max_mols_per_image 個，超過自動分頁）。
        """
        if seed is not None:
            set_seed(seed)
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
        smiles_list = [self.tokenizer.decode(tokens[i].cpu().tolist()) for i in range(num_samples)]

        if display_molecules:
            self._display_molecule_grid(
                smiles_list, mols_per_row=mols_per_row, max_mols_per_image=max_mols_per_image
            )

        return smiles_list

    def sample_from_prefix(
        self,
        smiles: Union[str, List[str]],
        num_samples: int,
        truncate_fraction: Optional[float] = None,
        truncate_fraction_range: tuple = (0.3, 0.7),
        randomize_input: bool = True,
        sampling_mode: str = 'multinomial',
        temperature: float = 1.0,
        seed: Optional[int] = None,
        display_molecules: bool = False,
        mols_per_row: int = 10,
        max_mols_per_image: int = 100,
    ) -> Union[List[str], List[List[str]]]:
        """
        把種子分子的 SMILES 截斷到某個比例當 prefix，讓模型接續自回歸生成剩下的部分，
        藉此在沒有連續潛在空間的情況下做「鄰近結構探索」。

        Args:
            smiles: 單一種子 SMILES，或一個 SMILES list——給 list 時會分別對每一個輸入
                各自做鄰近探索（互不影響）
            num_samples: 「每個」種子分子要生成幾個鄰近分子
            truncate_fraction: 截斷比例（0~1），不填則每個樣本各自在
                truncate_fraction_range 範圍內隨機挑一個比例（更多樣）
            truncate_fraction_range: truncate_fraction 為 None 時的隨機範圍
            randomize_input: 是否在截斷前，先用 RDKit 對種子分子做一次隨機化 SMILES
                重新表示（`Chem.MolToSmiles(doRandom=True)`，即 SMILES enumeration）。
                同一個分子有非常多種合法但原子順序不同的書寫方式，從同一個比例截斷，
                不同的書寫方式會切到不同的子結構，能大幅增加鄰近探索的多樣性
                （每個樣本各自獨立抽一次新的隨機表示法，不是整批共用同一個）
            sampling_mode: 'greedy' 或 'multinomial'
            temperature: multinomial 模式下的取樣溫度
            seed: (可選) 給值時，同一組參數重複呼叫會得到完全一樣的結果（涵蓋截斷比例、
                randomize_input 的隨機表示法、以及模型的自回歸採樣）；不填則維持原本
                每次呼叫都不同的隨機行為
            display_molecules: True 時額外畫出分子結構網格圖（每列 mols_per_row 個，
                每張圖最多 max_mols_per_image 個，超過自動分頁）。smiles 為 list 時，
                所有種子的鄰近分子會畫在同一組圖裡，legend 會標上 `[seed i]` 方便分辨
                來自哪個種子
            mols_per_row / max_mols_per_image: 見 display_molecules

        Returns:
            smiles 為單一字串時：List[str]（長度 num_samples）
            smiles 為 list 時：List[List[str]]，跟輸入的 smiles list 順序一一對應，
                每個子 list 長度都是 num_samples
        """
        if seed is not None:
            set_seed(seed)
        self.model.eval()
        single_input = isinstance(smiles, str)
        smiles_list = [smiles] if single_input else list(smiles)

        low, high = truncate_fraction_range

        # 每個 (seed, sample) 各自獨立決定：要不要重新隨機表示 -> 用哪個截斷比例 -> prefix token 序列。
        # 不同樣本的 prefix 長度可能不一樣（隨機表示法的 token 數不保證跟原始 SMILES 相同），
        # 所以不能像單一固定比例那樣全部塞進同一個 batch tensor；改成依「長度」分組，
        # 同一組內長度一致才一起 batch teacher force，避免用 padding 湊長度污染 hidden state。
        all_prefixes: List[List[int]] = []
        owner: List[int] = []  # all_prefixes[i] 屬於哪個 seed（smiles_list 的 index）

        candidate_idx = 0
        for seed_idx, seed_smiles in enumerate(smiles_list):
            for _ in range(num_samples):
                if randomize_input:
                    item_seed = None if seed is None else derive_seed(seed, candidate_idx)
                    source = randomize_smiles(seed_smiles, seed=item_seed)
                else:
                    source = seed_smiles
                candidate_idx += 1
                base_indices = self.tokenizer.encode(source, add_special_tokens=False)
                if len(base_indices) < 2:
                    # 隨機化失敗或分子過短時，退回用原始種子 SMILES
                    base_indices = self.tokenizer.encode(seed_smiles, add_special_tokens=False)
                if len(base_indices) < 2:
                    raise ValueError(f"分子太短，無法截斷探索: {seed_smiles}")

                fraction = truncate_fraction if truncate_fraction is not None else np.random.uniform(low, high)
                cut = max(1, min(len(base_indices), round(len(base_indices) * fraction)))
                all_prefixes.append([self.tokenizer.start_idx] + base_indices[:cut])
                owner.append(seed_idx)

        results_flat: List[Optional[str]] = [None] * len(all_prefixes)
        length_groups: Dict[int, List[int]] = {}
        for i, prefix_tokens in enumerate(all_prefixes):
            length_groups.setdefault(len(prefix_tokens), []).append(i)

        with torch.no_grad():
            for indices in length_groups.values():
                prefix_batch = torch.tensor(
                    [all_prefixes[i] for i in indices], dtype=torch.long, device=self.device
                )
                continuations = self.model.generate_from_prefix(
                    prefix_batch,
                    max_length=self.max_length,
                    sampling_mode=sampling_mode,
                    temperature=temperature,
                )
                for j, i in enumerate(indices):
                    full_tokens = all_prefixes[i][1:] + continuations[j].cpu().tolist()  # 去掉 START
                    results_flat[i] = self.tokenizer.decode(full_tokens)

        grouped: List[List[str]] = [[] for _ in smiles_list]
        for i, seed_idx in enumerate(owner):
            grouped[seed_idx].append(results_flat[i])

        if display_molecules:
            if single_input:
                self._display_molecule_grid(
                    grouped[0], mols_per_row=mols_per_row, max_mols_per_image=max_mols_per_image
                )
            else:
                flat_smiles = [s for group in grouped for s in group]
                flat_legends = [
                    f"[seed {seed_idx}] {s}" for seed_idx, group in enumerate(grouped) for s in group
                ]
                self._display_molecule_grid(
                    flat_smiles, legends=flat_legends,
                    mols_per_row=mols_per_row, max_mols_per_image=max_mols_per_image,
                )

        return grouped[0] if single_input else grouped

    def generate_analogs(
        self,
        smiles: Union[str, List[str]],
        num_candidates: int = 100,
        truncate_fraction: Optional[float] = None,
        truncate_fraction_range: tuple = (0.3, 0.7),
        randomize_input: bool = True,
        sampling_mode: str = 'multinomial',
        temperature: float = 1.0,
        seed: Optional[int] = None,
        filter_api=None,
        inference_api=None,
        target_spec: Optional[Dict[str, PropertySpec]] = None,
        dedupe: bool = True,
        top_k: Optional[int] = None,
        display_molecules: bool = False,
        mols_per_row: int = 10,
        max_mols_per_image: int = 100,
    ) -> pd.DataFrame:
        """
        針對一個（或一批）種子分子做「鄰近結構搜索」：用 sample_from_prefix 在附近取樣候選
        分子，依序套用結構規則過濾 (filter_api) 與性質目標的 pareto front 排序 (target_spec)，
        回傳排序後最好的候選分子。

        給一個 SMILES list 時，所有種子的候選分子會被**合併成同一個候選池**一起去重複、
        過濾、排序（而不是每個種子分開各自排序）——這樣可以直接把一組 lead 化合物丟進來，
        一次拿到整組裡面「附近最好的候選」，回傳的 DataFrame 會多一欄 'seed' 標記每個
        候選分子是從哪個種子生成的，需要的話可以自行用 `groupby('seed')` 拆開看。

        Args:
            smiles: 種子分子的 SMILES，或一個 SMILES list
            num_candidates: 「每個」種子取樣的候選分子數量
            truncate_fraction / truncate_fraction_range / randomize_input / sampling_mode / temperature:
                同 sample_from_prefix
            seed: (可選) 見 sample_from_prefix 的說明；不填則維持原本每次呼叫都不同的隨機行為
            filter_api: 結構規則過濾器，簽名為 filter_api(smiles_list) -> List[str]；不填則不過濾
            inference_api: 性質推論介面，需有 inference_pipeline(smiles_list, properties) -> DataFrame；
                有提供 target_spec 時必填
            target_spec: {性質名稱: PropertySpec}，用來做 pareto front 排序；不填則不排序
            dedupe: 是否對候選分子做 canonical 去重複，並排除跟任一個輸入種子分子相同的結果
            top_k: 只回傳排序後前 k 筆 (需搭配 target_spec)
            display_molecules: True 時額外畫出最終結果的分子結構網格圖（每列 mols_per_row
                個，每張圖最多 max_mols_per_image 個，超過自動分頁）。有 target_spec 時
                legend 會標上 `front=<front_rank>`，方便一眼看出排序好壞
            mols_per_row / max_mols_per_image: 見 display_molecules

        Returns:
            DataFrame，欄位為 ['smiles', 'seed']，有給 target_spec 時額外附上各性質欄位與 'front_rank'
        """
        single_input = isinstance(smiles, str)
        seed_list = [smiles] if single_input else list(smiles)

        per_seed_candidates = self.sample_from_prefix(
            seed_list, num_candidates,
            truncate_fraction=truncate_fraction,
            truncate_fraction_range=truncate_fraction_range,
            randomize_input=randomize_input,
            sampling_mode=sampling_mode, temperature=temperature,
            seed=seed,
        )

        candidate_rows = [
            {'smiles': s, 'seed': seed}
            for seed, candidates in zip(seed_list, per_seed_candidates)
            for s in candidates
        ]
        candidate_rows = [row for row in candidate_rows if Chem.MolFromSmiles(row['smiles']) is not None]

        if dedupe:
            seed_canonicals = {canonicalize_smiles(s) for s in seed_list}
            seen = set()
            deduped_rows = []
            for row in candidate_rows:
                canonical = canonicalize_smiles(row['smiles'])
                if canonical in seed_canonicals or canonical in seen:
                    continue
                seen.add(canonical)
                deduped_rows.append(row)
            candidate_rows = deduped_rows

        candidate_smiles = [row['smiles'] for row in candidate_rows]
        candidate_seeds = [row['seed'] for row in candidate_rows]

        if filter_api is not None:
            pass_set = set(filter_api(candidate_smiles))
            keep = [i for i, s in enumerate(candidate_smiles) if s in pass_set]
            candidate_smiles = [candidate_smiles[i] for i in keep]
            candidate_seeds = [candidate_seeds[i] for i in keep]

        if not candidate_smiles:
            result_df = pd.DataFrame(columns=['smiles', 'seed'])
        elif target_spec is None:
            result_df = pd.DataFrame({'smiles': candidate_smiles, 'seed': candidate_seeds})
        else:
            if inference_api is None:
                raise ValueError("提供 target_spec 時必須同時提供 inference_api")

            prop_df = inference_api.inference_pipeline(candidate_smiles, properties=list(target_spec.keys()))
            prop_df.insert(1, 'seed', candidate_seeds)
            objective_matrix = np.array([
                [target_spec[prop].to_objective(row[prop]) for prop in target_spec]
                for _, row in prop_df.iterrows()
            ])
            prop_df['front_rank'] = assign_pareto_fronts(objective_matrix)
            prop_df = prop_df.sort_values('front_rank').reset_index(drop=True)

            if top_k is not None:
                prop_df = prop_df.head(top_k)

            result_df = prop_df

        if display_molecules and len(result_df) > 0:
            if 'front_rank' in result_df.columns:
                legends = [
                    f"front={rank}  {s}" for rank, s in zip(result_df['front_rank'], result_df['smiles'])
                ]
            else:
                legends = result_df['smiles'].tolist()
            self._display_molecule_grid(
                result_df['smiles'].tolist(), legends=legends,
                mols_per_row=mols_per_row, max_mols_per_image=max_mols_per_image,
            )

        return result_df


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
    print("測試 2b: prefix 截斷接續生成，輸入一個 SMILES list")
    print("=" * 80)
    seed_list = ["CCOc1ccccc1", "CCN(CC)CC"]
    neighbors_list = generator.sample_from_prefix(seed_list, num_samples=3)
    for seed, neighbors_for_seed in zip(seed_list, neighbors_list):
        print(f"  種子分子: {seed}")
        for i, smiles in enumerate(neighbors_for_seed, 1):
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
    print("測試 4: seed 可復現性")
    print("=" * 80)
    sample_a = generator.sample(8, seed=123)
    sample_b = generator.sample(8, seed=123)
    assert sample_a == sample_b, "generator.sample(seed=123) 兩次呼叫結果不一致"
    sample_c = generator.sample(8, seed=456)
    assert sample_a != sample_c, "不同 seed 應該（極高機率）產生不同結果"
    print("  sample(seed=123) 兩次呼叫結果一致 ✓；換 seed=456 結果不同 ✓")

    prefix_a = generator.sample_from_prefix(seed_smiles, num_samples=6, seed=123)
    prefix_b = generator.sample_from_prefix(seed_smiles, num_samples=6, seed=123)
    assert prefix_a == prefix_b, "sample_from_prefix(seed=123) 兩次呼叫結果不一致"
    print("  sample_from_prefix(seed=123) 兩次呼叫結果一致 ✓")

    analogs_a = generator.generate_analogs(
        seed_smiles, num_candidates=20,
        filter_api=StructureFilter(), inference_api=PropertyInferenceAPI(),
        target_spec={
            "ClogP": PropertySpec(goal="range", low=1.0, high=3.0),
            "SAScore": PropertySpec(goal="minimize"),
        },
        top_k=5, seed=123,
    )
    analogs_b = generator.generate_analogs(
        seed_smiles, num_candidates=20,
        filter_api=StructureFilter(), inference_api=PropertyInferenceAPI(),
        target_spec={
            "ClogP": PropertySpec(goal="range", low=1.0, high=3.0),
            "SAScore": PropertySpec(goal="minimize"),
        },
        top_k=5, seed=123,
    )
    assert analogs_a['smiles'].tolist() == analogs_b['smiles'].tolist(), \
        "generate_analogs(seed=123) 兩次呼叫結果不一致"
    print("  generate_analogs(seed=123) 兩次呼叫結果一致 ✓")

    no_seed_a = generator.sample(8)
    no_seed_b = generator.sample(8)
    assert no_seed_a != no_seed_b, "不傳 seed 時應維持原本每次呼叫都不同的隨機行為"
    print("  不傳 seed 時維持原本非固定隨機行為 ✓")
    print()

    print("=" * 80)
    print("測試 3b: 鄰近分子搜索，輸入一個 SMILES list (合併排序，附 'seed' 欄)")
    print("=" * 80)
    analogs_df_multi = generator.generate_analogs(
        seed_list,
        num_candidates=50,
        filter_api=StructureFilter(),
        inference_api=PropertyInferenceAPI(),
        target_spec={
            "ClogP": PropertySpec(goal="range", low=1.0, high=3.0),
            "SAScore": PropertySpec(goal="minimize"),
        },
        top_k=8,
    )
    print(f"  種子分子: {seed_list}")
    print(analogs_df_multi)
    print()

    print("=" * 80)
    print("✓ 所有測試完成!")
    print("=" * 80)


if __name__ == "__main__":
    test_generator()
