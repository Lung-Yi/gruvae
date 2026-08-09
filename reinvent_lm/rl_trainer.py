"""
PropertyGuidedLMTrainer：在標準 SmilesLM 預訓練之外，額外用 REINFORCE (policy gradient)
針對「結構規則」與「分子性質目標」對模型做微調。跟 gruvae.rl_trainer.PropertyGuidedTrainer
邏輯完全對應，差別是這裡的底層模型是純 autoregressive GRU（沒有 encoder/z），
所以不需要 model_type 分支，也不需要在 log-prob 重算時傳遞/固定 z。

整體流程：
    1. 前 warmup_epochs 個 epoch 只做原本的監督式訓練（MLE），把模型調整成適合
       filter_api 篩選過的訓練資料的分布
    2. warmup 結束：把目前的模型凍結成 prior（`_snapshot_prior()`，RL 微調時用來做
       KL-style 正則化，避免生成多樣性崩潰），並額外從這個 prior model 大量取樣，
       算出每個性質 to_objective() 距離的平均值/標準差，當作絕對 reward 的正規化基準
       （`_compute_reward_normalization_stats()`）——這個基準之後整個訓練過程都固定
       不變，不會隨訓練往下漂移
    3. warmup 結束之後，每個 epoch 只做 RL 微調，不再穿插任何監督式訓練：
       用目前的模型從 BOS 開始自回歸採樣一批分子（multinomial 隨機採樣）
       -> 用 filter_api 判斷結構是否合規 -> 對合規的分子用 inference_api 計算性質，
       把每個性質的 to_objective() 距離轉成 z-score、取 exp(-max(z,0))，多個性質再用
       幾何平均合併成一個「絕對 desirability」分數（`_compute_absolute_desirability()`），
       reward 完全由這個絕對分數決定（不看任何排名/歷史比較），用 REINFORCE + baseline
       做 policy gradient 更新，並用凍結的 prior 模型做正則化
    4. 通過結構檢查的分子，除了拿去算 reward，也會被拿去更新一個「菁英 archive
       (elite_archive)」：跟 archive 目前的成員合併重新做一次 pareto front 排序，
       只保留 rank < elite_archive_rank（前幾層）的分子。這個 archive **不影響 reward
       或訓練**，純粹是讓使用者事後能分析「這個生成器最終找到哪些好分子」的歷史紀錄
    5. 每一輪 pareto front 1（相對於 elite_archive 最好的一層）分子的結構與性質會被
       印出來，並累積寫進 CSV log，方便訓練過程中/事後監看
    6. 每一輪更新完 elite_archive 之後，把 archive 裡目前 pareto rank 前幾層（預設
       前 5 層）的分子（SMILES、每個性質的原始數值、rank）整批覆蓋寫進另一份 CSV，
       只反映 archive 最新狀態，方便隨時查看目前最好的一批分子長怎樣
"""

import copy
import csv
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .training import Trainer
from .tokenizer import canonicalize_smiles
from .pareto import PropertySpec, assign_pareto_fronts
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class PropertyGuidedLMTrainer(Trainer):
    """
    繼承自 Trainer，複用 train_epoch / validate / save_checkpoint，
    只在 train() 的 epoch 迴圈中，warmup 結束後改成跑 property-guided 的 RL 微調回合。
    """

    def __init__(
        self,
        *args,
        filter_api: Callable[[List[str]], List[str]],
        inference_api,
        target_spec: Dict[str, PropertySpec],
        max_length: int,
        num_samples_per_round: int = 256,
        num_rl_rounds_per_epoch: int = 1,
        warmup_epochs: int = 5,
        reward_invalid: float = -1.0,
        reward_structure_fail: float = -0.5,
        reward_pass_base: float = 0.0,
        reward_pass_max: float = 1.0,
        prior_kl_weight: float = 0.1,
        sampling_temperature: float = 1.0,
        elite_archive_rank: int = 5,
        reward_normalization_num_samples: int = 5000,
        front1_log_path: Optional[str] = None,
        elite_archive_log_path: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            filter_api: 判斷結構是否合規，用法: filtered = filter_api(smiles_list)
            inference_api: 計算分子性質，用法:
                df = inference_api.inference_pipeline(smiles_list, properties=[...])
            target_spec: 每個性質要瞄準的目標，例如
                {"ClogP": PropertySpec(goal="range", low=1.0, high=3.0),
                 "SAScore": PropertySpec(goal="minimize")}
            max_length: 生成分子時的最大序列長度
            num_samples_per_round: 每個 RL round 取樣幾個分子
            num_rl_rounds_per_epoch: 每個 epoch 跑幾個 RL round
            warmup_epochs: 前幾個 epoch 只做原本的監督式訓練，之後才啟動 RL 微調；
                warmup 結束後就完全只靠 RL（loss_pg + prior 正則化）更新模型，不再穿插
                監督式訓練——這跟 REINVENT 原版（Olivecrona et al. 2017）Agent 微調階段
                的做法一致
            reward_invalid: RDKit 都無法解析時的 reward
            reward_structure_fail: 合法但 filter_api 判不合規時的 reward
            reward_pass_base: 合規但絕對 desirability 最差（接近 0）時的 reward
            reward_pass_max: 合規且絕對 desirability 最好（接近 1）時的 reward
            prior_kl_weight: prior regularization 的權重（防止 RL 微調時 mode collapse）
            sampling_temperature: multinomial 採樣的溫度
            elite_archive_rank: 「菁英 archive」保留 pareto rank 0 ~ (elite_archive_rank-1)
                （前幾層）的所有分子，不是保留固定數量——同一層裡不管有幾個分子都會
                全部留著，不會因為超過某個數量就被淘汰。這個 archive **純粹是事後分析
                用的歷史紀錄**（讓使用者知道這個生成器最終找到哪些好分子），完全不影響
                reward 或訓練本身。每個 RL round 結束後，會把這一輪通過結構檢查的分子
                跟 archive 目前的成員合併排序，更新 archive 內容並寫進
                elite_archive_log_path。
                注意：因為是依 rank（不是依數量）篩選，如果同一個 rank 內同時有很多
                分子打平（例如多個性質都是 range 型、達標的分子在該維度上都是同一個
                最低目標值 0），archive 大小可能會持續成長、沒有上限，需要留意（只影響
                elite_archive_log.csv 的檔案大小，不影響訓練）。
            reward_normalization_num_samples: RL 開始（warmup 結束、snapshot prior 之後）
                從 prior model 取樣幾個分子，用來估計每個性質 to_objective() 距離的
                平均值/標準差，當作絕對 reward 的正規化基準（見
                _compute_reward_normalization_stats/_compute_absolute_desirability）。
                這個基準之後整個訓練過程都固定不變，不會隨訓練往下漂移。
            front1_log_path: front 1 分子的 CSV log 路徑，預設存在 save_dir 底下
            elite_archive_log_path: 每一輪更新完 elite_archive 之後，archive 裡目前
                pareto rank 前 5 層的分子（SMILES、各性質原始數值、rank）會整批覆蓋
                寫進這份 CSV log（每次都覆蓋掉前一份，只反映最新狀態），預設存在
                save_dir 底下
        """
        super().__init__(*args, **kwargs)

        self.filter_api = filter_api
        self.inference_api = inference_api
        self.target_spec = target_spec
        self.max_length = max_length
        self.num_samples_per_round = num_samples_per_round
        self.num_rl_rounds_per_epoch = num_rl_rounds_per_epoch
        self.warmup_epochs = warmup_epochs
        self.reward_invalid = reward_invalid
        self.reward_structure_fail = reward_structure_fail
        self.reward_pass_base = reward_pass_base
        self.reward_pass_max = reward_pass_max
        self.prior_kl_weight = prior_kl_weight
        self.sampling_temperature = sampling_temperature

        self.elite_archive_rank = elite_archive_rank
        # canonical_smiles -> {'smiles': str, 'objective': np.ndarray, 'properties': dict}，
        # 純粹是事後分析用的歷史紀錄（見 elite_archive_rank 的說明），不影響 reward
        self.elite_archive: Dict[str, dict] = {}

        self.reward_normalization_num_samples = reward_normalization_num_samples
        # 每個性質 to_objective() 距離的 (mean, std)，只在 RL 開始時從 prior model 取樣算
        # 一次、之後固定不變；None 代表還沒校準，或校準樣本不足而停用（此時所有合規分子
        # 的絕對 desirability 都視為 0，reward 退化成一律拿 reward_pass_base）
        self.reward_norm_mean: Optional[Dict[str, float]] = None
        self.reward_norm_std: Optional[Dict[str, float]] = None

        self.front1_log_path = front1_log_path or os.path.join(self.save_dir, 'front1_log.csv')
        self.elite_archive_log_path = elite_archive_log_path or os.path.join(self.save_dir, 'elite_archive_log.csv')

        self.prior_model = None
        self.rl_history: List[dict] = []

    # ------------------------------------------------------------------
    # Prior snapshot（防止 RL 微調時 mode collapse）
    # ------------------------------------------------------------------
    def _snapshot_prior(self):
        """凍結目前的模型當作 RL 微調的 prior（避免 policy 離它太遠、造成多樣性崩潰）"""
        self.prior_model = copy.deepcopy(self.model)
        self.prior_model.eval()
        for p in self.prior_model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # 絕對 reward：從 prior model 大量取樣，估計每個性質的正規化基準
    # ------------------------------------------------------------------
    def _compute_reward_normalization_stats(self):
        """
        從剛凍結的 prior model 大量取樣，計算每個性質 to_objective() 距離的平均值/標準差，
        當作「這個性質在模型自然產出的分子裡本來就有多少變異」的固定基準，用來把不同單位
        的性質正規化成可比較的 z-score（見 _compute_absolute_desirability）。只在 RL 開始
        前算這一次，之後整個訓練過程都固定不變，不會因為 RL 讓分子越來越集中在目標附近
        就跟著往下漂移。
        """
        print(f"  校準絕對 reward：從 prior model 取樣 {self.reward_normalization_num_samples} 個分子...")
        tokens = self.prior_model.sample(
            num_samples=self.reward_normalization_num_samples,
            max_length=self.max_length,
            start_idx=self.tokenizer.start_idx,
            device=self.device,
            sampling_mode='multinomial',
            temperature=self.sampling_temperature,
        )
        smiles_list = [
            self.tokenizer.decode(tokens[i].detach().cpu().tolist()) for i in range(tokens.size(0))
        ]
        canonical_list = [canonicalize_smiles(s) for s in smiles_list]
        valid_smiles = [s for s, c in zip(smiles_list, canonical_list) if c]
        pass_smiles = self.filter_api(valid_smiles) if valid_smiles else []

        min_calibration_samples = 30
        if len(pass_smiles) < min_calibration_samples:
            print(
                f"  ⚠ 校準樣本不足（{len(pass_smiles)} < {min_calibration_samples}），"
                f"絕對 reward 停用（合規分子一律只拿 reward_pass_base）"
            )
            self.reward_norm_mean = None
            self.reward_norm_std = None
            return

        property_names = list(self.target_spec.keys())
        prop_df = self.inference_api.inference_pipeline(pass_smiles, properties=property_names)

        self.reward_norm_mean = {}
        self.reward_norm_std = {}
        for name in property_names:
            spec = self.target_spec[name]
            raw_objective = np.array([spec.to_objective(v) for v in prop_df[name].tolist()])
            self.reward_norm_mean[name] = float(np.mean(raw_objective))
            self.reward_norm_std[name] = float(max(np.std(raw_objective), 1e-6))

        print(
            f"  ✓ 校準完成（{len(pass_smiles)} 個合規分子）: " +
            ", ".join(
                f"{name}(mean={self.reward_norm_mean[name]:.3f}, std={self.reward_norm_std[name]:.3f})"
                for name in property_names
            )
        )

    def _compute_absolute_desirability(self, prop_df) -> np.ndarray:
        """
        對每個分子，用 reward_norm_mean/std（由 prior model 大量取樣算出、訓練中固定不變）
        把每個性質的 to_objective() 距離轉成 z-score：
            z = (to_objective(value) - mean_ref) / std_ref
            desirability_i = exp(-max(z, 0))
        「比 prior model 自然產出的族群平均還好」(z <= 0) 一律視為接近滿分，只有比平均差
        的部分才開始衰減——對 range 型性質尤其重要：prior model 隨機產出的分子多數會落在
        目標窄區間外，mean_ref 通常是正數，一個真的落在區間內的分子 (to_objective=0) 會
        得到 z<0、被夾到 0、desirability=1，符合預期。

        多個性質的 desirability 用幾何平均合併成單一絕對分數（呼應 REINVENT 官方慣例：
        任一性質嚴重沒達標時整體分數會被拉低，不會被其他性質平均掉），回傳 (0, 1] 區間。

        self.reward_norm_mean 是 None（還沒校準/校準失敗）時，直接回傳全 0（_score_batch
        裡合規分子的 reward 會全部退化成一律拿 reward_pass_base）。
        """
        if self.reward_norm_mean is None:
            return np.zeros(len(prop_df))

        property_names = list(self.target_spec.keys())
        log_desirability = np.zeros(len(prop_df))
        for name in property_names:
            spec = self.target_spec[name]
            raw_objective = np.array([spec.to_objective(v) for v in prop_df[name].tolist()])
            z = (raw_objective - self.reward_norm_mean[name]) / self.reward_norm_std[name]
            log_desirability += -np.clip(z, a_min=0, a_max=None)

        return np.exp(log_desirability / len(property_names))

    # ------------------------------------------------------------------
    # 共用的 log-prob 工具
    # ------------------------------------------------------------------
    def _generation_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """回傳 [N, L] 的 0/1 mask：保留到（且包含）第一個 END token 為止，其餘（含 PAD）都不計入"""
        is_end = (tokens == self.tokenizer.end_idx).long()
        end_cumsum = is_end.cumsum(dim=1)
        keep_mask = (end_cumsum - is_end) == 0  # 第一個 END(含)之前都是 True
        pad_mask = tokens != self.tokenizer.pad_idx
        return (keep_mask & pad_mask).float()

    def _sequence_log_prob(self, model, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        對每個樣本算 sum_t log p(token_t)（只算 mask 內的位置）。
        跟 VAE 版不同：不需要 z，直接把生成的 tokens 當成 teacher forcing 的輸入/目標，
        用同一個模型重新做一次前向傳播即可。
        """
        start_col = torch.full(
            (tokens.size(0), 1), self.tokenizer.start_idx,
            dtype=torch.long, device=tokens.device
        )
        decoder_input = torch.cat([start_col, tokens[:, :-1]], dim=1)

        logits, _ = model(decoder_input)
        log_probs = F.log_softmax(logits, dim=-1)
        token_logp = log_probs.gather(2, tokens.unsqueeze(-1)).squeeze(-1)  # [N, L]

        return (token_logp * mask).sum(dim=1)

    # ------------------------------------------------------------------
    # Reward 評分 / Elite archive（純分析用歷史紀錄）
    # ------------------------------------------------------------------
    def _rank_against_elite_archive(
        self, pass_smiles_ordered: List[str], objective_matrix: np.ndarray
    ) -> np.ndarray:
        """
        把這一輪通過結構檢查的分子跟 elite_archive 目前的成員合併，一起做一次 pareto
        front 排序，回傳「只對應這一輪分子」的 front rank（0 = 最好）。

        純粹用來累積 elite_archive／front1_log 這些事後分析用的歷史紀錄，不影響 reward。
        archive 是空的（訓練剛開始）時，這個函式會自然退化成單純對這一輪分子排序。
        """
        archive_keys = list(self.elite_archive.keys())
        if archive_keys:
            archive_objectives = np.stack([self.elite_archive[k]['objective'] for k in archive_keys])
            merged_objectives = np.concatenate([archive_objectives, objective_matrix], axis=0)
        else:
            merged_objectives = objective_matrix

        merged_ranks = assign_pareto_fronts(merged_objectives)
        return merged_ranks[len(archive_keys):]

    def _update_elite_archive(
        self, pass_smiles_ordered: List[str], objective_matrix: Optional[np.ndarray], prop_df
    ):
        """新分子併入 elite_archive -> 全 archive 重新 pareto 排序 -> 只保留 rank < elite_archive_rank 的分子
        （依名次篩選，不是依數量，同一個 rank 內的分子不會因為超過某個數量就被淘汰）"""
        if objective_matrix is None or not pass_smiles_ordered:
            return

        property_names = list(self.target_spec.keys())
        for local_idx, (smiles, obj_vec) in enumerate(zip(pass_smiles_ordered, objective_matrix)):
            canonical = canonicalize_smiles(smiles)
            if not canonical:
                continue
            existing = self.elite_archive.get(canonical)
            if existing is None or np.all(obj_vec <= existing['objective']):
                properties = {name: prop_df.iloc[local_idx][name] for name in property_names}
                self.elite_archive[canonical] = {'smiles': smiles, 'objective': obj_vec, 'properties': properties}

        # 每次有新分子併入都要重新排序篩選：一個原本在 rank < elite_archive_rank 內的
        # 分子，可能因為這一輪加入了更強的分子而被擠到 rank 變大，需要跟著淘汰
        keys = list(self.elite_archive.keys())
        objective_stack = np.stack([self.elite_archive[k]['objective'] for k in keys])
        ranks = assign_pareto_fronts(objective_stack)
        keep_keys = {keys[i] for i in range(len(keys)) if ranks[i] < self.elite_archive_rank}
        self.elite_archive = {k: v for k, v in self.elite_archive.items() if k in keep_keys}

    def _log_elite_archive_snapshot(self):
        """把 elite_archive 目前保留的全部分子（本來就已經只剩 rank < elite_archive_rank
        的分子，見 _update_elite_archive）依 rank 排序後整批寫進 elite_archive_log_path。
        每次呼叫都會覆蓋掉前一份，只反映 archive 最新狀態（archive 本身每個 RL round
        結束都會更新，所以不需要留存歷史 epoch 紀錄）"""
        if not self.elite_archive:
            return

        property_names = list(self.target_spec.keys())
        keys = list(self.elite_archive.keys())
        objective_stack = np.stack([self.elite_archive[k]['objective'] for k in keys])
        ranks = assign_pareto_fronts(objective_stack)
        order = np.argsort(ranks, kind='stable')

        with open(self.elite_archive_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['pareto_rank', 'smiles'] + property_names)
            for i in order:
                entry = self.elite_archive[keys[i]]
                writer.writerow(
                    [int(ranks[i]), entry['smiles']] +
                    [entry['properties'].get(name, '') for name in property_names]
                )

    def _score_batch(self, smiles_list: List[str]):
        """
        依「合法性 -> filter_api -> 絕對 desirability」算出每個分子的 reward。

        Returns:
            rewards: np.ndarray [N]
            pass_smiles_ordered: 通過結構檢查的 SMILES（用於算性質的順序）
            objective_matrix: 對應 pass_smiles_ordered 的目標值矩陣（越小越好），
                沒有通過結構檢查的分子則為 None
            prop_df: 只對 pass_smiles_ordered 算出的性質 DataFrame（沒有則為 None）
            front_ranks: 對應 pass_smiles_ordered、相對於 elite_archive 的 pareto front
                rank（0 = 最好，純分析用，不影響 reward；沒有則為 None）
            absolute_desirability: 對應 pass_smiles_ordered 的絕對 desirability（見
                _compute_absolute_desirability，沒有則為 None），直接決定 reward
        """
        rewards = np.full(len(smiles_list), self.reward_invalid, dtype=np.float64)

        canonical_list = [canonicalize_smiles(s) for s in smiles_list]
        valid_indices = [i for i, c in enumerate(canonical_list) if c]
        valid_smiles = [smiles_list[i] for i in valid_indices]

        if not valid_smiles:
            return rewards, [], None, None, None, None

        pass_set = set(self.filter_api(valid_smiles))

        pass_indices = []
        pass_smiles_ordered = []
        for i in valid_indices:
            if smiles_list[i] in pass_set:
                pass_indices.append(i)
                pass_smiles_ordered.append(smiles_list[i])
            else:
                rewards[i] = self.reward_structure_fail

        if not pass_smiles_ordered:
            return rewards, [], None, None, None, None

        property_names = list(self.target_spec.keys())
        prop_df = self.inference_api.inference_pipeline(pass_smiles_ordered, properties=property_names)

        objective_matrix = np.zeros((len(pass_smiles_ordered), len(property_names)))
        for col_idx, name in enumerate(property_names):
            spec = self.target_spec[name]
            objective_matrix[:, col_idx] = [spec.to_objective(v) for v in prop_df[name].tolist()]

        # 純分析用：更新/記錄相對於 elite_archive 的名次，不影響下面的 reward 計算
        front_ranks = self._rank_against_elite_archive(pass_smiles_ordered, objective_matrix)

        absolute_desirability = self._compute_absolute_desirability(prop_df)
        for local_idx, global_idx in enumerate(pass_indices):
            rewards[global_idx] = (
                self.reward_pass_base
                + (self.reward_pass_max - self.reward_pass_base) * absolute_desirability[local_idx]
            )

        return rewards, pass_smiles_ordered, objective_matrix, prop_df, front_ranks, absolute_desirability

    # ------------------------------------------------------------------
    # Front 1 監看用的 log
    # ------------------------------------------------------------------
    def _log_front1_molecules(
        self, epoch: int, round_idx: int,
        pass_smiles_ordered: List[str], prop_df, front_ranks
    ) -> List[dict]:
        """挑出這一輪 front 1（front_rank == 0）的分子，印出來並累積寫進 CSV"""
        if prop_df is None or front_ranks is None:
            return []

        property_names = list(self.target_spec.keys())
        front1_rows = []

        for local_idx, smiles in enumerate(pass_smiles_ordered):
            if front_ranks[local_idx] != 0:
                continue
            properties = {name: prop_df.iloc[local_idx][name] for name in property_names}
            front1_rows.append({'smiles': smiles, 'properties': properties})

        if front1_rows:
            write_header = not os.path.exists(self.front1_log_path)
            with open(self.front1_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['epoch', 'round', 'smiles'] + property_names)
                for row in front1_rows:
                    writer.writerow(
                        [epoch, round_idx, row['smiles']] +
                        [row['properties'].get(name, '') for name in property_names]
                    )

        return front1_rows

    # ------------------------------------------------------------------
    # 一個 RL round：採樣 -> 評分 -> policy gradient 更新 -> 更新 elite archive（分析用）
    # ------------------------------------------------------------------
    def run_rl_round(self, epoch: int, round_idx: int) -> Tuple[dict, List[str]]:
        tokens = self.model.sample(
            num_samples=self.num_samples_per_round,
            max_length=self.max_length,
            start_idx=self.tokenizer.start_idx,
            device=self.device,
            sampling_mode='multinomial',
            temperature=self.sampling_temperature
        )

        smiles_list = [
            self.tokenizer.decode(tokens[i].detach().cpu().tolist())
            for i in range(tokens.size(0))
        ]

        rewards, pass_smiles_ordered, objective_matrix, prop_df, front_ranks, absolute_desirability = \
            self._score_batch(smiles_list)
        front1_rows = self._log_front1_molecules(epoch, round_idx, pass_smiles_ordered, prop_df, front_ranks)
        # 純分析用歷史紀錄：跟 reward 計算完全無關，在算完這一輪的 reward 之後才更新，
        # 這樣這一輪的分子才是跟「更新前」的歷史最佳前緣比較
        self._update_elite_archive(pass_smiles_ordered, objective_matrix, prop_df)
        self._log_elite_archive_snapshot()

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        advantage = (rewards_t - rewards_t.mean()).detach()

        mask = self._generation_mask(tokens)

        self.model.train()
        seq_logp = self._sequence_log_prob(self.model, tokens, mask)
        loss_pg = -(advantage * seq_logp).mean()

        loss_prior = torch.zeros((), device=self.device)
        if self.prior_model is not None:
            with torch.no_grad():
                prior_seq_logp = self._sequence_log_prob(self.prior_model, tokens, mask)
            loss_prior = ((seq_logp - prior_seq_logp) ** 2).mean()

        loss_rl = loss_pg + self.prior_kl_weight * loss_prior

        self.optimizer.zero_grad()
        loss_rl.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)
        self.optimizer.step()

        num_valid = int((rewards > self.reward_invalid).sum())
        num_pass = int((rewards >= self.reward_pass_base).sum())
        max_front = int(front_ranks.max()) if front_ranks is not None and len(front_ranks) > 0 else 0
        # 這一輪抽樣分子裡，front rank 0~4（前五層，相對於 elite_archive，純分析用）各自有幾個
        top5_front_counts = (
            [int((front_ranks == r).sum()) for r in range(5)]
            if front_ranks is not None and len(front_ranks) > 0
            else [0, 0, 0, 0, 0]
        )
        mean_absolute_desirability = (
            float(absolute_desirability.mean()) if absolute_desirability is not None and len(absolute_desirability) > 0
            else 0.0
        )

        metrics = {
            'epoch': epoch,
            'round': round_idx,
            'num_sampled': len(smiles_list),
            'num_valid': num_valid,
            'num_pass_filter': num_pass,
            'num_front1': len(front1_rows),
            'max_front': max_front,
            'top5_front_counts': top5_front_counts,
            'elite_archive_size': len(self.elite_archive),
            'mean_reward': float(rewards_t.mean().item()),
            'mean_absolute_desirability': mean_absolute_desirability,
            'loss_pg': float(loss_pg.item()),
            'loss_prior': float(loss_prior.item()),
            'loss_rl': float(loss_rl.item()),
            'front1_rows': front1_rows,
        }
        return metrics, smiles_list

    # ------------------------------------------------------------------
    # 整體訓練迴圈
    # ------------------------------------------------------------------
    def train(self, num_epochs: int):
        """訓練模型：warmup 期間做原本的監督式訓練，warmup 結束後改成純 RL 微調"""
        print(f"開始訓練 {num_epochs} epochs（property-guided RL，warmup={self.warmup_epochs} epochs）...")
        print(f"訓練集大小: {len(self.train_loader.dataset)}")
        print(f"驗證集大小: {len(self.val_loader.dataset)}")
        print(f"設備: {self.device}")
        print(f"Front 1 log: {self.front1_log_path}\n")

        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            current_lr = self.optimizer.param_groups[0]['lr']

            # warmup 期間做監督式訓練；warmup 結束後永遠不再做（純 RL）
            run_supervised = epoch <= self.warmup_epochs
            if run_supervised:
                train_metrics = self.train_epoch(epoch)
                train_loss_str = f"{train_metrics['loss']:.4f}"
                self.history['train_loss'].append(train_metrics['loss'])
            else:
                train_loss_str = "skipped (warmup 已結束，純 RL 微調)"
                self.history['train_loss'].append(None)

            val_metrics = self.validate(epoch)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_perplexity'].append(val_metrics['perplexity'])
            self.history['sample_validity'].append(val_metrics['validity_rate'])
            self.history['lr'].append(current_lr)

            print(f"\n{'='*80}")
            print(f"Epoch {epoch} Summary:")
            print(f"{'='*80}")
            print(f"  訓練資料筆數: {len(self.train_loader.dataset)}")
            print(f"  Train Loss: {train_loss_str}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}, Perplexity: {val_metrics['perplexity']:.4f}")
            print(f"  Sample Validity Rate: {val_metrics['validity_rate']:.2%}")

            if epoch == self.warmup_epochs + 1:
                print(f"\n>>> Warmup 結束，snapshot 目前模型當作 RL 的 prior <<<")
                self._snapshot_prior()
                self._compute_reward_normalization_stats()

            if epoch > self.warmup_epochs:
                for round_idx in range(self.num_rl_rounds_per_epoch):
                    rl_metrics, sample_smiles = self.run_rl_round(epoch, round_idx)
                    self.rl_history.append(rl_metrics)
                    print(
                        f"  [RL round {round_idx + 1}/{self.num_rl_rounds_per_epoch}] "
                        f"sampled={rl_metrics['num_sampled']} valid={rl_metrics['num_valid']} "
                        f"pass_filter={rl_metrics['num_pass_filter']} front1={rl_metrics['num_front1']} "
                        f"max_front={rl_metrics['max_front']} "
                        f"top5_fronts(1-5)={rl_metrics['top5_front_counts']} "
                        f"elite_archive={rl_metrics['elite_archive_size']} "
                        f"mean_reward={rl_metrics['mean_reward']:.4f} "
                        f"abs_desirability={rl_metrics['mean_absolute_desirability']:.4f} "
                        f"loss_pg={rl_metrics['loss_pg']:.4f} loss_prior={rl_metrics['loss_prior']:.4f}"
                    )
                    if rl_metrics['front1_rows']:
                        print(f"    Front 1 分子（本輪最好的一層）:")
                        for row in rl_metrics['front1_rows'][:5]:
                            props_str = ", ".join(f"{k}={v:.3f}" for k, v in row['properties'].items())
                            print(f"      {row['smiles']}  ({props_str})")
                    else:
                        print(f"    範例生成分子: {sample_smiles[:5]}")

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, 'best_model.pt')
                print(f"  ✓ Best model saved!")

            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pt')

            print()


if __name__ == "__main__":
    # 簡單的煙霧測試：用隨機資料跑幾個 epoch，確認整條流程
    # （含「訓練資料只含合規 SMILES」+ front1/elite_archive log）可以跑通
    import torch as _torch
    from torch.utils.data import DataLoader

    from .tokenizer import SmilesTokenizer
    from .dataset import SmilesLMDataset, collate_fn
    from .models.lm import SmilesLM
    from .filters import StructureFilter
    from .properties import PropertyInferenceAPI
    from .seed_utils import set_seed

    set_seed(42)

    smiles_samples = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCC", "c1ccncc1"] * 20

    tokenizer = SmilesTokenizer()
    tokenizer.build_vocab(smiles_samples)

    max_length = 20
    filter_api = StructureFilter()

    base_smiles = filter_api(smiles_samples)
    print(f"訓練資料過濾: {len(smiles_samples)} -> {len(base_smiles)}（只保留合規的 SMILES）")

    dataset = SmilesLMDataset(smiles_list=base_smiles)
    loader = DataLoader(
        dataset, batch_size=8, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length=max_length),
        drop_last=True
    )

    model = SmilesLM(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=32, hidden_dim=64, num_layers=1, pad_idx=tokenizer.pad_idx
    )

    target_spec = {
        "ClogP": PropertySpec(goal="range", low=0.0, high=2.0),
        "SAScore": PropertySpec(goal="minimize"),
    }

    trainer = PropertyGuidedLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=loader,
        val_loader=loader,
        device=_torch.device('cpu'),
        save_dir='/tmp/reinvent_lm_rl_smoke_test',
        filter_api=filter_api,
        inference_api=PropertyInferenceAPI(),
        target_spec=target_spec,
        max_length=max_length,
        num_samples_per_round=32,
        num_rl_rounds_per_epoch=1,
        warmup_epochs=0,
    )

    trainer.train(num_epochs=2)
    print(
        f"✓ PropertyGuidedLMTrainer 煙囪測試通過，"
        f"elite_archive 累積了 {len(trainer.elite_archive)} 個分子（純分析用歷史紀錄），"
        f"front1_log={trainer.front1_log_path} 存在={os.path.exists(trainer.front1_log_path)}，"
        f"elite_archive_log={trainer.elite_archive_log_path} 存在={os.path.exists(trainer.elite_archive_log_path)}"
    )
