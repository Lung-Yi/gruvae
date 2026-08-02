"""
PropertyGuidedLMTrainer：在標準 SmilesLM 預訓練之外，額外用 REINFORCE (policy gradient)
針對「結構規則」與「分子性質目標」對模型做微調。跟 gruvae.rl_trainer.PropertyGuidedTrainer
邏輯完全對應，差別是這裡的底層模型是純 autoregressive GRU（沒有 encoder/z），
所以不需要 model_type 分支，也不需要在 log-prob 重算時傳遞/固定 z。

整體流程（每個 epoch，warmup 結束後才啟動）：
    1. 用目前的模型從 BOS 開始自回歸採樣一批分子（multinomial 隨機採樣，取得真正的隨機性）
    2. 用 filter_api 判斷結構是否合規
    3. 對合規的分子用 inference_api 計算性質，依照 target_spec 做 pareto front 排序
    4. 依「是否合規」+「pareto front 名次」組成單一 reward，用 REINFORCE + baseline
       做 policy gradient 更新，並用一份凍結的 prior 模型做正則化，避免生成多樣性崩潰
    5. 通過結構檢查的分子會被拿去更新一個持續累積的「動態訓練池 (dynamic pool)」：
       新分子併入池子，池子滿了就對整個池子重新做 pareto front 排序，淘汰最差的一批，
       讓池子始終保留「目前為止看過最好的一批分子」。這個池子會被直接同步進
       train_loader 的 Dataset，變成「真的存在於訓練資料中」，下個 epoch 的監督式訓練
       就會用到它們。
    6. 每一輪 pareto front 1（最好的一層）分子的結構與性質會被印出來，並累積寫進 CSV
       log，方便訓練過程中監看

注意：這個 Trainer 假設 train_loader.dataset 是 `reinvent_lm.dataset.DynamicSmilesLMDataset`
（或任何有 `set_dynamic_smiles(smiles_list)` 方法的 Dataset），這樣「動態訓練池」才能
真的反映到下一輪的監督式訓練上；如果不是，動態訓練池機制會自動停用並印出警告。
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
    只在 train() 的 epoch 迴圈中額外插入 property-guided 的 RL 微調回合，
    並維護一個會同步進訓練資料的動態分子池。
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
        dynamic_pool_enabled: bool = True,
        dynamic_pool_max_size: int = 500,
        front1_log_path: Optional[str] = None,
        supervised_training_during_rl: bool = True,
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
            warmup_epochs: 前幾個 epoch 只做原本的監督式訓練，之後才啟動 RL 微調
            reward_invalid: RDKit 都無法解析時的 reward
            reward_structure_fail: 合法但 filter_api 判不合規時的 reward
            reward_pass_base: 合規但 pareto front 最差時的 reward
            reward_pass_max: 合規且 pareto front 最好 (front 1 / rank 0) 時的 reward
            prior_kl_weight: prior regularization 的權重（防止 RL 微調時 mode collapse）
            sampling_temperature: multinomial 採樣的溫度
            dynamic_pool_enabled: 是否把合規分子動態加入/淘汰進訓練資料
            dynamic_pool_max_size: 動態訓練池最多保留幾個分子（用全池 pareto front 排序淘汰）
            front1_log_path: front 1 分子的 CSV log 路徑，預設存在 save_dir 底下
            supervised_training_during_rl: warmup 結束、RL 開始之後，是否每個 epoch 仍要
                做一次監督式訓練（train_epoch）。預設 True 維持原本行為；設 False 時，
                warmup 結束後就只靠 RL（loss_pg + prior 正則化）更新模型，不再穿插監督式
                訓練——這跟 REINVENT 原版（Olivecrona et al. 2017）Agent 微調階段的做法一致。
                注意：warmup 期間（epoch <= warmup_epochs）不受這個參數影響，一定會做監督式
                訓練，因為這正是 warmup 存在的目的。設 False 時，動態訓練池仍會照常累積/淘汰，
                但不再被拿去訓練模型（因為監督式訓練被跳過了），如果不需要它可以考慮同時把
                dynamic_pool_enabled 設 False 省一點計算。
        """
        super().__init__(*args, **kwargs)
        self.supervised_training_during_rl = supervised_training_during_rl

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

        self.dynamic_pool_enabled = dynamic_pool_enabled
        self.dynamic_pool_max_size = dynamic_pool_max_size
        # canonical_smiles -> {'smiles': str, 'objective': np.ndarray}
        self.dynamic_pool: Dict[str, dict] = {}

        if self.dynamic_pool_enabled and not hasattr(self.train_loader.dataset, 'set_dynamic_smiles'):
            print(
                "⚠ train_loader.dataset 不支援 set_dynamic_smiles()，"
                "dynamic_pool 機制已自動停用（請改用 reinvent_lm.dataset.DynamicSmilesLMDataset）"
            )
            self.dynamic_pool_enabled = False

        self.front1_log_path = front1_log_path or os.path.join(self.save_dir, 'front1_log.csv')

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
    # Reward / pareto front 評分
    # ------------------------------------------------------------------
    def _score_batch(self, smiles_list: List[str]):
        """
        依「合法性 -> filter_api -> pareto front」算出每個分子的 reward

        Returns:
            rewards: np.ndarray [N]
            pass_smiles_ordered: 通過結構檢查的 SMILES（用於算性質的順序）
            objective_matrix: 對應 pass_smiles_ordered 的目標值矩陣（越小越好，跨 round 可比較），
                沒有通過結構檢查的分子則為 None
            prop_df: 只對 pass_smiles_ordered 算出的性質 DataFrame（沒有則為 None）
            front_ranks: 對應 pass_smiles_ordered 的 pareto front rank（0 = 最好，沒有則為 None）
        """
        rewards = np.full(len(smiles_list), self.reward_invalid, dtype=np.float64)

        canonical_list = [canonicalize_smiles(s) for s in smiles_list]
        valid_indices = [i for i, c in enumerate(canonical_list) if c]
        valid_smiles = [smiles_list[i] for i in valid_indices]

        if not valid_smiles:
            return rewards, [], None, None, None

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
            return rewards, [], None, None, None

        property_names = list(self.target_spec.keys())
        prop_df = self.inference_api.inference_pipeline(pass_smiles_ordered, properties=property_names)

        objective_matrix = np.zeros((len(pass_smiles_ordered), len(property_names)))
        for col_idx, name in enumerate(property_names):
            spec = self.target_spec[name]
            objective_matrix[:, col_idx] = [spec.to_objective(v) for v in prop_df[name].tolist()]

        front_ranks = assign_pareto_fronts(objective_matrix)
        max_front = int(front_ranks.max()) if len(front_ranks) > 0 else 0

        for local_idx, global_idx in enumerate(pass_indices):
            scale = 1.0 - (front_ranks[local_idx] / max_front) if max_front > 0 else 1.0
            rewards[global_idx] = self.reward_pass_base + (self.reward_pass_max - self.reward_pass_base) * scale

        return rewards, pass_smiles_ordered, objective_matrix, prop_df, front_ranks

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
    # 動態訓練池：新分子併入 -> 全池重新 pareto 排序 -> 淘汰最差的一批 -> 同步進訓練資料
    # ------------------------------------------------------------------
    def _update_dynamic_pool(self, pass_smiles_ordered: List[str], objective_matrix: Optional[np.ndarray]):
        if not self.dynamic_pool_enabled or objective_matrix is None or not pass_smiles_ordered:
            return

        for smiles, obj_vec in zip(pass_smiles_ordered, objective_matrix):
            canonical = canonicalize_smiles(smiles)
            if not canonical:
                continue
            # 同一個分子重複出現時，只保留目標值比較好的那次紀錄
            existing = self.dynamic_pool.get(canonical)
            if existing is None or np.all(obj_vec <= existing['objective']):
                self.dynamic_pool[canonical] = {'smiles': smiles, 'objective': obj_vec}

        if len(self.dynamic_pool) > self.dynamic_pool_max_size:
            keys = list(self.dynamic_pool.keys())
            objective_stack = np.stack([self.dynamic_pool[k]['objective'] for k in keys])
            ranks = assign_pareto_fronts(objective_stack)
            # 依 front rank 排序，只保留最好的 dynamic_pool_max_size 個（同一 front 內不細分優劣）
            order = np.argsort(ranks, kind='stable')
            keep_keys = {keys[i] for i in order[:self.dynamic_pool_max_size]}
            self.dynamic_pool = {k: v for k, v in self.dynamic_pool.items() if k in keep_keys}

        self.train_loader.dataset.set_dynamic_smiles(
            [entry['smiles'] for entry in self.dynamic_pool.values()]
        )

    # ------------------------------------------------------------------
    # 一個 RL round：採樣 -> 評分 -> policy gradient 更新 -> 更新動態訓練池
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

        rewards, pass_smiles_ordered, objective_matrix, prop_df, front_ranks = self._score_batch(smiles_list)
        front1_rows = self._log_front1_molecules(epoch, round_idx, pass_smiles_ordered, prop_df, front_ranks)
        self._update_dynamic_pool(pass_smiles_ordered, objective_matrix)

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

        metrics = {
            'epoch': epoch,
            'round': round_idx,
            'num_sampled': len(smiles_list),
            'num_valid': num_valid,
            'num_pass_filter': num_pass,
            'num_front1': len(front1_rows),
            'dynamic_pool_size': len(self.dynamic_pool),
            'mean_reward': float(rewards_t.mean().item()),
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
        """訓練模型：每個 epoch 先做原本的監督式訓練，warmup 結束後再做 RL 微調"""
        print(f"開始訓練 {num_epochs} epochs（property-guided RL，warmup={self.warmup_epochs} epochs）...")
        print(f"訓練集大小: {len(self.train_loader.dataset)}（含動態訓練池: {self.dynamic_pool_enabled}）")
        print(f"驗證集大小: {len(self.val_loader.dataset)}")
        print(f"設備: {self.device}")
        if not self.supervised_training_during_rl:
            print("注意: supervised_training_during_rl=False，warmup 結束後將只靠 RL 更新模型")
        print(f"Front 1 log: {self.front1_log_path}\n")

        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            current_lr = self.optimizer.param_groups[0]['lr']

            # warmup 期間一定要做監督式訓練（這正是 warmup 存在的目的）；
            # warmup 結束、RL 開始之後，是否繼續做監督式訓練由 supervised_training_during_rl 決定
            run_supervised = self.supervised_training_during_rl or epoch <= self.warmup_epochs
            if run_supervised:
                train_metrics = self.train_epoch(epoch)
                train_loss_str = f"{train_metrics['loss']:.4f}"
                self.history['train_loss'].append(train_metrics['loss'])
            else:
                train_loss_str = "skipped (supervised_training_during_rl=False)"
                self.history['train_loss'].append(None)

            val_metrics = self.validate(epoch)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_perplexity'].append(val_metrics['perplexity'])
            self.history['sample_validity'].append(val_metrics['validity_rate'])
            self.history['lr'].append(current_lr)

            print(f"\n{'='*80}")
            print(f"Epoch {epoch} Summary:")
            print(f"{'='*80}")
            print(f"  訓練資料筆數（含動態池）: {len(self.train_loader.dataset)}")
            print(f"  Train Loss: {train_loss_str}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}, Perplexity: {val_metrics['perplexity']:.4f}")
            print(f"  Sample Validity Rate: {val_metrics['validity_rate']:.2%}")

            if epoch == self.warmup_epochs + 1:
                print(f"\n>>> Warmup 結束，snapshot 目前模型當作 RL 的 prior <<<")
                self._snapshot_prior()

            if epoch > self.warmup_epochs:
                for round_idx in range(self.num_rl_rounds_per_epoch):
                    rl_metrics, sample_smiles = self.run_rl_round(epoch, round_idx)
                    self.rl_history.append(rl_metrics)
                    print(
                        f"  [RL round {round_idx + 1}/{self.num_rl_rounds_per_epoch}] "
                        f"sampled={rl_metrics['num_sampled']} valid={rl_metrics['num_valid']} "
                        f"pass_filter={rl_metrics['num_pass_filter']} front1={rl_metrics['num_front1']} "
                        f"dynamic_pool={rl_metrics['dynamic_pool_size']} "
                        f"mean_reward={rl_metrics['mean_reward']:.4f} "
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
    # （含「訓練資料只含合規 SMILES」+「動態訓練池新增/淘汰」+ front1 log）可以跑通
    import torch as _torch
    from torch.utils.data import DataLoader

    from .tokenizer import SmilesTokenizer
    from .dataset import DynamicSmilesLMDataset, collate_fn
    from .models.lm import SmilesLM
    from .filters import StructureFilter
    from .properties import PropertyInferenceAPI

    smiles_samples = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCC", "c1ccncc1"] * 20

    tokenizer = SmilesTokenizer()
    tokenizer.build_vocab(smiles_samples)

    max_length = 20
    filter_api = StructureFilter()

    base_smiles = filter_api(smiles_samples)
    print(f"訓練資料過濾: {len(smiles_samples)} -> {len(base_smiles)}（只保留合規的 SMILES）")

    dataset = DynamicSmilesLMDataset(base_smiles)
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
        dynamic_pool_max_size=10,
    )

    trainer.train(num_epochs=2)
    print(
        f"✓ PropertyGuidedLMTrainer 煙霧測試通過，"
        f"動態訓練池累積了 {len(trainer.dynamic_pool)} 個分子，"
        f"目前訓練資料筆數: {len(trainer.train_loader.dataset)}"
    )
