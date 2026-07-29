"""
PropertyGuidedTrainer：在標準 VAE 監督式訓練之外，額外用 REINFORCE (policy gradient)
針對「結構規則」與「分子性質目標」對模型做微調。

整體流程（每個 epoch，warmup 結束後才啟動）：
    1. 用目前的模型從 prior 採樣一批分子（multinomial 隨機採樣，取得真正的隨機性）
    2. 用 filter_api 判斷結構是否合規
    3. 對合規的分子用 inference_api 計算性質，依照 target_spec 做 pareto front 排序
    4. 依「是否合規」+「pareto front 名次」組成單一 reward，用 REINFORCE + baseline
       做 policy gradient 更新，並用一份凍結的 prior 模型做正則化，避免生成多樣性崩潰
    5. 這一輪排在 front 1（最好的一層）且通過結構檢查的分子，會被加進一個持續累積的
       「elite buffer」；每個 epoch 額外用 elite buffer 裡的分子做一次標準 VAE 監督式
       訓練（reconstruction + KL），讓這些一直表現很好的分子持續留在訓練資料中
    6. 每一輪的 front 1 分子結構與性質會被印出來，並且累積寫進一個 CSV log，方便訓練
       過程中監看
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
from .dataset import collate_fn
from .models import compute_loss
from .pareto import PropertySpec, assign_pareto_fronts
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class PropertyGuidedTrainer(Trainer):
    """
    繼承自 Trainer，複用 train_epoch / validate / save_checkpoint，
    只在 train() 的 epoch 迴圈中額外插入 property-guided 的 RL 微調回合
    以及 elite buffer 的監督式訓練回合。
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
        elite_buffer_enabled: bool = True,
        elite_buffer_max_size: int = 200,
        elite_batch_size: int = 32,
        elite_train_rounds_per_epoch: int = 1,
        front1_log_path: Optional[str] = None,
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
            elite_buffer_enabled: 是否累積「一直合規且 front 1」的分子，持續加進訓練資料
            elite_buffer_max_size: elite buffer 最多保留幾個分子
            elite_batch_size: 用 elite buffer 訓練時的 batch size
            elite_train_rounds_per_epoch: 每個 epoch 用 elite buffer 訓練幾輪
            front1_log_path: front 1 分子的 CSV log 路徑，預設存在 save_dir 底下
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

        self.elite_buffer_enabled = elite_buffer_enabled
        self.elite_buffer_max_size = elite_buffer_max_size
        self.elite_batch_size = elite_batch_size
        self.elite_train_rounds_per_epoch = elite_train_rounds_per_epoch
        # canonical_smiles -> {smiles, properties, first_epoch, last_epoch, times_seen}
        self.elite_buffer: Dict[str, dict] = {}

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
    # 共用的解碼 / log-prob 工具
    # ------------------------------------------------------------------
    def _decode_logits(self, model, z: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        """依模型種類，給定固定的 z 直接做 teacher forcing 解碼（跳過 encoder）"""
        if self.model_type == 'transformer':
            return model.decode_with_z(z, decoder_input, teacher_forcing=True)
        return model.decoder(z, decoder_input, teacher_forcing=True)

    def _generation_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """回傳 [N, L] 的 0/1 mask：保留到（且包含）第一個 END token 為止，其餘（含 PAD）都不計入"""
        is_end = (tokens == self.tokenizer.end_idx).long()
        end_cumsum = is_end.cumsum(dim=1)
        keep_mask = (end_cumsum - is_end) == 0  # 第一個 END(含)之前都是 True
        pad_mask = tokens != self.tokenizer.pad_idx
        return (keep_mask & pad_mask).float()

    def _sequence_log_prob(self, model, z: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """對每個樣本算 sum_t log p(token_t)（只算 mask 內的位置）"""
        start_col = torch.full(
            (tokens.size(0), 1), self.tokenizer.start_idx,
            dtype=torch.long, device=tokens.device
        )
        decoder_input = torch.cat([start_col, tokens[:, :-1]], dim=1)

        logits = self._decode_logits(model, z, decoder_input)
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
            prop_df: 只對 pass_smiles_ordered 算出的性質 DataFrame（沒有通過結構檢查則為 None）
            front_ranks: 對應 pass_smiles_ordered 的 pareto front rank（0 = 最好，None 表示沒有）
        """
        rewards = np.full(len(smiles_list), self.reward_invalid, dtype=np.float64)

        canonical_list = [canonicalize_smiles(s) for s in smiles_list]
        valid_indices = [i for i, c in enumerate(canonical_list) if c]
        valid_smiles = [smiles_list[i] for i in valid_indices]

        if not valid_smiles:
            return rewards, [], None, None

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
            return rewards, [], None, None

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

        return rewards, pass_smiles_ordered, prop_df, front_ranks

    # ------------------------------------------------------------------
    # Elite buffer：累積「一直合規且 front 1」的分子，讓它們持續留在訓練資料中
    # ------------------------------------------------------------------
    def _update_elite_buffer_and_log(
        self, epoch: int, round_idx: int,
        pass_smiles_ordered: List[str], prop_df, front_ranks
    ) -> List[dict]:
        """把這一輪 front 1（front_rank == 0）的分子記錄下來，並視需要加進 elite buffer"""
        if prop_df is None or front_ranks is None:
            return []

        property_names = list(self.target_spec.keys())
        front1_rows = []

        for local_idx, smiles in enumerate(pass_smiles_ordered):
            if front_ranks[local_idx] != 0:
                continue

            canonical = canonicalize_smiles(smiles)
            if not canonical:
                continue

            properties = {name: prop_df.iloc[local_idx][name] for name in property_names}
            front1_rows.append({'smiles': smiles, 'canonical': canonical, 'properties': properties})

            if self.elite_buffer_enabled:
                if canonical in self.elite_buffer:
                    self.elite_buffer[canonical]['times_seen'] += 1
                    self.elite_buffer[canonical]['last_epoch'] = epoch
                else:
                    self.elite_buffer[canonical] = {
                        'smiles': smiles,
                        'properties': properties,
                        'first_epoch': epoch,
                        'last_epoch': epoch,
                        'times_seen': 1,
                    }

        if self.elite_buffer_enabled and len(self.elite_buffer) > self.elite_buffer_max_size:
            # 超過容量時，優先淘汰最久沒再被抽到、且被抽到次數少的分子
            sorted_keys = sorted(
                self.elite_buffer.keys(),
                key=lambda k: (self.elite_buffer[k]['last_epoch'], self.elite_buffer[k]['times_seen'])
            )
            num_to_remove = len(self.elite_buffer) - self.elite_buffer_max_size
            for key in sorted_keys[:num_to_remove]:
                del self.elite_buffer[key]

        self._log_front1(epoch, round_idx, front1_rows, property_names)
        return front1_rows

    def _log_front1(self, epoch: int, round_idx: int, front1_rows: List[dict], property_names: List[str]):
        """把 front 1 分子的結構與性質累積寫進 CSV，方便訓練過程中監看"""
        if not front1_rows:
            return

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

    def _train_on_elite_buffer(self, epoch: int) -> Optional[dict]:
        """用 elite buffer 裡累積的分子做一次標準 VAE 監督式訓練（reconstruction + KL）"""
        if not self.elite_buffer_enabled or not self.elite_buffer:
            return None

        elite_smiles = [entry['smiles'] for entry in self.elite_buffer.values()]

        # BatchNorm 在 train() 模式下要求每個 batch 至少有 2 筆資料，把會產生單筆 batch 的
        # 情況（例如 elite buffer 剛好剩 1 個分子，或最後一個 batch 只分到 1 筆）濾掉
        batches = [
            elite_smiles[start:start + self.elite_batch_size]
            for start in range(0, len(elite_smiles), self.elite_batch_size)
        ]
        batches = [b for b in batches if len(b) >= 2]
        if not batches:
            return None

        self.model.train()
        total_loss, total_recon, total_kl, num_batches = 0.0, 0.0, 0.0, 0

        for batch_smiles in batches:
            batch_pairs = [(s, s) for s in batch_smiles]
            encoder_input, decoder_input, decoder_target = collate_fn(
                batch_pairs, self.tokenizer, max_length=self.max_length
            )
            encoder_input = encoder_input.to(self.device)
            decoder_input = decoder_input.to(self.device)
            decoder_target = decoder_target.to(self.device)

            if self.model_type == 'transformer':
                src_key_padding_mask, tgt_key_padding_mask, tgt_mask = self.create_masks(
                    encoder_input, decoder_input
                )
                output, mu, logvar = self.model(
                    encoder_input, decoder_input,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    tgt_mask=tgt_mask,
                    teacher_forcing=True
                )
            else:
                output, mu, logvar = self.model(encoder_input, decoder_input, teacher_forcing=True)

            loss, recon_loss, kl_loss = compute_loss(
                output, decoder_target, mu, logvar,
                pad_idx=self.tokenizer.pad_idx, kl_weight=self.current_kl_weight
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)
            self.optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1

        return {
            'elite_buffer_size': len(elite_smiles),
            'loss': total_loss / num_batches,
            'recon': total_recon / num_batches,
            'kl': total_kl / num_batches,
        }

    # ------------------------------------------------------------------
    # 一個 RL round：採樣 -> 評分 -> policy gradient 更新 -> 更新 elite buffer
    # ------------------------------------------------------------------
    def run_rl_round(self, epoch: int, round_idx: int) -> Tuple[dict, List[str]]:
        tokens, z = self.model.sample_stochastic(
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

        rewards, pass_smiles_ordered, prop_df, front_ranks = self._score_batch(smiles_list)
        front1_rows = self._update_elite_buffer_and_log(epoch, round_idx, pass_smiles_ordered, prop_df, front_ranks)

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        advantage = (rewards_t - rewards_t.mean()).detach()

        mask = self._generation_mask(tokens)

        self.model.train()
        seq_logp = self._sequence_log_prob(self.model, z, tokens, mask)
        loss_pg = -(advantage * seq_logp).mean()

        loss_prior = torch.zeros((), device=self.device)
        if self.prior_model is not None:
            with torch.no_grad():
                prior_seq_logp = self._sequence_log_prob(self.prior_model, z, tokens, mask)
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
            'elite_buffer_size': len(self.elite_buffer),
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
        """訓練模型：每個 epoch 先做原本的監督式訓練，warmup 結束後再做 RL 微調 + elite buffer 訓練"""
        print(f"開始訓練 {num_epochs} epochs（property-guided RL，warmup={self.warmup_epochs} epochs）...")
        print(f"訓練集大小: {len(self.train_loader.dataset)}")
        print(f"驗證集大小: {len(self.val_loader.dataset)}")
        print(f"設備: {self.device}")
        print(f"Front 1 log: {self.front1_log_path}\n")

        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            self.current_kl_weight = self.get_kl_weight(epoch)
            current_lr = self.optimizer.param_groups[0]['lr']

            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon'].append(train_metrics['recon'])
            self.history['train_kl'].append(train_metrics['kl'])

            val_metrics = self.validate(epoch)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_recon'].append(val_metrics['recon'])
            self.history['val_kl'].append(val_metrics['kl'])
            self.history['val_recon_acc'].append(val_metrics['recon_acc'])

            self.history['kl_weight'].append(self.current_kl_weight)
            self.history['lr'].append(current_lr)

            print(f"\n{'='*80}")
            print(f"Epoch {epoch} Summary:")
            print(f"{'='*80}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Recon: {train_metrics['recon']:.4f}, KL: {train_metrics['kl']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Recon: {val_metrics['recon']:.4f}, KL: {val_metrics['kl']:.4f}")
            print(f"  Val Recon Acc: {val_metrics['recon_acc']:.2%}")

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
                        f"elite_buffer={rl_metrics['elite_buffer_size']} "
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

                for _ in range(self.elite_train_rounds_per_epoch):
                    elite_metrics = self._train_on_elite_buffer(epoch)
                    if elite_metrics:
                        print(
                            f"  [Elite buffer] size={elite_metrics['elite_buffer_size']} "
                            f"loss={elite_metrics['loss']:.4f} recon={elite_metrics['recon']:.4f} "
                            f"kl={elite_metrics['kl']:.4f}"
                        )

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, 'best_model.pt')
                print(f"  ✓ Best model saved!")

            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pt')

            print()


if __name__ == "__main__":
    # 簡單的煙霧測試：用隨機資料跑一個 epoch，確認整條流程（含 elite buffer / front1 log）可以跑通
    import torch as _torch
    from torch.utils.data import DataLoader

    from .tokenizer import SmilesTokenizer
    from .models import GRUVAE
    from .filters import StructureFilter
    from .properties import PropertyInferenceAPI

    smiles_samples = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCC", "c1ccncc1"] * 20

    tokenizer = SmilesTokenizer()
    tokenizer.build_vocab(smiles_samples)

    max_length = 20

    class _ListDataset(_torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            s = self.items[idx]
            return s, s

    dataset = _ListDataset(smiles_samples)
    loader = DataLoader(
        dataset, batch_size=8, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length=max_length)
    )

    model = GRUVAE(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=32, hidden_dim=64, latent_dim=16, num_layers=1
    )

    target_spec = {
        "ClogP": PropertySpec(goal="range", low=0.0, high=2.0),
        "SAScore": PropertySpec(goal="minimize"),
    }

    trainer = PropertyGuidedTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=loader,
        val_loader=loader,
        device=_torch.device('cpu'),
        model_type='gru',
        save_dir='/tmp/gruvae_rl_smoke_test',
        filter_api=StructureFilter(),
        inference_api=PropertyInferenceAPI(),
        target_spec=target_spec,
        max_length=max_length,
        num_samples_per_round=32,
        num_rl_rounds_per_epoch=1,
        warmup_epochs=0,
        elite_buffer_max_size=10,
    )

    trainer.train(num_epochs=2)
    print(f"✓ PropertyGuidedTrainer 煙霧測試通過，elite buffer 累積了 {len(trainer.elite_buffer)} 個分子")
