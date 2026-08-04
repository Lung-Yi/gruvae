"""
SmilesLM 訓練邏輯
純 autoregressive 語言模型：teacher forcing 訓練、自回歸生成驗證（sample validity rate）
"""

import os
import yaml
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import random

from .tokenizer import SmilesTokenizer
from .dataset import SmilesLMDataset, DynamicSmilesLMDataset, collate_fn
from .models.lm import SmilesLM
from .seed_utils import set_seed, seed_worker
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class Trainer:
    """SmilesLM 訓練器（純 MLE 預訓練）"""

    def __init__(
        self,
        model: SmilesLM,
        tokenizer: SmilesTokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-3,
        save_dir: str = './checkpoints',
        grad_clip_max_norm: float = 1.0,
        save_interval: int = 5,
        num_sample: int = 10,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.lr = lr
        self.grad_clip_max_norm = grad_clip_max_norm
        self.save_interval = save_interval
        self.num_sample = num_sample

        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        os.makedirs(save_dir, exist_ok=True)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_perplexity': [],
            'sample_validity': [],
            'lr': [],
        }

    def train_epoch(self, epoch: int) -> dict:
        """訓練一個 epoch"""
        if hasattr(self.train_loader.dataset, 'set_epoch'):
            self.train_loader.dataset.set_epoch(epoch)
        self.model.train()

        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for input_seq, target_seq in pbar:
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)

            logits, _ = self.model(input_seq)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_seq.reshape(-1),
                ignore_index=self.tokenizer.pad_idx,
                reduction='mean',
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return {'loss': total_loss / num_batches}

    def validate(self, epoch: int) -> dict:
        """驗證：算 val loss/perplexity，並用自回歸生成算 sample validity rate"""
        self.model.eval()

        total_loss = 0
        num_batches = 0
        max_length = None

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for input_seq, target_seq in pbar:
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                max_length = input_seq.size(1)

                logits, _ = self.model(input_seq)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_seq.reshape(-1),
                    ignore_index=self.tokenizer.pad_idx,
                    reduction='mean',
                )

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        perplexity = float(np.exp(min(avg_loss, 20)))

        # 自回歸生成一批分子，算 validity rate（取代 VAE 版的 reconstruction accuracy）
        samples = self.model.sample(
            num_samples=self.num_sample,
            max_length=max_length,
            start_idx=self.tokenizer.start_idx,
            device=self.device,
            sampling_mode='multinomial',
            temperature=1.0,
        )

        sampled_smiles = [self.tokenizer.decode(samples[i].cpu().tolist()) for i in range(self.num_sample)]
        num_valid = sum(1 for s in sampled_smiles if Chem.MolFromSmiles(s) is not None)
        validity_rate = num_valid / self.num_sample if self.num_sample > 0 else 0.0

        print(f"\n{'='*80}")
        print(f"Sampled Molecules (Epoch {epoch}):")
        print(f"{'='*80}")
        for i, smiles in enumerate(sampled_smiles, 1):
            valid = Chem.MolFromSmiles(smiles) is not None
            print(f"[{i}] {'✓' if valid else '✗'} {smiles}")
        print(f"Validity rate: {validity_rate:.2%}")
        print(f"{'='*80}\n")

        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'validity_rate': validity_rate,
        }

    def train(self, num_epochs: int):
        """訓練模型"""
        print(f"開始訓練 {num_epochs} epochs...")
        print(f"訓練集大小: {len(self.train_loader.dataset)}")
        print(f"驗證集大小: {len(self.val_loader.dataset)}")
        print(f"設備: {self.device}")
        print(f"初始學習率: {self.lr}\n")

        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            current_lr = self.optimizer.param_groups[0]['lr']

            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])

            val_metrics = self.validate(epoch)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_perplexity'].append(val_metrics['perplexity'])
            self.history['sample_validity'].append(val_metrics['validity_rate'])
            self.history['lr'].append(current_lr)

            print(f"\n{'='*80}")
            print(f"Epoch {epoch} Summary:")
            print(f"{'='*80}")
            print(f"  Learning Rate:  {current_lr:.6f}")
            print(f"  Train Loss:     {train_metrics['loss']:.4f}")
            print(f"  Val Loss:       {val_metrics['loss']:.4f}")
            print(f"  Val Perplexity: {val_metrics['perplexity']:.4f}")
            print(f"  Sample Validity Rate: {val_metrics['validity_rate']:.2%}")

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, 'best_model.pt')
                print(f"  ✓ Best model saved!")

            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pt')

            print()

    def save_checkpoint(self, epoch: int, filename: str):
        """保存檢查點"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)


def load_config(config_path):
    """載入 YAML 配置檔案"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path='configs/train_reinvent.yaml'):
    print(f"載入配置檔案: {config_path}")
    config = load_config(config_path)
    print(f"配置載入成功!\n")

    seed = config['seed']
    deterministic_cuda = config['device'].get('deterministic_cuda', True)
    set_seed(seed, deterministic_cuda=deterministic_cuda)

    use_cuda = config['device']['use_cuda']
    device = torch.device('cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu')
    print(f"使用設備: {device}\n")

    checkpoint_config = config['checkpoint']
    save_dir = checkpoint_config['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    load_from = checkpoint_config.get('load_from')

    train_csv = config['data']['train_csv']
    df = pd.read_csv(train_csv)
    smiles_list = df['smiles'].tolist()

    tokenizer = SmilesTokenizer()
    pretrained_tokenizer_path = None
    if load_from:
        candidate = os.path.join(os.path.dirname(load_from), 'tokenizer.json')
        if os.path.exists(candidate):
            pretrained_tokenizer_path = candidate

    if pretrained_tokenizer_path:
        print(f"載入預訓練模型旁的 tokenizer: {pretrained_tokenizer_path}")
        tokenizer.load(pretrained_tokenizer_path)
    else:
        print("建立 tokenizer...")
        tokenizer.build_vocab(smiles_list)

    tokenizer.save(os.path.join(save_dir, 'tokenizer.json'))

    print("\n建立 Dataset...")
    max_length = config['data']['max_length']
    train_split = config['data']['train_split']
    # 訓練時是否每次取用都用 RDKit 重新隨機化 SMILES 書寫法（SMILES enumeration，
    # 一種資料增強：同一個分子在不同 epoch 看到的書寫法不同，能增加生成多樣性、
    # 降低對特定 canonical 寫法的過擬合）；驗證集固定用 canonical 寫法，
    # 確保 val loss/perplexity 每個 epoch 的量測基準一致、可以互相比較
    randomize_training_smiles = config['data'].get('randomize_smiles', True)

    property_guided_config = config.get('property_guided', {})
    pg_enabled = property_guided_config.get('enabled', False)

    filter_api = None
    if pg_enabled:
        filter_api = build_structure_filter(property_guided_config)
        num_before = len(smiles_list)
        filtered_smiles_list = filter_api(smiles_list)
        print(
            f"訓練資料結構過濾: {num_before} -> {len(filtered_smiles_list)} "
            f"(只保留通過 filter_api 的 SMILES)"
        )

        shuffled_smiles = filtered_smiles_list[:]
        random.shuffle(shuffled_smiles)
        train_size = int(train_split * len(shuffled_smiles))
        train_dataset = DynamicSmilesLMDataset(
            shuffled_smiles[:train_size], randomize=randomize_training_smiles, seed=seed
        )
        val_dataset = DynamicSmilesLMDataset(shuffled_smiles[train_size:], randomize=False, seed=seed)
    else:
        # 直接切 smiles list 分別建兩個 Dataset（而不是用 random_split 包同一個 Dataset 實例），
        # 這樣訓練集/驗證集才能各自套用不同的 randomize 設定
        shuffled_smiles = smiles_list[:]
        random.shuffle(shuffled_smiles)
        train_size = int(train_split * len(shuffled_smiles))
        train_dataset = SmilesLMDataset(
            smiles_list=shuffled_smiles[:train_size], randomize=randomize_training_smiles, seed=seed
        )
        val_dataset = SmilesLMDataset(smiles_list=shuffled_smiles[train_size:], randomize=False, seed=seed)

    print(f"訓練集: {len(train_dataset)}, 驗證集: {len(val_dataset)}")

    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']

    # persistent_workers 明確鎖在 False：RL 動態訓練池（DynamicSmilesLMDataset.set_dynamic_smiles）
    # 會在 epoch 之間直接改動 Dataset 物件的內容，若 worker 跨 epoch 存活（persistent_workers=True），
    # 已經 fork 出去的 worker 會看不到這個更新，動態池同步會被悄悄破壞；False（預設行為）時每個
    # epoch 重新 fork worker，才能保證看到 main process 最新的 dataset 狀態。
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length=max_length),
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=False,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length=max_length),
        num_workers=num_workers,
        persistent_workers=False,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )

    print("\n建立模型...")
    model_config = config['model']
    model = SmilesLM(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=model_config['embedding_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout'],
        pad_idx=tokenizer.pad_idx,
    )
    print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")

    if load_from:
        print(f"\n載入預訓練模型權重: {load_from}")
        pretrained_checkpoint = torch.load(load_from, map_location=device)
        model.load_state_dict(pretrained_checkpoint['model_state_dict'])
        print("✓ 預訓練權重載入完成")

    training_config = config['training']
    validation_config = config['validation']

    common_trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=training_config['learning_rate'],
        grad_clip_max_norm=training_config['grad_clip_max_norm'],
        save_dir=save_dir,
        save_interval=checkpoint_config['save_interval'],
        num_sample=validation_config['num_sample'],
    )

    if pg_enabled:
        print("\n啟用 Property-Guided RL 訓練模式")
        trainer = build_property_guided_trainer(
            common_trainer_kwargs, property_guided_config,
            default_max_length=max_length, filter_api=filter_api,
        )
    else:
        trainer = Trainer(**common_trainer_kwargs)

    num_epochs = training_config['num_epochs']
    trainer.train(num_epochs=num_epochs)


def build_structure_filter(pg_config: dict):
    """依 config 的 `property_guided.structure_filter` 區塊建立 filter_api"""
    from .filters import StructureFilter

    filter_config = pg_config.get('structure_filter', {})
    filter_kwargs = {
        k: v for k, v in dict(
            max_ring_size=filter_config.get('max_ring_size', 8),
            max_heavy_atoms=filter_config.get('max_heavy_atoms', 60),
            min_heavy_atoms=filter_config.get('min_heavy_atoms', 2),
            forbidden_smarts=filter_config.get('forbidden_smarts'),
            desired_smarts=filter_config.get('desired_smarts'),
            require_all_desired=filter_config.get('require_all_desired', True),
        ).items() if v is not None
    }
    return StructureFilter(**filter_kwargs)


def build_property_guided_trainer(common_trainer_kwargs: dict, pg_config: dict, default_max_length: int, filter_api):
    """依 config 的 `property_guided` 區塊建立 PropertyGuidedLMTrainer"""
    from .rl_trainer import PropertyGuidedLMTrainer
    from .properties import PropertyInferenceAPI
    from .pareto import PropertySpec

    inference_api = PropertyInferenceAPI()

    target_spec = {}
    for prop_name, spec_config in pg_config['target_spec'].items():
        target_spec[prop_name] = PropertySpec(
            goal=spec_config['goal'],
            low=spec_config.get('low'),
            high=spec_config.get('high'),
        )

    dynamic_pool_config = pg_config.get('dynamic_training_data', {})

    return PropertyGuidedLMTrainer(
        **common_trainer_kwargs,
        filter_api=filter_api,
        inference_api=inference_api,
        target_spec=target_spec,
        max_length=pg_config.get('max_length', default_max_length),
        num_samples_per_round=pg_config.get('num_samples_per_round', 256),
        num_rl_rounds_per_epoch=pg_config.get('num_rl_rounds_per_epoch', 1),
        warmup_epochs=pg_config.get('warmup_epochs', 5),
        reward_invalid=pg_config.get('reward_invalid', -1.0),
        reward_structure_fail=pg_config.get('reward_structure_fail', -0.5),
        reward_pass_base=pg_config.get('reward_pass_base', 0.0),
        reward_pass_max=pg_config.get('reward_pass_max', 1.0),
        prior_kl_weight=pg_config.get('prior_kl_weight', 0.1),
        sampling_temperature=pg_config.get('sampling_temperature', 1.0),
        dynamic_pool_enabled=dynamic_pool_config.get('enabled', True),
        dynamic_pool_max_size=dynamic_pool_config.get('max_pool_size', 500),
        elite_archive_rank=pg_config.get('elite_archive_rank', 5),
        reward_scale_power=pg_config.get('reward_scale_power', 1.0),
        archive_stagnation_patience_epochs=pg_config.get('archive_stagnation_patience_epochs', 3),
        archive_stagnation_watch_rank=pg_config.get('archive_stagnation_watch_rank', 1),
        archive_prune_keep_rank=pg_config.get('archive_prune_keep_rank', 1),
        front1_log_path=pg_config.get('front1_log_path'),
        elite_archive_log_path=pg_config.get('elite_archive_log_path'),
        supervised_training_during_rl=pg_config.get('supervised_training_during_rl', True),
    )
