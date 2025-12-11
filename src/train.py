"""
GRU-VAE 訓練腳本
支援 teacher forcing、validation 和分子重建/採樣可視化
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import random

# 嘗試相對導入，如果失敗則使用絕對導入（方便從根目錄運行）
try:
    from .data.tokenizer import SmilesTokenizer, canonicalize_smiles
    from .data.dataset import get_dataloader, SmilesVAEDataset, collate_fn
    from .models.gru_vae import GRUVAE, compute_loss
    from .models.transformer_vae import TransformerVAE
except ImportError:
    from data.tokenizer import SmilesTokenizer, canonicalize_smiles
    from data.dataset import get_dataloader, SmilesVAEDataset, collate_fn
    from models.gru_vae import GRUVAE, compute_loss
    from models.transformer_vae import TransformerVAE

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class Trainer:
    """VAE 訓練器（支援 GRU 和 Transformer）"""

    def __init__(
        self,
        model,  # GRUVAE 或 TransformerVAE
        tokenizer: SmilesTokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        model_type: str = 'gru',  # 'gru' 或 'transformer'
        lr: float = 1e-3,
        kl_weight_start: float = 0.0,
        kl_weight_end: float = 1.0,
        kl_anneal_epochs: int = 10,
        save_dir: str = './checkpoints',
        grad_clip_max_norm: float = 1.0,
        save_interval: int = 5,
        num_reconstruct: int = 10,
        num_sample: int = 10
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_type = model_type.lower()
        self.save_dir = save_dir
        self.lr = lr
        self.grad_clip_max_norm = grad_clip_max_norm
        self.save_interval = save_interval
        self.num_reconstruct = num_reconstruct
        self.num_sample = num_sample

        # KL annealing 參數
        self.kl_weight_start = kl_weight_start
        self.kl_weight_end = kl_weight_end
        self.kl_anneal_epochs = kl_anneal_epochs
        self.current_kl_weight = kl_weight_start

        # 優化器
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # 創建保存目錄
        os.makedirs(save_dir, exist_ok=True)

        # 訓練歷史
        self.history = {
            'train_loss': [],
            'train_recon': [],
            'train_kl': [],
            'val_loss': [],
            'val_recon': [],
            'val_kl': [],
            'val_recon_acc': [],
            'kl_weight': [],
            'lr': []
        }

    def create_masks(self, encoder_input, decoder_input):
        """
        為 Transformer 模型創建 masks

        Args:
            encoder_input: [batch_size, seq_len]
            decoder_input: [batch_size, seq_len]

        Returns:
            src_key_padding_mask, tgt_key_padding_mask, tgt_mask
        """
        if self.model_type != 'transformer':
            return None, None, None

        # Padding mask: True 表示該位置是 padding，需要被 mask
        src_key_padding_mask = (encoder_input == self.tokenizer.pad_idx)
        tgt_key_padding_mask = (decoder_input == self.tokenizer.pad_idx)

        # Causal mask: 防止 decoder 看到未來的 token
        seq_len = decoder_input.size(1)
        tgt_mask = self.model.generate_square_subsequent_mask(seq_len).to(self.device)

        return src_key_padding_mask, tgt_key_padding_mask, tgt_mask

    def get_kl_weight(self, epoch: int) -> float:
        """計算當前 epoch 的 KL weight（線性 annealing）"""
        if epoch > self.kl_anneal_epochs:
            return self.kl_weight_end

        # 線性增長
        progress = epoch / self.kl_anneal_epochs
        kl_weight = self.kl_weight_start + (self.kl_weight_end - self.kl_weight_start) * progress
        return kl_weight

    def train_epoch(self, epoch: int) -> dict:
        """訓練一個 epoch"""
        self.model.train()

        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for encoder_input, decoder_input, decoder_target in pbar:
            # 移動到設備
            encoder_input = encoder_input.to(self.device)
            decoder_input = decoder_input.to(self.device)
            decoder_target = decoder_target.to(self.device)

            # 前向傳播（使用 teacher forcing）
            if self.model_type == 'transformer':
                # Transformer 需要額外的 masks
                src_key_padding_mask, tgt_key_padding_mask, tgt_mask = self.create_masks(
                    encoder_input, decoder_input
                )
                output, mu, logvar = self.model(
                    encoder_input,
                    decoder_input,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    tgt_mask=tgt_mask,
                    teacher_forcing=True
                )
            else:
                # GRU 模型
                output, mu, logvar = self.model(
                    encoder_input,
                    decoder_input,
                    teacher_forcing=True
                )

            # 計算損失（使用當前的 KL weight）
            loss, recon_loss, kl_loss = compute_loss(
                output,
                decoder_target,
                mu,
                logvar,
                pad_idx=self.tokenizer.pad_idx,
                kl_weight=self.current_kl_weight
            )

            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)
            self.optimizer.step()

            # 統計
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1

            # 更新進度條
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}'
            })

        # 計算平均值
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches

        return {
            'loss': avg_loss,
            'recon': avg_recon,
            'kl': avg_kl
        }

    def validate(self, epoch: int) -> dict:
        """驗證並顯示重建和採樣結果"""
        num_reconstruct = self.num_reconstruct
        num_sample = self.num_sample
        self.model.eval()

        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0

        # 用於計算 reconstruction accuracy
        total_correct = 0
        total_samples = 0

        # 收集一些樣本用於展示
        reconstruct_examples = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for batch_idx, (encoder_input, decoder_input, decoder_target) in enumerate(pbar):
                # 移動到設備
                encoder_input = encoder_input.to(self.device)
                decoder_input = decoder_input.to(self.device)
                decoder_target = decoder_target.to(self.device)

                # 前向傳播（不使用 teacher forcing）
                if self.model_type == 'transformer':
                    # Transformer 需要額外的 masks
                    # 注意：validation 時不使用 teacher forcing，但仍需要 masks
                    src_key_padding_mask = (encoder_input == self.tokenizer.pad_idx)
                    # 對於非 teacher forcing 模式，不需要 tgt_key_padding_mask 和 tgt_mask
                    # 因為模型內部會自回歸生成
                    output, mu, logvar = self.model(
                        encoder_input,
                        decoder_input,
                        src_key_padding_mask=src_key_padding_mask,
                        tgt_key_padding_mask=None,
                        tgt_mask=None,
                        teacher_forcing=False
                    )
                else:
                    # GRU 模型
                    output, mu, logvar = self.model(
                        encoder_input,
                        decoder_input,
                        teacher_forcing=False
                    )

                # 計算損失（使用當前的 KL weight）
                loss, recon_loss, kl_loss = compute_loss(
                    output,
                    decoder_target,
                    mu,
                    logvar,
                    pad_idx=self.tokenizer.pad_idx,
                    kl_weight=self.current_kl_weight
                )

                # 統計
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
                num_batches += 1

                # 計算 reconstruction accuracy
                predictions = output.argmax(dim=-1)  # [batch_size, seq_len]

                for i in range(encoder_input.size(0)):
                    target_smiles = self.tokenizer.decode(decoder_target[i].cpu().tolist())
                    pred_smiles = self.tokenizer.decode(predictions[i].cpu().tolist())

                    # Canonicalize 兩者進行比較
                    target_canonical = canonicalize_smiles(target_smiles)
                    pred_canonical = canonicalize_smiles(pred_smiles)

                    if target_canonical == pred_canonical:
                        total_correct += 1
                    total_samples += 1

                # 收集重建樣本用於展示
                if len(reconstruct_examples) < num_reconstruct:
                    for i in range(min(encoder_input.size(0), num_reconstruct - len(reconstruct_examples))):
                        target_smiles = self.tokenizer.decode(decoder_target[i].cpu().tolist())
                        pred_smiles = self.tokenizer.decode(predictions[i].cpu().tolist())

                        # Canonicalize 用於比較
                        target_canonical = canonicalize_smiles(target_smiles)
                        pred_canonical = canonicalize_smiles(pred_smiles)

                        reconstruct_examples.append({
                            'target': target_smiles,
                            'reconstructed': pred_smiles,
                            'target_canonical': target_canonical,
                            'pred_canonical': pred_canonical
                        })

                # 更新進度條
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}'
                })

        # 計算平均值
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches
        recon_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        # 顯示重建結果
        print(f"\n{'='*80}")
        print(f"Reconstruction Examples (Epoch {epoch}):")
        print(f"{'='*80}")
        for i, example in enumerate(reconstruct_examples[:num_reconstruct], 1):
            print(f"\n[{i}] Target:        {example['target']}")
            print(f"    Reconstructed: {example['reconstructed']}")
            print(f"    Target (Can):  {example['target_canonical']}")
            print(f"    Recon (Can):   {example['pred_canonical']}")
            match = "✓" if example['target_canonical'] == example['pred_canonical'] else "✗"
            print(f"    Match: {match}")

        # 從潛在空間採樣
        print(f"\n{'='*80}")
        print(f"Sampled Molecules from Latent Space (Epoch {epoch}):")
        print(f"{'='*80}")

        max_length = decoder_input.size(1)
        samples = self.model.sample(
            num_samples=num_sample,
            max_length=max_length,
            start_idx=self.tokenizer.start_idx,
            device=self.device
        )

        for i in range(num_sample):
            smiles = self.tokenizer.decode(samples[i].cpu().tolist())
            print(f"[{i+1}] {smiles}")

        print(f"{'='*80}\n")

        return {
            'loss': avg_loss,
            'recon': avg_recon,
            'kl': avg_kl,
            'recon_acc': recon_accuracy
        }

    def train(self, num_epochs: int):
        """訓練模型"""
        print(f"開始訓練 {num_epochs} epochs...")
        print(f"訓練集大小: {len(self.train_loader.dataset)}")
        print(f"驗證集大小: {len(self.val_loader.dataset)}")
        print(f"設備: {self.device}")
        print(f"初始學習率: {self.lr}")
        print(f"KL Weight 範圍: {self.kl_weight_start} -> {self.kl_weight_end} (over {self.kl_anneal_epochs} epochs)\n")

        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            # 更新 KL weight (annealing)
            self.current_kl_weight = self.get_kl_weight(epoch)

            # 獲取當前學習率
            current_lr = self.optimizer.param_groups[0]['lr']

            # 訓練
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon'].append(train_metrics['recon'])
            self.history['train_kl'].append(train_metrics['kl'])

            # 驗證
            val_metrics = self.validate(epoch)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_recon'].append(val_metrics['recon'])
            self.history['val_kl'].append(val_metrics['kl'])
            self.history['val_recon_acc'].append(val_metrics['recon_acc'])

            # 記錄 KL weight 和 learning rate
            self.history['kl_weight'].append(self.current_kl_weight)
            self.history['lr'].append(current_lr)

            # 輸出統計
            print(f"\n{'='*80}")
            print(f"Epoch {epoch} Summary:")
            print(f"{'='*80}")
            print(f"  Learning Rate:     {current_lr:.6f}")
            print(f"  KL Weight:         {self.current_kl_weight:.4f}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Recon: {train_metrics['recon']:.4f}, KL: {train_metrics['kl']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Recon: {val_metrics['recon']:.4f}, KL: {val_metrics['kl']:.4f}")
            print(f"  Val Recon Acc:     {val_metrics['recon_acc']:.2%} ({int(val_metrics['recon_acc'] * len(self.val_loader.dataset))}/{len(self.val_loader.dataset)})")

            # 保存最佳模型
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, 'best_model.pt')
                print(f"  ✓ Best model saved!")

            # 定期保存
            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pt')

            print()

    def save_checkpoint(self, epoch: int, filename: str):
        """保存檢查點"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)


def load_config(config_path):
    """載入 YAML 配置檔案"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path='train.yaml'):
    # 載入配置
    print(f"載入配置檔案: {config_path}")
    config = load_config(config_path)
    print(f"配置載入成功!\n")

    # 設置隨機種子
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 設備
    use_cuda = config['device']['use_cuda']
    device = torch.device('cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu')
    print(f"使用設備: {device}\n")

    # 建立 tokenizer
    print("建立 tokenizer...")
    tokenizer = SmilesTokenizer()

    # 從 CSV 讀取並建立詞彙表
    train_csv = config['data']['train_csv']
    df = pd.read_csv(train_csv)
    smiles_list = df['smiles'].tolist()
    tokenizer.build_vocab(smiles_list)

    # 保存 tokenizer
    save_dir = config['checkpoint']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(os.path.join(save_dir, 'tokenizer.json'))

    # 分割訓練/驗證集
    print("\n建立 Dataset...")
    max_length = config['data']['max_length']
    dataset = SmilesVAEDataset(train_csv, tokenizer, max_length=max_length)

    train_split = config['data']['train_split']
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"訓練集: {len(train_dataset)}, 驗證集: {len(val_dataset)}")

    # 建立 DataLoader
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length=max_length),
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length=max_length),
        num_workers=num_workers
    )

    # 建立模型
    print("\n建立模型...")
    model_config = config['model']
    model_type = model_config.get('model_type', 'gru').lower()

    if model_type == 'transformer':
        print("使用 Transformer VAE 模型")
        model = TransformerVAE(
            vocab_size=tokenizer.vocab_size,
            d_model=model_config.get('d_model', 512),
            nhead=model_config.get('nhead', 8),
            num_encoder_layers=model_config.get('num_encoder_layers', 8),
            num_decoder_layers=model_config.get('num_decoder_layers', 6),
            dim_feedforward=model_config.get('dim_feedforward', 2048),
            latent_dim=model_config.get('latent_dim', 128),
            dropout=model_config.get('dropout', 0.1),
            max_len=model_config.get('max_len', 128),
            pad_idx=tokenizer.pad_idx,
            start_idx=tokenizer.start_idx,
            end_idx=tokenizer.end_idx
        )
    elif model_type == 'gru':
        print("使用 GRU VAE 模型")
        model = GRUVAE(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=model_config['embedding_dim'],
            hidden_dim=model_config['hidden_dim'],
            latent_dim=model_config['latent_dim'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout'],
            bidirectional=model_config['bidirectional'],
        )
    else:
        raise ValueError(f"不支援的模型類型: {model_type}，請使用 'gru' 或 'transformer'")

    print(f"模型類型: {model_type.upper()}")
    print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")

    # 建立訓練器
    training_config = config['training']
    validation_config = config['validation']
    checkpoint_config = config['checkpoint']

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        model_type=model_type,
        lr=training_config['learning_rate'],
        kl_weight_start=training_config['kl_weight_start'],
        kl_weight_end=training_config['kl_weight_end'],
        kl_anneal_epochs=training_config['kl_anneal_epochs'],
        grad_clip_max_norm=training_config['grad_clip_max_norm'],
        save_dir=save_dir,
        save_interval=checkpoint_config['save_interval'],
        num_reconstruct=validation_config['num_reconstruct'],
        num_sample=validation_config['num_sample']
    )

    # 開始訓練
    num_epochs = training_config['num_epochs']
    trainer.train(num_epochs=num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GRU-VAE 訓練腳本')
    parser.add_argument(
        '--config',
        type=str,
        default='train.yaml',
        help='訓練配置檔案路徑 (default: train.yaml)'
    )
    args = parser.parse_args()

    main(config_path=args.config)
