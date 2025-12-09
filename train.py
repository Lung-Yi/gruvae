"""
GRU-VAE 訓練腳本
支援 teacher forcing、validation 和分子重建/採樣可視化
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import random

from tokenizer import SmilesTokenizer, canonicalize_smiles
from dataset import get_dataloader, SmilesVAEDataset, collate_fn
from model import GRUVAE, compute_loss


class Trainer:
    """GRU-VAE 訓練器"""

    def __init__(
        self,
        model: GRUVAE,
        tokenizer: SmilesTokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-3,
        kl_weight_start: float = 0.0,
        kl_weight_end: float = 1.0,
        kl_anneal_epochs: int = 10,
        save_dir: str = './checkpoints'
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.lr = lr

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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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

    def validate(self, epoch: int, num_reconstruct: int = 10, num_sample: int = 10) -> dict:
        """驗證並顯示重建和採樣結果"""
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
            if epoch % 5 == 0:
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


def main():
    # 設置隨機種子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}\n")

    # 建立 tokenizer
    print("建立 tokenizer...")
    tokenizer = SmilesTokenizer()

    # 從 CSV 讀取並建立詞彙表
    df = pd.read_csv('./data/train_processed.csv')
    smiles_list = df['smiles'].tolist()
    tokenizer.build_vocab(smiles_list)

    # 保存 tokenizer
    tokenizer.save('./checkpoints/tokenizer.json')

    # 分割訓練/驗證集
    print("\n建立 Dataset...")
    dataset = SmilesVAEDataset('./data/train_processed.csv', tokenizer, max_length=100)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"訓練集: {len(train_dataset)}, 驗證集: {len(val_dataset)}")

    # 建立 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length=100),
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length=100),
        num_workers=0
    )

    # 建立模型
    print("\n建立模型...")
    model = GRUVAE(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        latent_dim=128,
        num_layers=2,
        dropout=0.1
    )

    print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")

    # 建立訓練器
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=1e-3,
        kl_weight_start=0.0,    # KL weight 從 0 開始
        kl_weight_end=0.1,       # 逐漸增加到 0.1
        kl_anneal_epochs=10,     # 在 10 個 epochs 內線性增加
        save_dir='./checkpoints'
    )

    # 開始訓練
    trainer.train(num_epochs=20)


if __name__ == "__main__":
    main()
