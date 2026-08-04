"""
集中管理可復現性 (reproducibility) 相關的 seeding 工具。

涵蓋 torch / numpy / random / RDKit 四個隨機源，以及（可選的）GPU 上 cuDNN 的
determinism 設定。`reinvent_lm/` 底下需要固定隨機性的地方（training.py 的
main()、DataLoader 的 worker_init_fn、MoleculeGenerator/SmilesLM 的 sample 系列
方法）都透過這個模組共用同一套邏輯，避免各處各自 inline 一份不完整的版本。
"""

import random
from typing import Optional

import numpy as np
import torch
from rdkit import rdBase


def set_seed(seed: int, deterministic_cuda: bool = True) -> None:
    """
    把 random / numpy / torch（CPU + CUDA）/ RDKit 的全域 RNG 都種到同一個 seed。

    注意：`rdBase.SeedRandomNumberGenerator()` 實測過**不足以**讓 RDKit 的
    `doRandom=True`/`MolToRandomSmilesVect` 在同一個 process 裡重複呼叫時保持
    可重現（RDKit 內部似乎還有一份不會被這個 API 重置的殘留狀態，同一個 seed
    在該 process 已經呼叫過其他隨機 SMILES 之後就會產生不同結果）。這裡仍然呼叫
    它，當作對其他未審查過的 RDKit 隨機性來源的 best-effort 保底，但
    `reinvent_lm.tokenizer.randomize_smiles(seed=...)` 真正的可重現性保證來自
    它自己內部用的本地 `random.Random(seed)` 實例（見該函式的說明），不依賴這裡。

    deterministic_cuda=True 時，額外把 cuDNN 設成 deterministic 模式（關掉
    benchmark 自動調參）並開啟 `torch.use_deterministic_algorithms(warn_only=True)`
    ——這是 PyTorch 官方建議的 best-effort 做法：極少數沒有 deterministic CUDA
    實作的 op 只會印警告，不會直接噴例外。GPU 上的 RNN backward 在少數 cuDNN
    版本下仍可能無法保證 100% bit-exact，這是已知限制，這裡不強行解決。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rdBase.SeedRandomNumberGenerator(seed)

    if deterministic_cuda and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def derive_seed(*values: int) -> int:
    """
    把多個整數值（例如 base_seed、epoch、item index）混合成一個 32-bit deterministic
    seed，給需要「每個項目各自一個獨立、可重現 seed」的場合使用（例如
    Chem.MolToRandomSmilesVect 的 randomSeed 參數）。用 numpy 的 SeedSequence 做混合，
    而不是 Python 內建 hash()（str/bytes 的 hash 預設會被 PYTHONHASHSEED 隨機化，
    不保證跨 process/跨執行一致）。
    """
    return int(np.random.SeedSequence(list(values)).generate_state(1)[0])


def seed_worker(worker_id: int) -> None:
    """
    給 `DataLoader(num_workers>0, worker_init_fn=seed_worker)` 用。

    PyTorch 只會自動幫 worker 自己的 torch RNG 依 `base_seed + worker_id` 重新
    seed，不會處理 random/numpy/RDKit——這三個如果不額外處理，所有 worker 會在
    fork 時繼承完全相同的一份全域 RNG 狀態副本。這裡用 `torch.initial_seed()`
    （worker 行程裡已經是 PyTorch 幫忙算好的 per-worker 值）重新種好其餘三個
    RNG，讓每個 worker 各自獨立。目前 reinvent_lm 的 randomize_smiles 已經改用
    逐項顯式 seed + 本地 random.Random 實例（見 tokenizer.py/dataset.py 的
    derive_seed 用法），完全不依賴這裡重新種好的全域 RNG，這個函式純粹是
    防禦性補強，防止未來有其他程式碼在 worker 裡用到全域隨機性。
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    rdBase.SeedRandomNumberGenerator(int(worker_seed))
