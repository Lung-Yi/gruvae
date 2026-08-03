"""
分子性質目標設定 (PropertySpec) 與 pareto front 排序工具
只做 non-dominated sorting 分出 front rank，同一個 front 內不再比較優劣。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PropertySpec:
    """
    描述一個分子性質要瞄準的目標。

    goal:
        'maximize' - 越大越好
        'minimize' - 越小越好
        'range'    - 落在 [low, high] 區間內就算好（不細分區間內的優劣），
                     超出區間則以超出的距離當作懲罰
    """
    goal: str
    low: Optional[float] = None
    high: Optional[float] = None

    def __post_init__(self):
        if self.goal not in ('maximize', 'minimize', 'range'):
            raise ValueError(f"未知的 goal: {self.goal}，請用 'maximize'/'minimize'/'range'")
        if self.goal == 'range' and (self.low is None or self.high is None):
            raise ValueError("goal='range' 時必須同時提供 low 與 high")

    def to_objective(self, value: float) -> float:
        """把性質值轉成『越小越好』的目標值，供 pareto 排序使用"""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return float('inf')

        if self.goal == 'maximize':
            return -value
        elif self.goal == 'minimize':
            return value
        else:  # range
            if value < self.low:
                return self.low - value
            if value > self.high:
                return value - self.high
            return 0.0


def _pairwise_dominates(objective_matrix: np.ndarray, chunk_size: Optional[int] = None) -> np.ndarray:
    """
    向量化計算 [N, N] 的支配關係矩陣：dominates[i, j] = True 代表 i 支配 j
    （i 在所有目標都 <= j，且至少一個目標嚴格 < j；目標皆為越小越好）

    支援任意目標維度數 M。用 chunk 沿著 i 這一軸分批算，避免一次性配置
    [N, N, M] 的中間陣列（那個才是真正吃記憶體、也拖慢速度的地方）；
    每個 chunk 算完就沿著 M 那一軸 reduce 掉，只留下 [chunk, N] 的布林矩陣。
    """
    n, m = objective_matrix.shape

    if chunk_size is None:
        # 讓每個 chunk 的中間陣列 (chunk_size * n * m) 大約落在 2000 萬個元素以內，
        # 同時兼顧向量化的效益（chunk 不要切得太碎）
        budget = 20_000_000
        chunk_size = max(1, min(n, budget // max(n * m, 1)))

    dominates_matrix = np.zeros((n, n), dtype=bool)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        obj_chunk = objective_matrix[start:end, None, :]   # [c, 1, M]
        obj_all = objective_matrix[None, :, :]             # [1, N, M]
        leq = np.all(obj_chunk <= obj_all, axis=2)          # [c, N]
        lt = np.any(obj_chunk < obj_all, axis=2)            # [c, N]
        dominates_matrix[start:end] = leq & lt

    np.fill_diagonal(dominates_matrix, False)  # 自己不支配自己
    return dominates_matrix


def assign_pareto_fronts(objective_matrix: np.ndarray, chunk_size: Optional[int] = None) -> np.ndarray:
    """
    Fast non-dominated sorting（NSGA-II 的第一步驟，只做 front rank，不做 crowding distance）。
    向量化實作，支援任意數量的目標維度 (M)，且不論 M 有多少，
    逐點比較的部分都用 numpy 矩陣運算取代 Python 巢狀迴圈 + 逐點函式呼叫，
    N 上到幾千筆時可以從幾十秒降到零點幾秒。

    Args:
        objective_matrix: [N, M]，每一欄都已轉成「越小越好」
        chunk_size: 向量化計算支配矩陣時，沿 N 這一軸分批的大小；
            不填會依 N 和 M 自動選一個兼顧記憶體與速度的值

    Returns:
        front_ranks: [N]，每個樣本所屬的 front index（0 = 最好的一層）
    """
    objective_matrix = np.asarray(objective_matrix, dtype=np.float64)
    if objective_matrix.ndim == 1:
        objective_matrix = objective_matrix[:, None]

    n = objective_matrix.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)

    dominates_matrix = _pairwise_dominates(objective_matrix, chunk_size=chunk_size)
    domination_count = dominates_matrix.sum(axis=0)  # 每個點被幾個人支配

    front_ranks = np.full(n, -1, dtype=int)
    remaining_mask = np.ones(n, dtype=bool)
    front_idx = 0

    while remaining_mask.any():
        current_front_mask = remaining_mask & (domination_count == 0)

        if not current_front_mask.any():
            # 理論上不會發生，除非目標值有 NaN/inf 造成比較異常；保底避免無窮迴圈
            front_ranks[remaining_mask] = front_idx
            break

        front_ranks[current_front_mask] = front_idx

        # 這一層的點被移除後，他們原本支配的其他點，支配數要跟著扣掉
        reduction = dominates_matrix[current_front_mask][:, remaining_mask].sum(axis=0)
        domination_count[remaining_mask] -= reduction

        remaining_mask &= ~current_front_mask
        front_idx += 1

    return front_ranks


def _assign_pareto_fronts_naive(objective_matrix: np.ndarray) -> np.ndarray:
    """未向量化的參考實作（純 Python 巢狀迴圈），只用來驗證向量化版本結果一致"""
    def dominates(a, b):
        return bool(np.all(a <= b) and np.any(a < b))

    n = objective_matrix.shape[0]
    domination_count = np.zeros(n, dtype=int)
    dominated_sets = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(objective_matrix[i], objective_matrix[j]):
                dominated_sets[i].append(j)
                domination_count[j] += 1
            elif dominates(objective_matrix[j], objective_matrix[i]):
                dominated_sets[j].append(i)
                domination_count[i] += 1

    front_ranks = np.full(n, -1, dtype=int)
    current_front = [i for i in range(n) if domination_count[i] == 0]
    front_idx = 0
    while current_front:
        next_front = []
        for i in current_front:
            front_ranks[i] = front_idx
            for j in dominated_sets[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front = next_front
        front_idx += 1
    return front_ranks


if __name__ == "__main__":
    import time

    # 簡單測試：4 個點，目標皆為越小越好
    points = np.array([
        [1.0, 4.0],  # 非支配
        [2.0, 2.0],  # 非支配
        [4.0, 1.0],  # 非支配
        [3.0, 3.0],  # 被前三者支配
    ])
    ranks = assign_pareto_fronts(points)
    print("Front ranks:", ranks)
    assert ranks[0] == 0 and ranks[1] == 0 and ranks[2] == 0
    assert ranks[3] > 0
    print("✓ pareto front 基本測試通過")

    # 正確性測試：向量化版本 vs 未向量化版本，在不同維度數下結果要一致
    print("\n正確性驗證（向量化 vs 樸素實作）：")
    rng = np.random.default_rng(42)
    for n, m in [(50, 2), (80, 3), (120, 5), (60, 8)]:
        data = rng.random((n, m))
        fast_ranks = assign_pareto_fronts(data)
        naive_ranks = _assign_pareto_fronts_naive(data)
        assert np.array_equal(fast_ranks, naive_ranks), f"n={n}, m={m} 結果不一致！"
        print(f"  n={n:4d}, m={m}: 一致 ✓")

    # 效能測試：多維度 (M) 下的速度，確認 M 增加時不會又變回逐點迴圈的速度
    print("\n效能測試（不同 N / M 組合）：")
    for n in [1024, 2048, 5000]:
        for m in [2, 5, 10]:
            data = rng.random((n, m))
            t0 = time.perf_counter()
            assign_pareto_fronts(data)
            dt = time.perf_counter() - t0
            print(f"  n={n:5d}, m={m:2d}  time={dt:7.3f}s")

    print("\n✓ pareto front 向量化實作驗證通過")
