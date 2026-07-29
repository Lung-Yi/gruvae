"""
分子性質目標設定 (PropertySpec) 與 pareto front 排序工具
只做 non-dominated sorting 分出 front rank，同一個 front 內不再比較優劣。
"""

from dataclasses import dataclass
from typing import List, Optional

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


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """a 是否支配 b：a 在所有目標都 <= b，且至少一個目標嚴格 < b（目標皆為越小越好）"""
    return bool(np.all(a <= b) and np.any(a < b))


def assign_pareto_fronts(objective_matrix: np.ndarray) -> np.ndarray:
    """
    Fast non-dominated sorting（NSGA-II 的第一步驟，只做 front rank，不做 crowding distance）

    Args:
        objective_matrix: [N, M]，每一欄都已轉成「越小越好」

    Returns:
        front_ranks: [N]，每個樣本所屬的 front index（0 = 最好的一層）
    """
    n = objective_matrix.shape[0]
    domination_count = np.zeros(n, dtype=int)   # 有幾個樣本支配我
    dominated_sets: List[List[int]] = [[] for _ in range(n)]  # 我支配了誰

    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(objective_matrix[i], objective_matrix[j]):
                dominated_sets[i].append(j)
                domination_count[j] += 1
            elif _dominates(objective_matrix[j], objective_matrix[i]):
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
    print("✓ pareto front 測試通過")
