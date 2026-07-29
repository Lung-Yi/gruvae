"""
結構規則過濾器 (filter_api 的預設簡單實作)
用法: filtered_smiles_list = filter_api(smiles_list)
"""

from typing import List, Optional

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# 一些常見、示範性質的「不合規」官能基/結構 SMARTS（可依實際需求增減）
DEFAULT_FORBIDDEN_SMARTS = [
    "[N;R0]=[N;R0]=[N;R0]",   # 疊氮基 azide（不穩定，示範用）
    "[#6]=[#6]=[#6]",          # allene（合成困難，示範用）
    "[OX2,SX2][OX2,SX2]",      # 過氧化物/過硫化物鍵結（不穩定，示範用）
]


class StructureFilter:
    """
    根據簡單的結構規則過濾生成出來的 SMILES。

    這是一個「先求可用」的預設實作，使用者可以依實際需求換掉或擴充規則
    （例如换成 PAINS filter、公司內部的合成可行性規則等）。
    """

    def __init__(
        self,
        max_ring_size: int = 8,
        max_heavy_atoms: int = 60,
        min_heavy_atoms: int = 2,
        forbidden_smarts: Optional[List[str]] = None,
    ):
        self.max_ring_size = max_ring_size
        self.max_heavy_atoms = max_heavy_atoms
        self.min_heavy_atoms = min_heavy_atoms

        smarts_list = forbidden_smarts if forbidden_smarts is not None else DEFAULT_FORBIDDEN_SMARTS
        self.forbidden_patterns = []
        for smarts in smarts_list:
            patt = Chem.MolFromSmarts(smarts)
            if patt is not None:
                self.forbidden_patterns.append(patt)

    def _passes(self, smiles: str) -> bool:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        num_heavy = mol.GetNumHeavyAtoms()
        if num_heavy < self.min_heavy_atoms or num_heavy > self.max_heavy_atoms:
            return False

        ring_info = mol.GetRingInfo()
        if any(len(ring) > self.max_ring_size for ring in ring_info.AtomRings()):
            return False

        for patt in self.forbidden_patterns:
            if mol.HasSubstructMatch(patt):
                return False

        return True

    def __call__(self, smiles_list: List[str]) -> List[str]:
        """回傳通過結構規則檢查的 SMILES 子集合"""
        return [smiles for smiles in smiles_list if self._passes(smiles)]


if __name__ == "__main__":
    filter_api = StructureFilter()
    test_smiles = [
        "CCO",                     # 通過
        "c1ccccc1",                # 通過
        "not_a_smiles",            # 無效，濾掉
        "C" * 70,                  # 太大，濾掉
    ]
    result = filter_api(test_smiles)
    print("輸入:", test_smiles)
    print("通過:", result)
