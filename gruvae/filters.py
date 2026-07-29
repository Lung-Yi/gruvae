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
        desired_smarts: Optional[List[str]] = None,
        require_all_desired: bool = True,
    ):
        """
        Args:
            forbidden_smarts: 黑名單，分子只要符合其中任何一個 pattern 就會被濾掉
            desired_smarts: 白名單/必要結構，分子要符合這裡的規則才會通過
            require_all_desired: desired_smarts 要「全部都要符合」(True，預設)
                還是「符合其中一個就好」(False，適合列出多個可接受的替代骨架)
        """
        self.max_ring_size = max_ring_size
        self.max_heavy_atoms = max_heavy_atoms
        self.min_heavy_atoms = min_heavy_atoms
        self.require_all_desired = require_all_desired

        forbidden_list = forbidden_smarts if forbidden_smarts is not None else DEFAULT_FORBIDDEN_SMARTS
        self.forbidden_patterns = self._compile_smarts(forbidden_list)
        self.desired_patterns = self._compile_smarts(desired_smarts or [])

    @staticmethod
    def _compile_smarts(smarts_list: List[str]):
        patterns = []
        for smarts in smarts_list:
            patt = Chem.MolFromSmarts(smarts)
            if patt is not None:
                patterns.append(patt)
        return patterns

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

        if self.desired_patterns:
            matches = (mol.HasSubstructMatch(patt) for patt in self.desired_patterns)
            if self.require_all_desired:
                if not all(matches):
                    return False
            else:
                if not any(matches):
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

    # desired_smarts 示範：要求分子一定要含有苯環
    aromatic_only_filter = StructureFilter(desired_smarts=["c1ccccc1"])
    test_smiles_2 = ["CCO", "c1ccccc1CCO", "CCCCCC"]
    result_2 = aromatic_only_filter(test_smiles_2)
    print("\n只接受含苯環的分子:", test_smiles_2)
    print("通過:", result_2)
