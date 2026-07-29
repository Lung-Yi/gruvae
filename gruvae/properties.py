"""
分子性質推論介面 (inference_api 的預設簡單實作)
用法: property_dataframe = inference_api.inference_pipeline(smiles_list, properties=["ClogP", "SAScore"])
"""

from typing import Callable, Dict, List

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

try:
    import os
    import sys
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer as _sascorer
    _HAS_SASCORER = True
except ImportError:
    _HAS_SASCORER = False


def _sa_score_fallback(mol) -> float:
    """
    簡化的合成可及性 (synthetic accessibility) 啟發式估計，
    只在環境沒有 RDKit contrib 的 sascorer 時當作佔位替代方案。
    數值大致落在 1(容易合成) ~ 10(困難合成)，僅供示範，不建議直接用於正式篩選。
    """
    num_rings = Descriptors.RingCount(mol)
    num_heavy = mol.GetNumHeavyAtoms()
    num_hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))
    score = 1.0 + 0.15 * num_rings + 0.05 * num_hetero + 0.01 * num_heavy
    return float(min(max(score, 1.0), 10.0))


class PropertyInferenceAPI:
    """
    分子性質推論的預設簡單實作，用 RDKit 內建描述子計算。
    之後若要換成真正的 ML/ADMET 模型，只要保持 inference_pipeline 的介面不變即可。
    """

    def __init__(self):
        self._property_funcs: Dict[str, Callable] = {
            "ClogP": lambda mol: Crippen.MolLogP(mol),
            "SAScore": self._sa_score,
            "MolWt": lambda mol: Descriptors.MolWt(mol),
            "QED": lambda mol: Descriptors.qed(mol),
            "TPSA": lambda mol: Descriptors.TPSA(mol),
        }

    def _sa_score(self, mol) -> float:
        if _HAS_SASCORER:
            return float(_sascorer.calculateScore(mol))
        return _sa_score_fallback(mol)

    def register_property(self, name: str, func: Callable) -> None:
        """讓使用者自行擴充性質計算函式，func 簽名為 func(rdkit_mol) -> float"""
        self._property_funcs[name] = func

    def inference_pipeline(self, smiles_list: List[str], properties: List[str]) -> pd.DataFrame:
        """
        計算一批 SMILES 的指定性質

        Args:
            smiles_list: 輸入的 SMILES 列表
            properties: 要計算的性質名稱列表，例如 ["ClogP", "SAScore"]

        Returns:
            DataFrame，欄位為 ['smiles'] + properties，無法解析或計算失敗的值為 NaN
        """
        unknown = [p for p in properties if p not in self._property_funcs]
        if unknown:
            raise ValueError(f"未知的性質: {unknown}，請先用 register_property 註冊")

        rows = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            row = {"smiles": smiles}
            for prop in properties:
                if mol is None:
                    row[prop] = float("nan")
                    continue
                try:
                    row[prop] = self._property_funcs[prop](mol)
                except Exception:
                    row[prop] = float("nan")
            rows.append(row)

        return pd.DataFrame(rows)


if __name__ == "__main__":
    api = PropertyInferenceAPI()
    df = api.inference_pipeline(
        ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"],
        properties=["ClogP", "SAScore", "MolWt"]
    )
    print(df)
