"""
SMILES Tokenizer 模組
將 SMILES 字串轉換為 token 序列，並提供編碼/解碼功能
"""

import re
import json
from typing import List, Dict
from rdkit import Chem


class SmilesTokenizer:
    """SMILES 分詞器"""

    # SMILES 的 token 正則表達式模式
    # 匹配多字符原子、單字符原子、括號、鍵結符號等
    PATTERN = r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

    def __init__(self):
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}

        # 特殊 tokens
        self.pad_token = '<PAD>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.unk_token = '<UNK>'

        # 初始化特殊 tokens
        self.special_tokens = [self.pad_token, self.start_token, self.end_token, self.unk_token]
        for i, token in enumerate(self.special_tokens):
            self.token_to_idx[token] = i
            self.idx_to_token[i] = token

    def tokenize(self, smiles: str) -> List[str]:
        """將 SMILES 字串分詞為 token 列表"""
        tokens = re.findall(self.PATTERN, smiles)
        return tokens

    def build_vocab(self, smiles_list: List[str]) -> None:
        """從 SMILES 列表建立詞彙表"""
        vocab = set()

        for smiles in smiles_list:
            tokens = self.tokenize(smiles)
            vocab.update(tokens)

        # 將詞彙添加到 token_to_idx
        start_idx = len(self.special_tokens)
        for idx, token in enumerate(sorted(vocab), start=start_idx):
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

        print(f"詞彙表大小: {len(self.token_to_idx)} (包含 {len(self.special_tokens)} 個特殊 tokens)")

    def encode(self, smiles: str, add_special_tokens: bool = True) -> List[int]:
        """將 SMILES 編碼為索引列表"""
        tokens = self.tokenize(smiles)

        # 轉換為索引
        indices = [self.token_to_idx.get(token, self.token_to_idx[self.unk_token])
                   for token in tokens]

        # 添加特殊 tokens
        if add_special_tokens:
            indices = [self.token_to_idx[self.start_token]] + indices + [self.token_to_idx[self.end_token]]

        return indices

    def decode(self, indices: List[int], remove_special_tokens: bool = True) -> str:
        """將索引列表解碼為 SMILES"""
        tokens = []

        for idx in indices:
            if idx in self.idx_to_token:
                token = self.idx_to_token[idx]

                # 遇到 END token 就停止
                if token == self.end_token:
                    break

                # 跳過特殊 tokens
                if remove_special_tokens and token in self.special_tokens:
                    continue

                tokens.append(token)

        return ''.join(tokens)

    def save(self, filepath: str) -> None:
        """保存 tokenizer 到文件"""
        data = {
            'token_to_idx': self.token_to_idx,
            'idx_to_token': {int(k): v for k, v in self.idx_to_token.items()},
            'special_tokens': self.special_tokens
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Tokenizer 已保存到: {filepath}")

    def load(self, filepath: str) -> None:
        """從文件加載 tokenizer"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.token_to_idx = data['token_to_idx']
        self.idx_to_token = {int(k): v for k, v in data['idx_to_token'].items()}
        self.special_tokens = data['special_tokens']

        print(f"Tokenizer 已從 {filepath} 加載")
        print(f"詞彙表大小: {len(self.token_to_idx)}")

    @property
    def vocab_size(self) -> int:
        """返回詞彙表大小"""
        return len(self.token_to_idx)

    @property
    def pad_idx(self) -> int:
        """返回 PAD token 的索引"""
        return self.token_to_idx[self.pad_token]

    @property
    def start_idx(self) -> int:
        """返回 START token 的索引"""
        return self.token_to_idx[self.start_token]

    @property
    def end_idx(self) -> int:
        """返回 END token 的索引"""
        return self.token_to_idx[self.end_token]


def randomize_smiles(smiles: str) -> str:
    """隨機化 SMILES（改變原子順序但保持分子結構）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles

    # 使用不同的隨機種子生成不同的 SMILES 表示
    try:
        random_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        return random_smiles
    except:
        return smiles


def canonicalize_smiles(smiles: str) -> str:
    """規範化 SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    return canonical_smiles


if __name__ == "__main__":
    # 測試 tokenizer
    tokenizer = SmilesTokenizer()

    # 測試 SMILES
    test_smiles = [
        "CCO",
        "c1ccccc1",
        "CC(=O)OC1=CC=CC=C1C(=O)O",
    ]

    # 建立詞彙表
    tokenizer.build_vocab(test_smiles)

    # 測試編碼/解碼
    for smiles in test_smiles:
        print(f"\n原始 SMILES: {smiles}")
        tokens = tokenizer.tokenize(smiles)
        print(f"Tokens: {tokens}")
        indices = tokenizer.encode(smiles)
        print(f"Indices: {indices}")
        decoded = tokenizer.decode(indices)
        print(f"解碼結果: {decoded}")
        print(f"匹配: {smiles == decoded}")
