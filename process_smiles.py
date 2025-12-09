#!/usr/bin/env python3
"""
處理 SMILES 資料：
1. 讀取 train.txt 文件
2. 移除手性和順反異構信息
3. 過濾掉重原子數 > 20 的分子
4. 輸出到 CSV 文件
"""

import csv
from rdkit import Chem
from rdkit.Chem import Descriptors


def remove_stereochemistry(smiles):
    """移除 SMILES 中的手性和順反異構信息"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 移除立體化學信息
        Chem.RemoveStereochemistry(mol)
        # 轉換回 SMILES（不包含立體化學信息）
        clean_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        return clean_smiles
    except Exception as e:
        print(f"處理 SMILES 時出錯: {smiles}, 錯誤: {e}")
        return None


def count_heavy_atoms(smiles):
    """計算重原子數（非氫原子數）"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol.GetNumHeavyAtoms()
    except Exception as e:
        print(f"計算重原子數時出錯: {smiles}, 錯誤: {e}")
        return None


def process_smiles_file(input_file, output_file, max_heavy_atoms=20):
    """
    處理 SMILES 文件

    Args:
        input_file: 輸入文件路徑
        output_file: 輸出 CSV 文件路徑
        max_heavy_atoms: 最大重原子數閾值
    """
    processed_data = []
    skipped_count = 0
    invalid_count = 0

    # 讀取輸入文件
    with open(input_file, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    print(f"總共讀取 {len(smiles_list)} 個 SMILES")

    # 處理每個 SMILES
    for idx, smiles in enumerate(smiles_list, 1):
        # 移除立體化學信息
        clean_smiles = remove_stereochemistry(smiles)

        if clean_smiles is None:
            invalid_count += 1
            continue

        # 計算重原子數
        heavy_atoms = count_heavy_atoms(clean_smiles)

        if heavy_atoms is None:
            invalid_count += 1
            continue

        # 過濾重原子數 > max_heavy_atoms 的分子
        if heavy_atoms > max_heavy_atoms:
            skipped_count += 1
            continue

        processed_data.append({
            'smiles': clean_smiles,
            'heavy_atoms': heavy_atoms
        })

        if idx % 1000 == 0:
            print(f"已處理 {idx} 個 SMILES...")

    # 寫入 CSV 文件
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['smiles', 'heavy_atoms'])
        writer.writeheader()
        writer.writerows(processed_data)

    print(f"\n處理完成！")
    print(f"有效分子數: {len(processed_data)}")
    print(f"過濾掉的分子數（重原子數 > {max_heavy_atoms}）: {skipped_count}")
    print(f"無效的 SMILES 數: {invalid_count}")
    print(f"結果已保存到: {output_file}")


if __name__ == "__main__":
    input_file = "./data/train.txt"
    output_file = "./data/train_processed.csv"

    process_smiles_file(input_file, output_file, max_heavy_atoms=20)
