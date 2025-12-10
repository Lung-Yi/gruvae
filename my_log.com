載入配置檔案: train.yaml
配置載入成功!

使用設備: cuda

建立 tokenizer...
詞彙表大小: 27 (包含 4 個特殊 tokens)
Tokenizer 已保存到: ./checkpoints_large/tokenizer.json

建立 Dataset...
載入 517445 個 SMILES
訓練集: 465700, 驗證集: 51745

建立模型...
模型參數量: 16,640,795
開始訓練 150 epochs...
訓練集大小: 465700
驗證集大小: 51745
設備: cuda
初始學習率: 0.00075
KL Weight 範圍: 0.0 -> 0.01 (over 100 epochs)


================================================================================
Reconstruction Examples (Epoch 1):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1ccccc1F)c1ccsc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1ccccc1F)c1ccsc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2=C(Oc3ccccc3)CC2)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2=C(Oc3ccccc3)CC2)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC)cccc1C(=O)N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC)cccc1C(=O)N(C)C
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Match: ✓

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CCN(c2nnc(C(F)(F)F)n2)C(=O)O1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CCN(c2nnc(C(F)(F)F)n2)C(=O)O1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1ccc2cccc(CCC(=O)NC(C)(C)CO)c2c1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1ccc2cccc(CCC(=O)NC(C)(C)CO)c2c1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 1):
================================================================================
[1] Cc1ccc(NC(=O)C2CCc3cc(CO)ccc32)n1
[2] CC(=O)Nc1nc(C2CC2)ccc1C(=O)OC
[3] Cc1cc(NC(=O)C2CCOC2)c(C)nc1
[4] Cc1cc(C(=O)Nc2cccnc2SCC(C)O)c[nH]1
[5] CC(O)CNc1ccc(C(=O)c2ccc[nH]2)nc1C
[6] Cc1ccc(C(=O)Nc2cccn3CCO2)c(C)c1
[7] Cc1cc(C(=O)Nc2cccnc2CCO)c(C)[nH]1
[8] Cc1cc(NC(=O)C2CC2)n(Cc2cccs2)c1C
[9] Cc1ccc(C(=O)Nc2cccn3CC2)c(OC)c1
[10] Cc1ccc(NC(=O)c2ccno2)c(CC2CC2)c1
================================================================================


================================================================================
Epoch 1 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0001
  Train - Loss: 0.2899, Recon: 0.2284, KL: 615.3828
  Val   - Loss: 2.3715, Recon: 2.3707, KL: 7.8026
  Val Recon Acc:     49.94% (25843/51745)
  ✓ Best model saved!


================================================================================
Reconstruction Examples (Epoch 2):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2COc3ccccc3O2)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2COc3ccccc3O2)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC2CC2)cccc1N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC2CC2)cccc1N(C)C
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Match: ✓

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CN(c2nnc(C(F)(F)F)s2)CC1(N)=O
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CN(c2nnc(C(F)(F)F)s2)CC1(N)=O
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Match: ✓

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 2):
================================================================================
[1] Cc1ccc(NC(=O)C2CC2)c(OCCn2cccn2)c1
[2] CC(NC(=O)C1CC1)c1ccc2c(c1)CC(=O)O2
[3] Cc1ccc(C(=O)N2CCc3cc(Cl)ccn32)s1
[4] CC(=O)N(CC1CC1)c1ccc(C#N)cc1O
[5] CC(=O)N(CC1CC1)c1ccc(C)cc1Cl
[6] CC(Nc1ccc(CO)cc1Cl)C1CC1
[7] Cc1ncc(C(=O)N2CCOCC2)c(C)c1Br
[8] CC(Nc1ccc(CO)cn1)C1CCS(=O)(=O)N1
[9] CC(C)NC(=O)C1CCOc2ccc(N)nc21
[10] CC(O)CN1CCOC(C)c2ncc(C(=O)OC)c21
================================================================================


================================================================================
Epoch 2 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0002
  Train - Loss: 0.0728, Recon: 0.0713, KL: 7.4457
  Val   - Loss: 1.7894, Recon: 1.7880, KL: 7.4223
  Val Recon Acc:     65.46% (33870/51745)
  ✓ Best model saved!


================================================================================
Reconstruction Examples (Epoch 3):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2COc3ccccc3S2(=O)=O
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2COc3ccccc3S2(=O)=O
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Match: ✓

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Match: ✓

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CC(O)CN1C(=O)c1snnc1C(F)(F)F
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CC(O)CN1C(=O)c1snnc1C(F)(F)F
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Match: ✓

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 3):
================================================================================
[1] Cc1ccc(CC(=O)N2CCc3cc(C(C)=O)cnc32)[nH]1
[2] Cc1ccc(N(C)CC2CCOC2=O)nc1C
[3] Cc1ccc(C(=O)Nc2ccn(C)n2)c(O)c1CC#N
[4] Cc1ccc(C(=O)Nc2ccc(CO)cn2)c(C)c1
[5] CC1CN(Cc2ccc(C)cc2)C(=O)c2ncco2)c1
[6] CC1CCN(C(=O)c2ccc(C)nc2Br)C1
[7] CC(NC(=O)c1ccc(C)cc1Cl)c1cccs1
[8] CC1CCN(Cc2ccc(C)cc2O)C(=O)n1
[9] Cc1cc(C(=O)Nc2cccnc2CC#N)c(C)o1
[10] CC(NC(C)=O)c1cc(C2CC2)nc(=O)o1
================================================================================


================================================================================
Epoch 3 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0003
  Train - Loss: 0.0574, Recon: 0.0554, KL: 6.8102
  Val   - Loss: 1.6624, Recon: 1.6588, KL: 12.0140
  Val Recon Acc:     69.55% (35987/51745)
  ✓ Best model saved!


================================================================================
Reconstruction Examples (Epoch 4):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1ccccc1F)c1cccs1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1ccccc1F)c1cccs1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2C(=O)OCc3ccccc32)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2C(=O)OCc3ccccc32)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Match: ✓

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Match: ✓

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CC(O)C(O)CN1c1nnc(C(F)(F)F)s1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CC(O)C(O)CN1c1nnc(C(F)(F)F)s1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(CCC(=O)NC(C)(C)CO)cc2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CCC(=O)NC(C)(C)CO)cc2ccccc21
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 4):
================================================================================
[1] CC(NC(=O)c1ccn(C)c1)c1ncccc1Cl
[2] CC(Nc1ccc(C)cc1Cl)C(=O)N1CCC(C)C1
[3] Cc1cc(C(=O)N(C)CC2CCOC2)cnc1Br
[4] Cc1ccc(C(=O)Nc2ccn(C)c2)c2c1CC(C)O2
[5] Cc1ccc(NC(=O)C2CC2)c(C)c1-c1ccco1
[6] CC(NC(=O)c1ccc(=O)n2cccnc12)C1CC1
[7] Cc1cc(N2CCC(=O)CC2)c(C(C)=O)cc1Cl
[8] Cc1ccc(C2CCOC2=O)cc1NCc1ccn(C)c1
[9] Cc1cc(C(=O)NCC2CC2)cn2ccc(OC)cc12
[10] Cc1ccc(C(=O)Nc2ccnc(CO)n2)c(C)c1
================================================================================


================================================================================
Epoch 4 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0004
  Train - Loss: 0.0518, Recon: 0.0494, KL: 5.9793
  Val   - Loss: 50.8981, Recon: 21.5691, KL: 73322.5238
  Val Recon Acc:     71.38% (36933/51745)


================================================================================
Reconstruction Examples (Epoch 5):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Match: ✓

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2COc3ccccc32)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2COc3ccccc32)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC(=O)N(C)C)cccc1C1CC1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC(=O)N(C)C)cccc1C1CC1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Match: ✓

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CC(O)CN1c1nnc(C(F)(F)F)s1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CC(O)CN1c1nnc(C(F)(F)F)s1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Match: ✓

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 5):
================================================================================
[1] CC1CC1CNC(=O)c1ccc(OC)c2ncccc12
[2] CC(NC(=O)CSc1ncc(C)o1)c1ccc(Cl)cc1
[3] Cc1nc2cc(C(=O)NCCO)ccc2n1C
[4] CC(O)CNC(=O)c1cc(C)nc1c1ccc[nH]1
[5] Cc1cccnc1NCC(=O)c1cc(C2CC2)no1
[6] Cc1ccc(C(=O)N(C)CC2CC2)c(OC)n1
[7] Cc1cccnc1NC(=O)C1CCOC1C
[8] CC(C)NC(=O)c1ccc(SCC2CC2)nc1N
[9] CCn1cc(C(=O)NC2CC2)c(OC(C)=O)c1C
[10] CC1CCC(=O)N(c2ccc(CO)cc2Cl)C1
================================================================================


================================================================================
Epoch 5 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0005
  Train - Loss: 0.0491, Recon: 0.0465, KL: 5.2486
  Val   - Loss: 502864765660334587904.0000, Recon: 57510980236.8965, KL: 1005729470444060253618176.0000
  Val Recon Acc:     73.28% (37921/51745)


================================================================================
Reconstruction Examples (Epoch 6):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2C(=O)Oc3ccccc32)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2C(=O)Oc3ccccc32)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Match: ✓

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Match: ✓

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CN(C(F)(F)F)Cc2nnc(C(=O)O)s21
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CN(C(F)(F)F)Cc2nnc(C(=O)O)s21
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Match: ✓

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 6):
================================================================================
[1] Cc1ccc(NC(=O)Cc2ncc(CO)s2)cc1C
[2] CC1CC1c1cc(NC(=O)c2ccco2)c(C)nc1
[3] Cc1ccc(CC(=O)N2CCOCC2C)cc1Cl
[4] CC(C)N(Cc1ccc(Br)cc1)c1nccc(=O)o1
[5] Cc1cc(C(=O)N2CCOCC2)cc2cccnc12
[6] Cc1cc2c(nc1NCC(=O)c1ccco1)CCC2=O
[7] CC(C)N(Cc1ccc(=O)n(C)c1)C1CC1
[8] Cc1cc(NC(=O)c2ccco2)cc2nccc(CC#N)c12
[9] CC(C)NC(=O)Cc1ccc(O)c2cccnc12
[10] CC(C)n1cc(C(=O)NCC2CC2)c(C)n1
================================================================================


================================================================================
Epoch 6 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0006
  Train - Loss: 0.0477, Recon: 0.0449, KL: 4.6535
  Val   - Loss: 299752931799579754496.0000, Recon: 24755629941.8121, KL: 499588196234519046520832.0000
  Val Recon Acc:     74.14% (38364/51745)


================================================================================
Reconstruction Examples (Epoch 7):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Match: ✓

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2COc3ccccc32)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2COc3ccccc32)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Match: ✓

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Match: ✓

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CC(O)CN1C(=O)c1snnc1C(F)(F)F
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CC(O)CN1C(=O)c1snnc1C(F)(F)F
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(C(=O)NCCC(C)(C)CO)c2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(C(=O)NCCC(C)(C)CO)c2ccccc21
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 7):
================================================================================
[1] COC(=O)c1ccc(NC(=O)c2ccc(C)o2)nc1
[2] CC(O)c1ccc(CN2CCC(=O)C2)c(C)c1
[3] CC(Nc1ccc(C(=O)CO)cc1Cl)C1CC1
[4] Cc1cc(C(=O)Nc2ccc(CC(F)F)cc2Cl)cn1C
[5] Cc1cc(C(=O)NCC(O)c2cccs2)c(C)n1N
[6] Cc1ccc(C(=O)NC2CCc3ccoc32)c(C)n1
[7] Nc1nc(C(C)C)ccc1NC(=O)c1ccnc(F)c1
[8] COc1ccc(C(=O)NCCc2ncc(C3CC3)n2C)cc1
[9] CC1OCCN1C(=O)c1cc(C)ncc1Br
[10] Cc1cc(NCC(=O)c2ccco2)c(C)n1
================================================================================


================================================================================
Epoch 7 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0007
  Train - Loss: 0.0474, Recon: 0.0445, KL: 4.1909
  Val   - Loss: 1726069.9483, Recon: 1828.7422, KL: 2463201863.2424
  Val Recon Acc:     73.64% (38104/51745)


================================================================================
Reconstruction Examples (Epoch 8):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2COc3ccccc3C2=O)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2COc3ccccc3C2=O)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Match: ✓

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Match: ✓

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CN(C(=O)C(F)(F)F)CC1c1nnc(O)s1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CN(C(=O)C(F)(F)F)CC1c1nnc(O)s1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Match: ✓

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 8):
================================================================================
[1] CC1COCCN1C(=O)CSc1ccc(Cl)cc1N
[2] CC(NC(=O)Cc1ccc(N)c2nccc(C)c12
[3] Cc1cc(C(=O)N2CCC(=O)NC2)nc2cc(C)ccc12
[4] CC(NCC(=O)N1CCOCC1)c1ccc(C)s1
[5] CC(O)CNC(=O)c1cc(C)nc2ccc(F)cc12
[6] CN1CCC(NC(=O)c2ccco2)Cc2cc(C)cnc21
[7] CN1CCc2nc(C(=O)Nc3ccco3)ccc2C1C
[8] CC(Cc1nccs1)Nc1ccc(C#N)cc1Br
[9] Cc1cccc(C(=O)NC2CCOC2)c1C
[10] O=C(Nc1cccc(C(=O)C2CC2)n1)c1ccco1
================================================================================


================================================================================
Epoch 8 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0008
  Train - Loss: 0.0481, Recon: 0.0450, KL: 3.8404
  Val   - Loss: 1.4281, Recon: 1.4251, KL: 3.7101
  Val Recon Acc:     73.40% (37980/51745)
  ✓ Best model saved!


================================================================================
Reconstruction Examples (Epoch 9):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Match: ✓

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC2CC2)cccc1N(C)C(=O)NC
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC2CC2)cccc1N(C)C(=O)NC
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Match: ✓

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CC(O)CN1c1nnc(C(F)(F)F)s1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CC(O)CN1c1nnc(C(F)(F)F)s1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Match: ✓

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 9):
================================================================================
[1] CC(C)c1ncc(C(=O)N2CCC(C)C2)c(=O)o1
[2] CC(=O)NC(c1ccc(OC)cc1)C(=O)n1ccnc1
[3] Cn1cc(C(=O)NCc2ccc(SC3CC3)cc2)nc1C
[4] Cc1cc(C(=O)Nc2cccn2C)oc1C
[5] CC(OC(=O)c1ccc2c(c1)CC(C)N2)n1ccc(F)c
[6] Cc1nc(CNC(=O)C2CC2)cc2ccc(OC)cc12
[7] OC1CCC(NC(=O)c2ccc(C)s2)C1O
[8] CC(NS(C)(=O)=O)c1ccc(OC)c2c1CC2
[9] CCC(=O)Nc1cc2c(n1)CCC(=O)N2
[10] CN(C(=O)c1ccc2c(c1)ncn2C)CC(C)O
================================================================================


================================================================================
Epoch 9 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0009
  Train - Loss: 0.0483, Recon: 0.0451, KL: 3.5677
  Val   - Loss: 1.4005, Recon: 1.3972, KL: 3.6181
  Val Recon Acc:     73.89% (38235/51745)
  ✓ Best model saved!


================================================================================
Reconstruction Examples (Epoch 10):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Match: ✓

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2COc3ccccc3C2=O)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2COc3ccccc3C2=O)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(C2CC2)cccc1C(=O)N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(C(=O)N(C)C)cccc1C1CC1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Match: ✓

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Match: ✓

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(CCC(=O)NC(C)(C)C)c2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CCC(=O)NC(C)(C)C)c2ccccc21
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 10):
================================================================================
[1] c1cc2c(cc1C(=O)OC(C)C)NCC2
[2] OCC1CCC(NC(=O)c2cc(C)ccn2)C1
[3] CC1CC(=O)N(Cc2ccc(Br)cc2)CC1C
[4] COC(=O)CNC(c1ccc(C)nc1)C1CC1=O
[5] Cc1ccc(C(=O)NC2CCC(C)O2)c2cc(C)ccn12
[6] Cc1cn(CC(=O)NC2CC2)c2ccc(C)c(O)c12
[7] Cc1cc(O)c(NC(=O)CN(C)C2CC2)c(C)n1
[8] Cc1ccc(C(=O)NC2CC2C)c(C)n1
[9] Cc1ccc(NCC(=O)c2ncco2)c(C)c1
[10] CC(NC(=O)CSc1ccc2c(c1)CCC2)C(N)=O
================================================================================


================================================================================
Epoch 10 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0010
  Train - Loss: 0.0492, Recon: 0.0458, KL: 3.3506
  Val   - Loss: 1.4393, Recon: 1.4360, KL: 3.2892
  Val Recon Acc:     73.30% (37929/51745)


================================================================================
Reconstruction Examples (Epoch 11):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Match: ✓

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Match: ✓

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Match: ✓

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)c1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)c1ccccc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CC(O)CN(c2nnc(C(F)(F)F)s2)C1=O
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CC(O)CN(c2nnc(C(F)(F)F)s2)C1=O
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)(CO)NC(=O)CCn1cc(C)c2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cc1cn(CCC(=O)NC(C)(C)CO)c2ccccc12
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 11):
================================================================================
[1] CC(=O)c1nc(NC2CCOC2)cc2ccccc12
[2] Cc1ccc(C(=O)N2CCC(O)C2)c(=O)[nH]1
[3] Oc1cccc2c(CC(=O)N(C)C)ncnc12
[4] CC(Nc1nc2ccc(Cl)cc2c(=O)o1)C1CCC1
[5] CC(=O)NC(CCO)c1cccc(-c2cccs2)n1
[6] =O)N(Cc1cc(OC)ccc1F)c1cccs1
[7] CC(NC(=O)Cc1ncc2ccoc2c1)N1CCC1
[8] Cc1nc(C(=O)Nc2ccc(C(C)=O)cn2)sn1
[9] Cc1cc(C(=O)N2CCNC(=O)C2)c(C)nc1O
[10] Cc1nc(CN2CCOC(=O)C2)c(C)cc1F
================================================================================


================================================================================
Epoch 11 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0011
  Train - Loss: 0.0503, Recon: 0.0468, KL: 3.1696
  Val   - Loss: 21.0156, Recon: 5.5655, KL: 14045.5982
  Val Recon Acc:     72.78% (37658/51745)


================================================================================
Reconstruction Examples (Epoch 12):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Match: ✓

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Match: ✓

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Match: ✓

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1C(=O)OC(c2snnc2C(F)(F)F)N1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1NC(c2snnc2C(F)(F)F)OC1=O
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(CC(=O)NCCC(C)(C)CO)c2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CC(=O)NCCC(C)(C)CO)c2ccccc21
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 12):
================================================================================
[1] CC1CCC(=O)N(Cc2ccc(C)cc2F)C1
[2] COC(=O)N1CCC(COc2ccc(C)cc2)C1=O
[3] CC1CCC(NC(=O)c2ccc(C)s2)C1O
[4] CC1CCC(C(=O)NCc2ccc(OC)cc2)n1
[5] Cc1ccc(CC(=O)Nc2nc(C)cs2)o1
[6] CC(NC(=O)CC1CC1)c1ccc(C)s1
[7] Cc1cc(-c2cccc(Cl)c2)c2c(n1)N(C(=O)CO)C
[8] Nc1ccc2c(c1)C(=O)Nc1cc(C)n(C)c1
[9] Cc1cc(C(=O)NCCO)nc(-c2cccnc2C)c1O
[10] CC(c1cccc(=O)[nH]1)N1CCOC(C)C1
================================================================================


================================================================================
Epoch 12 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0012
  Train - Loss: 0.0515, Recon: 0.0479, KL: 3.0264
  Val   - Loss: 1.6242, Recon: 1.6082, KL: 13.2562
  Val Recon Acc:     71.46% (36977/51745)


================================================================================
Reconstruction Examples (Epoch 13):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccs1)Nc1cccc(F)c1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccs1)Nc1cccc(F)c1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Match: ✓

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(C(=O)N(C)C)cccc1OC1CC1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Match: ✓

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)c1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)c1ccccc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CN(c2nnc(C(F)(F)F)s2)CC1(O)C(=O)O
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CN(c2nnc(C(F)(F)F)s2)CC1(O)C(=O)O
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Match: ✓

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 13):
================================================================================
[1] CS(=O)(=O)c1ccc(NC(=O)Cc2ccc(C)n2)c(F
[2] CN(C(=O)Cc1ccc(Br)cc1OC)C1CC1
[3] CN(CC(C)=O)c1ccc2c(c1)CCO2
[4] CC(NS(C)(=O)=O)C(=O)NCc1cccc(OC)c1
[5] FO2
[6] CC1(C)CN(C(=O)CC2CC2)c2ccc(OC)nc21
[7] CN(C(=O)c1ccc(OCC2CC2)nc1C)C1CC1
[8] Cc1ccc(NC(=O)c2ccc(C)o2)cc1Br
[9] NC(=O)NCC(O)c1cccc(Sc2nccn2C)c1
[10] Nc1cc(S(=O)(=O)CC2CCCO2)c(F)cn1
================================================================================


================================================================================
Epoch 13 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0013
  Train - Loss: 0.0539, Recon: 0.0501, KL: 2.9418
  Val   - Loss: 1.5434, Recon: 1.5396, KL: 2.9269
  Val Recon Acc:     71.11% (36798/51745)


================================================================================
Reconstruction Examples (Epoch 14):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Match: ✓

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC(=O)c2ccccc2OC2CC2)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC(=O)c2ccccc2OC2CC2)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Match: ✓

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)c1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)c1ccccc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CC(O)CN1c1nnc(C(F)(F)F)s1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CC(O)CN1c1nnc(C(F)(F)F)s1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Match: ✓

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 14):
================================================================================
[1] 3)N(C)C(=O)Cc1ccc(Cl)cc1
[2] COCC(C)N1CCc2ccc(F)nc21
[3] COC(=O)c1cc(NCC2CC2)c(C)n1-c1ccco1
[4] C1CCC(CO)CC1NC(=O)c1cccc(S(C)=O)c1
[5] O=C(CSc1nccc(=O)c1Br)NC
[6] CC1NC(=O)CC(c2ccncc2)c2cc[nH]c21
[7] OC(C)CN1C(=O)Cc2c(-c3ccc(Cl)cc3)nccc21
[8] Cc1ccc(NCC(=O)NC(C)c2ccncc2)c(C)c1
[9] =O)c2cccc(NCc3ccc(O)cc3)c2c1
[10] O)c(C)cc1OCc1ccc2c(c1)CC2
================================================================================


================================================================================
Epoch 14 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0014
  Train - Loss: 0.0559, Recon: 0.0520, KL: 2.8317
  Val   - Loss: 53483.8135, Recon: 65.6124, KL: 38155860.9959
  Val Recon Acc:     70.60% (36531/51745)


================================================================================
Reconstruction Examples (Epoch 15):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COc1cccc(CNC(=O)Nc2cccs2)c1F
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COc1cccc(CNC(=O)Nc2cccs2)c1F
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2C(=O)OC3ccccc32)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2C(=O)OC3ccccc32)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Match: ✓

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Match: ✓

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CC(=O)N(c2nnc(C(F)(F)F)s2)C1O
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CC(=O)N(c2nnc(C(F)(F)F)s2)C1O
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(CCC(=O)NCC(C)(C)CO)c2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CCC(=O)NCC(C)(C)CO)c2ccccc21
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 15):
================================================================================
[1] C(C)N1CCc2ccc(C(=O)OC3CC3)cc2n1
[2] CN(C)C(=O)NCc1cnc(C)cc1F
[3] Fc1cccnc1NC(=O)Cc1ccc(C2CC2)nc1
[4] CC(c1ccc2c(c1)OCC2)NC(C)c1ccc[nH]1
[5] c1cnn2c(NCC(O)c3ccccc3)csc12
[6] Cc1ncc(C(=O)Nc2cccc(N(C)C)c2)n2c1CCS2
[7] Cc1cc(C(=O)NCC(C)c2ccc(F)cc2Cl)no1
[8] C1COCCN1C(=O)Nc1ccc(C)[nH]1
[9] COc1ccc(CNc2ncc(C)cc2Cl)nc1
[10] Nc1oc(C(=O)NCC2CCCO2)nc1C(C)C
================================================================================


================================================================================
Epoch 15 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0015
  Train - Loss: 0.0614, Recon: 0.0571, KL: 2.8100
  Val   - Loss: 1.7443, Recon: 1.7389, KL: 3.5537
  Val Recon Acc:     68.57% (35479/51745)


================================================================================
Reconstruction Examples (Epoch 16):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Match: ✓

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2Cc3ccccc3OC2=O)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2Cc3ccccc3OC2=O)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC(C)C)cccc1C(=O)N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC(C)C)cccc1C(=O)N(C)C
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Match: ✓

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CC(C(F)(F)F)N(c2nnc(C(F)(F)F)s2)C1=O
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CC(C(F)(F)F)N(c2nnc(C(F)(F)F)s2)C1=O
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Match: ✓

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 16):
================================================================================
[1] OCC1CNc2cc(C(=O)Oc3ccc(F)cc3)nnc21
[2] CC(N)c1ccc2c(c1Br)OCCN2C
[3] Cc1ccc(NC(=O)CSc2nccc(=O)o2)c(C)c1
[4] OC(C)CN(C)C(=O)c1cc(F)cc2ccncc12
[5] COC(=O)C1CCc2cc(C)c(Cl)cc2O1
[6] Cc1nn(CC(=O)Nc2ccc(Br)cc2)c(SC)c1C#N
[7] COc1cc(C(C)N2CCOCC2=O)cc(F)c1
[8] =O)c1cc(N(C)Cc2ccc(Cl)cc2)nc(C)s1
[9] CCn1ncc2c(C(C)OC)ccc(C(F)F)c21
[10] Cc1occ(C(C)NS(=O)(=O)c2cccs2)c1F
================================================================================


================================================================================
Epoch 16 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0016
  Train - Loss: 0.0618, Recon: 0.0574, KL: 2.7293
  Val   - Loss: 1.7161, Recon: 1.7117, KL: 2.7236
  Val Recon Acc:     68.53% (35461/51745)


================================================================================
Reconstruction Examples (Epoch 17):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Match: ✓

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2COc3ccccc3C2=O)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2COc3ccccc3C2=O)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC2CC2)cccc1N(C)C(=O)NC
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC2CC2)cccc1N(C)C(=O)NC
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Match: ✓

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CC(O)(C(F)(F)F)C(c2nnc(C(F)(F)F)s2)C1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CC(c2nnc(C(F)(F)F)s2)C(O)(C(F)(F)F)C1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(CC(C)(C)C)c2ccccc2c1CC(=O)NCO
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CC(C)(C)C)c2ccccc2c1CC(=O)NCO
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 17):
================================================================================
[1] CC(NC(C)=O)CSc1nc2ccc(OC)cc2s1
[2] COc1ccc(C(=O)NC2CCN(C(C)C)C2)nc1
[3] Cc1ncc(NC(=O)Cc2ccc(C)cc2)c(Cl)n1
[4] Cc1ccc(C(=O)NCc2nccc(C)n2)c(O)c1C
[5] OC(C)c1ccc(NCc2cccn2)c(=O)o1
[6] Cn1c(C(=O)NC2CC2)nc2ccc(Cl)cc21
[7] CCc1cc(=O)[nH]c(C2CCN(C(C)=O)C2)n1
[8] COC(=O)C(C)Nc1ccc(C#N)cn1
[9] CC(=O)NC1CCOc2ccc(CC#N)cn21
[10] =O)(C)c1ccc(C#N)c(Br)c1
================================================================================


================================================================================
Epoch 17 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0017
  Train - Loss: 0.0641, Recon: 0.0596, KL: 2.6518
  Val   - Loss: 1.7724, Recon: 1.7679, KL: 2.6324
  Val Recon Acc:     66.83% (34579/51745)


================================================================================
Reconstruction Examples (Epoch 18):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccs1)c1ccc(F)cc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccs1)c1ccc(F)cc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2Cc3ccccc3OC2=O)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2Cc3ccccc3OC2=O)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Match: ✓

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Match: ✓

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CN(c2nnc(C(F)(F)F)s2)CC1=O
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CN(c2nnc(C(F)(F)F)s2)CC1=O
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)(CO)NCCC(=O)Oc1ccccc1C
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cc1ccccc1OC(=O)CCNC(C)(C)CO
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 18):
================================================================================
[1] CCn1c(CCO)nc2cccc(S(=O)(=O)NC)c21
[2] Cc1ncc(C(=O)NC2CC2)c2ccc(OC)cc12
[3] COc1ccc(C(=O)Nc2cnccc2O)n1CC1CC1
[4] O=C(CSc1cccc(F)c1)c1nc2cc(N)ccc2[nH]1
[5] Cc1cc(NC(C)c2ncc[nH]2)cn1CCO
[6] CC(C)NC(=O)c1ccc(Cl)c2c1CCO2
[7] CN(C(=O)c1ccco1)c1ccc(F)cc1C
[8] NC(=O)c1cc(NCc2ccc(Br)cc2)cn1C1CC1
[9] CN(C(=O)C1CCOc2ccc(F)cc21)c1nccn1C
[10] COc1ncc(C(=O)NC(C)C)c2c(OC)cccc12
================================================================================


================================================================================
Epoch 18 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0018
  Train - Loss: 0.0672, Recon: 0.0625, KL: 2.6130
  Val   - Loss: 1.9276, Recon: 1.9229, KL: 2.6118
  Val Recon Acc:     64.68% (33471/51745)


================================================================================
Reconstruction Examples (Epoch 19):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Match: ✓

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC(=O)c2ccccc2)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC(=O)c2ccccc2)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Match: ✓

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)c1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)c1ccccc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CC(O)(C(F)(F)F)CCN1c1nnc(C(F)(F)F)s1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CC(O)(C(F)(F)F)CCN1c1nnc(C(F)(F)F)s1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(NC(=O)CCC(C)(C)c2ccccc2C)cc1=O
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(NC(=O)CCC(C)(C)c2ccccc2C)cc1=O
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 19):
================================================================================
[1] CC(=O)N1CCC(c2ccc(F)c(CO)c2)C1
[2] OCCN(C1CC1)C(=O)c1ccc(Br)cn1
[3] O)c1cccc(CNc2ccc(=O)nc2)c1
[4] OC(=O)NC1CCc2c(cc(C)c(F)c2)n1
[5] CC(C)n1cc(C(=O)NCCc2ccoc2)cc1Br
[6] CC(=O)N1CCc2cc(-c3ccc(Cl)cc3F)no2)N1
[7] CC(Cc1cccc(OC(F)F)c1)C(=O)NC1CC1
[8] Nc1cc(CC(=O)N2CCOC2)nc2cccc(C)c12
[9] Cc1ccc2c(c1)CCNC2CS(=O)(=O)C(C)C
[10] O=C(Cc1ccco1)Nc1ncc(F)c(C2CC2)c1
================================================================================


================================================================================
Epoch 19 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0019
  Train - Loss: 0.0748, Recon: 0.0698, KL: 2.6238
  Val   - Loss: 2.2554, Recon: 2.2476, KL: 4.0957
  Val Recon Acc:     63.08% (32642/51745)


================================================================================
Reconstruction Examples (Epoch 20):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1snnc1NC(=O)c1cc2ccccc2o1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1snnc1NC(=O)c1cc2ccccc2o1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Match: ✓

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)c1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)c1ccccc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(C(=O)N(C)CC(C)C)c(Br)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(C(=O)N(C)CC(C)C)c(Br)c1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CC(O)CN1C(=O)c1nnc(C(F)(F)F)s1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CC(O)CN1C(=O)c1nnc(C(F)(F)F)s1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(CCC(=O)NC(C)(C)CO)cc2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CCC(=O)NC(C)(C)CO)cc2ccccc21
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 20):
================================================================================
[1] COc1ccc(C2CCS(=O)(=O)NCC2CC2CC2)s1
[2] OC1CN(C(=O)CCc2ccc[nH]2)C(CC)C1
[3] CC(C)NC(=O)c1ccc(COc2cccc(F)c2)o1
[4] NC(COc1nnc(C)o1)c1ccc(NC(C)=O)cc1Br
[5] Cc1nc2c(cc1CNS(C)(=O)=O)CCO2
[6] O=C(NC1CC=CC(C)C)c1-c1cccc(F)c1
[7] CCc1ccc(NC(=O)c2c(C)n[nH]c2C#N)c(C)n1
[8] CN(CC(F)F)C(=O)c1cccc(Cl)c1C(C)O
[9] CC1CC1NC(=O)c1ccc(CO)c2cc[nH]c12
[10] CC(=O)N1CCC(C(=O)OCc2cccc(F)c2)C1C
================================================================================


================================================================================
Epoch 20 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0020
  Train - Loss: 0.0800, Recon: 0.0748, KL: 2.5971
  Val   - Loss: 2.1544, Recon: 2.1492, KL: 2.5942
  Val Recon Acc:     60.40% (31253/51745)


================================================================================
Reconstruction Examples (Epoch 21):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC(=O)c2ccccc2OC2C2)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC(=O)c2ccccc2OC2C2)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1cccc(OC(=O)C2CC2)c1N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1cccc(OC(=O)C2CC2)c1N(C)C
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)c1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)c1ccccc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CC(O)(C(F)(F)F)Nc2snnc2C1=O
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CC(O)(C(F)(F)F)Nc2snnc2C1=O
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(CCC(=O)NC(C)(CO)CO)cc1C
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cc1cc(CCC(=O)NC(C)(CO)CO)cn1C
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 21):
================================================================================
[1] CC1CNc2ccc(C(=O)c3ccc(F)cc3F)nc2C1
[2] N=C(NCc1ccco1)c1ccc2c(n1)CCN2
[3] Cn1cc(F)c(-c2nc3ccc(C)cc3N2CC2)c1
[4] O=C(Nc1ccncc1C1CC1)c1cccn(C)c1=O
[5] Cc1ccc(CNC(=O)Cc2ccc3c(c2)CCC3)n1C(C)
[6] CC1CCC(c2nc(C(C)C)ccc2Cl)cc1
[7] CN(Cc1cccc2c1C)CC(=O)C2
[8] CC1(C)CN(C(=O)c2sccc2C(F)(F)F)CCO1
[9] C1(C(=O)N2CCOC(c3ccccn3)CC2)C(N)C1
[10] COC(=O)C1CCN(c2ccc(C)cc2)CC1CO
================================================================================


================================================================================
Epoch 21 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0021
  Train - Loss: 0.0865, Recon: 0.0810, KL: 2.6138
  Val   - Loss: 758.7680, Recon: 16.6800, KL: 353375.2417
  Val Recon Acc:     57.16% (29579/51745)


================================================================================
Reconstruction Examples (Epoch 22):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)c1cccs1)Nc1cccc(F)c1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)c1cccs1)Nc1cccc(F)c1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2C(=O)Oc3ccccc32)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2C(=O)Oc3ccccc32)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Match: ✓

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NCC(Cc1ccccc1)NCc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NCC(Cc1ccccc1)NCc1ccccc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CCN(c2nnc(C(F)(F)F)s2)CC1O
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CCN(c2nnc(C(F)(F)F)s2)CC1O
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(C(=O)NCCC(C)(C)CO)c2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(C(=O)NCCC(C)(C)CO)c2ccccc21
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 22):
================================================================================
[1] Cc1cccc(NC(=O)c2cc(C(F)(F)F)cs2)n1
[2] )c(C(=O)NC2CC2)c1ccco1
[3] COC(=O)Cc1cnc2cccc(S(N)(=O)=O)c2c1
[4] Cn1c(SCC(=O)N(C)C)nc2ccc(Cl)cc21
[5] CS(=O)(=O)NC(C)c1cc(C)nc2ccc(F)cc12
[6] cccc2)Cc2ccc(Cl)cc21
[7] Cc1cc(C(=O)NC2CCC2)c2cccc(C)nc12
[8] CSc1nc(C2CC2)cc(C(=O)OCc2cccs2)n1C
[9] Cc1cccc(C(C)N2CCSc3ncccc3C2)c1
[10] CC1NCCN(Cc2cccnc2)C1=O
================================================================================


================================================================================
Epoch 22 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0022
  Train - Loss: 0.0923, Recon: 0.0866, KL: 2.5970
  Val   - Loss: inf, Recon: 1148064758133696000.0000, KL: inf
  Val Recon Acc:     54.63% (28269/51745)


================================================================================
Reconstruction Examples (Epoch 23):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)CNC(=O)C1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)CNC(=O)C1CC1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2Cc3ccccc3C2=O)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2Cc3ccccc3C2=O)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Match: ✓

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Match: ✓

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN(CC(F)(F)F)C(=O)Nc1nnc(C2CO)s1)C2(O)O
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN(CC(F)(F)F)C(=O)Nc1nnc(C2CO)s1)C2(O)O
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Match: ✓

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 23):
================================================================================
[1] (C)CSc1nnc(=O)n1Cc1cccc(OC)c1
[2] CC(O)CN(C)c1ccc(S(=O)(=O)NC2CC2)s1
[3] OCC1(OC)CCc2ccc(OC)cc21
[4] COC(=O)c1ccc(NCc2c(O)cccc2Cl)nc1
[5] CC1(CC)CCN(C(=O)c2ccc(OC)nc2)CC1
[6] NCCOC(=O)c1ccc(C2CC2(Cl)Cl)cc1
[7] NC(=O)c1ccc(OC2CC2)cc1NC(C)=O
[8] c1cnc(CCc2cccc(Cl)c2)n1C(F)F
[9] CNC(c1cccc2nc(C(N)=O)ns1)CC2
[10] CC(C)N(C)c1ccc(-c2ncc(CO)[nH]2)cc1C
================================================================================


================================================================================
Epoch 23 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0023
  Train - Loss: 0.0965, Recon: 0.0905, KL: 2.5914
  Val   - Loss: 2.6377, Recon: 2.6321, KL: 2.4536
  Val Recon Acc:     52.14% (26981/51745)


================================================================================
Reconstruction Examples (Epoch 24):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC(=O)c2ccccc2OC)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC(=O)c2ccccc2OC)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC(=O)N(C)C2CC2)cccc1OC
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC)cccc1OC(=O)N(C)C1CC1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(Cc1ccccc1)NC(=O)Cc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(Cc1ccccc1)NC(=O)Cc1ccccc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CC(O)(C(F)(F)F)CN1c1nnc(C(N)=O)s1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CC(O)(C(F)(F)F)CN1c1nnc(C(N)=O)s1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(CO)c2cc(CCC(C)(C)NC(=O)c3ccccc3)cc
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CO)c2cc(CCC(C)(C)NC(=O)c3ccccc3)cc
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 24):
================================================================================
[1] COc1ccc(C2CC(=O)NCC2(C)C)nc1
[2] CC(=O)N(C)CC(=O)NCc1cccc(F)c1
[3] Cc1nc(C(=O)Nc2cccs2)c(COC)n1C
[4] Cc1nn(Cc2cccc(C#N)c2)c(C)c1NC(=O)C1CC
[5] O)(=O)C(c2ccc(OC)cc2)C1
[6] CC(=O)NCC1CCOc2ccc(C)cc2C1=O
[7] CCN1C(=O)CN(C(C)c2ccc(O)cc2F)C1
[8] NF)CC)n2c(N)nccc21
[9] CN1C(=O)CCN1c1ncc(O)c2cccc(Br)c12
[10] (=O)c2cc(CO)ccc2C(=O)N(C)CC1CC1
================================================================================


================================================================================
Epoch 24 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0024
  Train - Loss: 0.1016, Recon: 0.0955, KL: 2.5310
  Val   - Loss: 2.8517, Recon: 2.8455, KL: 2.5759
  Val Recon Acc:     49.04% (25374/51745)


================================================================================
Reconstruction Examples (Epoch 25):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2C(=O)Oc3ccccc32)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2C(=O)Oc3ccccc32)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1cccc(OC(=O)N(C)C2CC2)c1C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1cccc(OC(=O)N(C)C2CC2)c1C
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Match: ✓

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CC(O)CN1c1nnc(C(F)(F)F)s1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CC(O)CN1c1nnc(C(F)(F)F)s1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)(CO)NC(=O)CCc1cn(C)c2ccccc12
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Match: ✓

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 25):
================================================================================
[1] CC(C)C1CN(Cc2ncccc2C2CC2)CCO1
[2] CC(C)c1ccc(C(=O)N2CCOCC2)c(C#N)n1
[3] Cc1nc(N2CCOC(CO)C2)ccc1Br
[4] CC1CN(Cc2cccs2)CCC1C(=O)OC
[5] cc1C(CO)c1cc2c(Cl)cccn2n1
[6] =c1cccc(CC(O)Nc2cc[nH]n2)c1
[7] Cc1cccn2c(C)cc(NC(=O)C3CCCO3)c12
[8] O=C(NCCOC(=O)C1CCC1)c1nc(C2CC2)ccc1F
[9] CCOC(C)C1COc2ccc(C#N)c(OC)c2C1
[10] CCC(O)CNC(=O)c1ccc(OC)nc1
================================================================================


================================================================================
Epoch 25 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0025
  Train - Loss: 0.1076, Recon: 0.1013, KL: 2.5272
  Val   - Loss: 2.8886, Recon: 2.8823, KL: 2.5310
  Val Recon Acc:     47.97% (24823/51745)


================================================================================
Reconstruction Examples (Epoch 26):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Match: ✓

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2=Cc3ccccc3OC2)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2=Cc3ccccc3OC2)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1c(OC(=O)N(C)C)cccc1C(=O)N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1c(OC(=O)N(C)C)cccc1C(=O)N(C)C
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)C(Cc1ccccc1)NCc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)C(Cc1ccccc1)NCc1ccccc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CC(O)CN1c1nnc(C(F)(F)F)s1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CC(O)CN1c1nnc(C(F)(F)F)s1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Match: ✓

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 26):
================================================================================
[1] 3)cc(C(C)c2ccc(O)cc2)CC1
[2] Cc1cc(C(=O)NC2COCC2)ccc1SC(C)CO
[3] CN1C(=O)CC(S(=O)(=O)c2cccc(F)c2)N1
[4] CC1CCC(=O)N(c2nc(-c3ccccc3)ccc2)C1
[5] CC1CN(C(=O)c2cccc(CO)n2)CCN1
[6] COc1ncc(C(=O)N(C)CC2CCO2)cc1S(C)(=O)=
[7] COc1cccc(C(=O)NCc2ccc(C)s2)c1C(F)F
[8] CC(C)Cn1ccsc1S(=O)(=O)N1CC1
[9] O=C(Cc1cn2c(n1)CCCO2)Nc1cccs1
[10] Cc1ccc(N2CCCC(OCC(F)F)C2)no1
================================================================================


================================================================================
Epoch 26 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0026
  Train - Loss: 0.1120, Recon: 0.1056, KL: 2.4753
  Val   - Loss: nan, Recon: nan, KL: inf
  Val Recon Acc:     45.83% (23715/51745)


================================================================================
Reconstruction Examples (Epoch 27):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)C1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)C1CC1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccs1)c1ccc(F)cc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccs1)c1ccc(F)cc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nsc(NC(=O)c2cccc3c2OC)n1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nsc(NC(=O)c2cccc3c2OC)n1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1cccc(OC2CC2)c1N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1cccc(OC2CC2)c1N(C)C
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)C(Cc1ccccc1)NCc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)C(Cc1ccccc1)NCc1ccccc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Match: ✓

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CC(O)(c2nnc(C(F)(F)F)s2)CC1=O
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CC(O)(c2nnc(C(F)(F)F)s2)CC1=O
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1c(CCC(=O)NC(C)(C)CO)cc2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1c(CCC(=O)NC(C)(C)CO)cc2ccccc21
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2cnc(COC)s2)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2cnc(COC)s2)cs1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 27):
================================================================================
[1] CC1CCC(C(=O)NCc2c(C)n[nH]c2N)CC1
[2] Cc1ccc(OC(=O)c2cn(C)c(N)n2)cc1
[3] CC(C)N(C)Cc1cn(C)nc1-c1cccc(C#N)c1
[4] OCC(=O)NC1(c2ccc(F)cc2F)cn2c1-C2
[5] CC(=O)N1CC(C)c2ccc(OCC3CC3)cc21
[6] CCOc1ncc(CC(=O)N2CCc3c(C)ccnc32)o1
[7] CN(C)C(Cc1cccc(SC)c1)c1cc(N)cnc1C1CC1
[8] CCN(C(=O)Nc1cnc(C)c2ccccc12)C1CC1
[9] CC1CN(C(=O)c2cc(Cl)ccn2)CCO1
[10] Nc3ccc(N(C)C)ccc2c1)CCO
================================================================================


================================================================================
Epoch 27 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0027
  Train - Loss: 0.1266, Recon: 0.1199, KL: 2.4923
  Val   - Loss: 3.5191, Recon: 3.5118, KL: 2.6725
  Val Recon Acc:     37.86% (19589/51745)


================================================================================
Reconstruction Examples (Epoch 28):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC2COc3ccccc32)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC2COc3ccccc32)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1cccc(OC)c1C(=O)N(C)C1CC1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1cccc(OC)c1C(=O)N(C)C1CC1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Match: ✓

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COC(=O)c1cc(Br)ccc1N(CC)C(C)C
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCN(c1ccc(Br)cc1C(=O)OC)C(C)C
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CCOC(c2nnc(C(F)(F)F)s2)C1=O
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CCOC(c2nnc(C(F)(F)F)s2)C1=O
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(CCO)(C)C(=O)NCc1cn(C)c2ccccc12
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CNC(=O)C(C)(C)CCO)c2ccccc21
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2sc(COC)nc2C)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2sc(COC)nc2C)cs1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 28):
================================================================================
[1] NC1CCC(CO)Nc2ccn[nH]2)c1=O
[2] CCC(NC(=O)c1cccc(-c2ncccc1)C2
[3] COc1ccc(C(=O)C2CC2)cc1C#N
[4] CC(C)Oc1cccc(CN(C)C(F)(F)F)c1
[5] Cc1cc2ncc(C(=O)OCC)cc2s1
[6] NCC(C)CNC(=O)c1ccc2c(c1)OCCO2
[7] CCC(=O)NC(C)c1cccc(OC)c1
[8] Cc1c(=O)[nH]c2cccn2c1C(=O)Nc1ccc(C)s1
[9] COc1cccc(NC(C)C(=O)c2cnc(C)cc2)c1Br
[10] COc1ccc2c(c1)C(=O)C(c1cnn(C)c1)O2
================================================================================


================================================================================
Epoch 28 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0028
  Train - Loss: 0.1443, Recon: 0.1372, KL: 2.5167
  Val   - Loss: 43379.5263, Recon: 207.0984, KL: 15418724.5824
  Val Recon Acc:     31.49% (16297/51745)


================================================================================
Reconstruction Examples (Epoch 29):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)CNC(=O)NCCOC(=O)CC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)CNC(=O)NCCOC(=O)CC1CC1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)c1cccs1)Nc1ccc(F)cc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)c1cccs1)Nc1ccc(F)cc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC(=O)c2ccccc2OC)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC(=O)c2ccccc2OC)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1cccc(OC2CCC2)c1N(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1cccc(OC2CCC2)c1N(C)C
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Match: ✓

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(C(=O)N(C)CC(C)C)c(Br)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(C(=O)N(C)CC(C)C)c(Br)c1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CCN(C(=O)c2nnc(C(F)(F)F)s2)C1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CCN(C(=O)c2nnc(C(F)(F)F)s2)C1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)(CO)NC(=O)CCn1cc(CO)c2ccccc21
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(C)(CO)NC(=O)CCn1cc(CO)c2ccccc21
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2C)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2C)cs1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 29):
================================================================================
[1] COC(=O)C1CC1c1cccc(N(C)CC(Cl)Cl)c1
[2] C(=O)CCNC(=O)c1ccc(C(F)(F)F)c(COC)c1
[3] CCCN(C)C(=O)c1ccc(C)c2cn[nH]c21
[4] Cccc2ccc(O)cc12
[5] [nH]C(=O)c1cc(C(F)(F)F)cc2c1NCC(N)C2
[6] NC(=O)C1CC(O)c2cccc(OC)c2C1
[7] C(=O)NCC1CC1
[8] C1CCN(C(=O)c2ccc(C)nc2)CC1
[9] C)c1nc2c([nH]1)CCC(=O)OC2
[10] C1CCOC1C(=O)NC1CC1
================================================================================


================================================================================
Epoch 29 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0029
  Train - Loss: 0.1655, Recon: 0.1580, KL: 2.5854
  Val   - Loss: 4.0265, Recon: 4.0187, KL: 2.6709
  Val Recon Acc:     28.18% (14583/51745)


================================================================================
Reconstruction Examples (Epoch 30):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Match: ✓

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)c1cccs1)Nc1ccc(F)cc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)c1cccs1)Nc1ccc(F)cc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CC[nH]cc1nc2c(s1)C(=O)OCC(Nc1ccccc1)C2
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CC[nH]cc1nc2c(s1)C(=O)OCC(Nc1ccccc1)C2
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1cccc(N(C)C(=O)N2CCOCC2)c1C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1cccc(N(C)C(=O)N2CCOCC2)c1C
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)C(NCc1ccccc1)c1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)C(NCc1ccccc1)c1ccccc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COCCN(C)C(=O)c1ccc(Br)c(N(C)C)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COCCN(C)C(=O)c1ccc(Br)c(N(C)C)c1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC1CN(CC(F)(F)F)C(=O)N1c1nnc(CO)s1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC1CN(CC(F)(F)F)C(=O)N1c1nnc(CO)s1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cn1c(C)cc(C(=O)NCCC(O)CO)c1C
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cc1cc(C(=O)NCCC(O)CO)c(C)n1C
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1ncc(C(=O)NCCc2ccc(O)cc2)c1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1cc(C(=O)NCCc2ccc(O)cc2)cn1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2scnc2COC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2scnc2COC)cs1
    Match: ✓

================================================================================
Sampled Molecules from Latent Space (Epoch 30):
================================================================================
[1] CCC(NC(=O)N(C)C)c1ccc(F)nc1
[2] Cc1cc(C(=O)N2CCOCC2C)cc(C(C)C)s1
[3] CCCCCCCC2NC(=O)c3cccc(C)c32)c1
[4] Cn1nc(C)c2c1CC(CC(=O)Nc1cccs1)C2
[5] NN(CCOC)C(=O)c1ccc(F)cc1
[6] C1CCn2c(nc(NC(F)F)cc2=O)C1
[7] c1cc(-c2nccn2C(=O)OC(C)(C)C)c(C)o1
[8] CC(C)C(=O)Nc1cccc2cnn12
[9] CC(C)Oc1cc(C(=O)c2cnc3c(s2)cc(N)cc13)
[10] COc1c(C(=O)OCCO)nc2c(-c3cscc3)nc(C)cc
================================================================================


================================================================================
Epoch 30 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0030
  Train - Loss: 0.1822, Recon: 0.1743, KL: 2.6539
  Val   - Loss: 52.2345, Recon: 23.0873, KL: 9715.7357
  Val Recon Acc:     19.84% (10266/51745)


================================================================================
Reconstruction Examples (Epoch 31):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCC(=O)NC1CCC(C)(C)C1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC1(C)CCC(NC(=O)CNC(=O)OC(C)(C)C)C1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(CNC(=O)Nc1cccc(F)c1)c1cccs1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCOc1nnc(SC2CC(=O)Nc3ccccc32)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCOc1nnc(SC2CC(=O)Nc3ccccc32)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CN(C)C(=O)c1cccc(NC(=O)C2CCOC2)c1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CN(C)C(=O)c1cccc(NC(=O)C2CCOC2)c1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)Cc1cccc(NCc2ccccc2)c1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)Cc1cccc(NCc2ccccc2)c1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(Br)cc1C(=O)N(C)C(C)C
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(Br)cc1C(=O)N(C)C(C)C
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(=O)N1CC(c2nnc(C(F)(F)F)s2)CC1O
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(=O)N1CC(c2nnc(C(F)(F)F)s2)CC1O
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)(C)NC(=O)CCc1cn(C)c2ccccc12
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cn1cc(CCC(=O)NC(C)(C)C)c2ccccc21
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Match: ✓

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCNc1nc(-c2cnc(C)nc2OC)cs1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCNc1nc(-c2cnc(C)nc2OC)cs1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 31):
================================================================================
[1] CN(CCO)C(=O)c1ccc(F)cc1
[2] COC(=O)C(CC)NC(=O)c1ccc(N2CCC2)nc1
[3] CNc1ncc(-c2ccc3ccc(C)cc3s2)cn1
[4] CC(C)c1ccc(=O)[nH]c1NC(C)c1ccco1
[5] C=C(C)OCC(=O)Nc1cc(C(C)C)ccc1C
[6] C(=O)Cc1ccc2c(c1)OCCO2
[7] c1cc(C2CC(=O)N(CC(F)F)C2)c2cccnn2n1
[8] C(=O)Nc1cc(C)nc(Sc2nc(C3CC3)cs2)n1
[9] CC(C)c1ccccc1C(=O)Nc1ccccn1
[10] CCC(C)NC(=O)c1cc(SCCO)ccc1C
================================================================================


================================================================================
Epoch 31 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0031
  Train - Loss: 0.2460, Recon: 0.2380, KL: 2.6090
  Val   - Loss: 5.3440, Recon: 5.3360, KL: 2.5916
  Val Recon Acc:     9.17% (4745/51745)


================================================================================
Reconstruction Examples (Epoch 32):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)OC(=O)NCCNC(=O)C1CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)OC(=O)NCCNC(=O)C1CC1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COC(=O)NCC(=O)Nc1ccccc1F
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COC(=O)NCC(=O)Nc1ccccc1F
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=CCSc1nnc(NC(=O)c2ccccc2OC)s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=CCSc1nnc(NC(=O)c2ccccc2OC)s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CNC(=O)c1cccc(C(=O)OC(C)C)c1N1CCC1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CNC(=O)c1cccc(C(=O)OC(C)C)c1N1CCC1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: NC(=O)NCc1ccccc1NCc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   NC(=O)NCc1ccccc1NCc1ccccc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COc1ccc(C(=O)N(C)CC(C)C)cc1Br
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COc1ccc(C(=O)N(C)CC(C)C)cc1Br
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CN1CC(O)(c2nnc(C(F)(F)F)s2)C1=O
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CN1CC(O)(c2nnc(C(F)(F)F)s2)C1=O
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cc1ccccc1C(O)CCC(=O)N(C)Cc1ccccn1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cc1ccccc1C(O)CCC(=O)N(C)Cc1ccccn1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCn1ncc(C(=O)NCCO)c1=O
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCn1ncc(C(=O)NCCO)c1=O
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: COCCNc1ncnc2sc(-c3ccsc3)sc12
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   COCCNc1ncnc2sc(-c3ccsc3)sc12
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 32):
================================================================================
[1] CN(C)C(=O)Cc1ccc(Sc2ccc(C)s2)nc1
[2] COCC1CCCc2cccc(N(C)C)c21
[3] CC(=O)Nc1ccc2c(c1)C(=O)CCC2=O
[4] COc1ccc2ncnc(NC(=O)CC(C)C)c2ccccc12
[5] CCC(=O)c1cc(C(=O)C2(C)CC2)ncc1F
[6] CCc1nc(C(=O)NCCC#N)c2cccc(OC)c2o1
[7] CC1CC(C#N)CS(=O)(=O)c1ccc(C)n(C)c1=O
[8] CC(C)s1cccc(C(=O)N(C)CCO)n1
[9] NC(=O)C(CC)c1ccc(Br)cc1N
[10] CNC(=O)c1ccc(S(=O)(=O)Cc2ccc(C)s2)o1
================================================================================


================================================================================
Epoch 32 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0032
  Train - Loss: 0.2673, Recon: 0.2585, KL: 2.7399
  Val   - Loss: 5.4759, Recon: 5.4669, KL: 2.8132
  Val Recon Acc:     7.09% (3669/51745)


================================================================================
Reconstruction Examples (Epoch 33):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)CC(C)NC(=O)NC(=O)C(=O)NCC1CCCO1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)CC(C)NC(=O)NC(=O)C(=O)NCC1CCCO1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: O=C(Nc1cccc(F)c1)NCC1CC1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   O=C(NCC1CC1)Nc1cccc(F)c1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C=C=OCC(=O)C(O)CC(=O)Nc1nc2ccccc2s1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C=C=OCC(=O)C(O)CC(=O)Nc1nc2ccccc2s1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CN(CC(=O)NC(=O)Nc1cccc(OC)c1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CN(CC(=O)NC(=O)Nc1cccc(OC)c1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(=O)CC(=O)NCC1Cc2ccccc2C1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(=O)CC(=O)NCC1Cc2ccccc2C1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(=O)Cc1ccc(C)cc1O
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(=O)Cc1ccc(C)cc1O
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(=O)Nnc(N1C)csc(N)c(N)c(N)c(N)c(N)c(N)
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(=O)Nnc(N1C)csc(N)c(N)c(N)c(N)c(N)c(N)
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(C)C
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCn1ccc(CCCCCCCCCCCCCCCCCCCCCC(=O)CCCCC(
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCn1ccc(CCCCCCCCCCCCCCCCCCCCCC(=O)CCCCC(
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCc1ccc(-c2nc(C3CC3)c(NC)n2)s1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCc1ccc(-c2nc(C3CC3)c(NC)n2)s1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 33):
================================================================================
[1] COC(=O)CC(=O)Nc1ccc(F)cc1
[2] (FFF(((((((((((((((((((((((((((((((((
[3] c1ccc(NCC(C)C)cc1
[4] (C)CC(=O)NCc1ccc(C)cc1
[5] C)(C)C(=O)COC(=O)c1cccc(C)c1
[6] CCOC(=O)c2ccc(C)nc2)n1
[7] CCCN(C)C(=O)C(C)(C)Cc1cccs1
[8] CN1CCC(C(=O)N=c3ccc(O)cc3)CCC1
[9] OC(=O)C(C)Nc1ccccc1C
[10] CC(C)(CO)C(=O)Nc1ccc(C(N)=O)nc1
================================================================================


================================================================================
Epoch 33 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0033
  Train - Loss: 0.5993, Recon: 0.5903, KL: 2.7415
  Val   - Loss: 335.7868, Recon: 18.9593, KL: 96008.3630
  Val Recon Acc:     0.02% (12/51745)


================================================================================
Reconstruction Examples (Epoch 34):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)C(=O)NC(C)C(=O)NCC1CCCO1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(NC(=O)C(C)(C)C)C(=O)NCC1CCCO1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COc1ccc(CNC(=O)NCc2cccs2)cc1F
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COc1ccc(CNC(=O)NCc2cccs2)cc1F
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CCC(=O)NC1CCC(C)CC1=O
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCC(=O)NC1CCC(C)CC1=O
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: COc1ccccc1C(=O)NCCc1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   COc1ccccc1C(=O)NCCc1ccccc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CCCC(C)NC(=O)C(C)NC(=O)c1ccc(C)cc1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCCC(C)NC(=O)C(C)NC(=O)c1ccc(C)cc1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(C)(C)C(=O)Nc1nnc(C(F)(F)F)s1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(C)(C)C(=O)Nc1nnc(C(F)(F)F)s1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CCC(C)CCC(=O)N1CCC(C(=O)Nc2cccnc2)CC1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CCC(C)CCC(=O)N1CCC(C(=O)Nc2cccnc2)CC1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCC(=O)NC(C)c1ccc(-n2cccn2)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCC(=O)NC(C)c1ccc(-n2cccn2)cc1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCCNc1ncnc2sc(C)cc12
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCCNc1ncnc2sc(C)cc12
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 34):
================================================================================
[1] CNC(=O)CNC(=O)CC1CC1)c1ccc(C(C)(C)C)o
[2] 1cc(Br)cc1C(=O)NC(C)CC
[3] CCOc1ccccc1-c1ccc(-c2ccccc2)c(C)n1
[4] COC(=O)c1ccc(C(=O)N2CCCC2)c(Cl)c1
[5] CC(C)C(=O)NCC(=O)NCc1ccccc1Cl
[6] OCC(NC(=O)c1cc(F)c(Cl)c(Cl)c1)C1CC1
[7] CC(=O)NCC(C)C)n1
[8] c1cccc(CNC(=O)NCC(C)C)c1
[9] CCCNC(=O)CCC(=O)NCc1ccc(C(N)=O)nc1
[10] CCC(C)c1cccc(OCc2ccccc2)c1OC
================================================================================


================================================================================
Epoch 34 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0034
  Train - Loss: 0.9244, Recon: 0.9153, KL: 2.6758
  Val   - Loss: 8020988640778.4424, Recon: 215130.4200, KL: 2359114358795755.5000
  Val Recon Acc:     0.00% (1/51745)


================================================================================
Reconstruction Examples (Epoch 35):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)C(=O)NCC(C)C(=O)NC1CCCC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)C(=O)NCC(C)C(=O)NC1CCCC1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)C(=O)Nc1ccc(C(=O)Nc2cccs2)cc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)C(=O)Nc1ccc(C(=O)Nc2cccs2)cc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CCc1cccc1CNC(=O)CCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCc1cccc1CNC(=O)CCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(C)C(=O)NCC(C)C)C(=O)c1ccc(F)cc1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CC(C)C(=O)NCC(C)C)C(=O)c1ccc(F)cc1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(C)CN(CC)C(=O)c1ccccc1N
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CCN(CC(C)C)C(=O)c1ccccc1N
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(C)C(=O)NCC(C)C(=O)c1ccc(F)cc1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(C)C(=O)NCC(C)C(=O)c1ccc(F)cc1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(C)C(=O)NCC(C)(C)C(=O)Nc1cccs1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(C)C(=O)NCC(C)(C)C(=O)Nc1cccs1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)CN(C)C(=O)c1ccc(F)cc1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(C)CN(C)C(=O)c1ccc(F)cc1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(C)C(=O)NCc1ccccc1C(=O)NCCC(C)C
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(C)CCNC(=O)c1ccccc1CNC(=O)C(C)C
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 35):
================================================================================
[1] CCN(C(=O)c1ccc(C)cc1)C(C)C
[2] CC(=O)c1ccc(C(=O)NCCC(C)C)c(C)o1
[3] COCCN(C(=O)COc2ccc(F)cc2F)C1
[4] CC(C)CNC(=O)c1ccccc1
[5] c1ccccc1C(=O)NCc1ccoc1
[6] CNc1ccc(C(=O)NCC(C)C)cc1
[7] CCc1ccc(C(=O)Nc2nccc(C)c2)cc1
[8] c2ccccc2c1ccccc1N
[9] O(CC(=O)F)C1CC1
[10] OCCOCCOCC1CC1
================================================================================


================================================================================
Epoch 35 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0035
  Train - Loss: 1.0110, Recon: 0.9990, KL: 3.4086
  Val   - Loss: 12805.7729, Recon: 7.2386, KL: 3656723.9556
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 36):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CCOC(=O)NC(=O)CCC(=O)NCC(C)C)C(=O)NCC(C)
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCOC(=O)NC(=O)CCC(=O)NCC(C)C)C(=O)NCC(C)
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)C(=O)NC(=O)c1ccc(C(=O)NCCO)cc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)C(=O)NC(=O)c1ccc(C(=O)NCCO)cc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CCOC(=O)c1ccc(C(=O)NCC2CCCC2)cn1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCOC(=O)c1ccc(C(=O)NCC2CCCC2)cn1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(C)C(=O)NC(=O)CCc1ccccc1F
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CC(C)C(=O)NC(=O)CCc1ccccc1F
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: O=C(NCCc1ccccc1)NCC1CCCCC1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   O=C(NCCc1ccccc1)NCC1CCCCC1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(C)C(=O)NC(=O)CCC(=O)NCC(C)C)C(N)=O)c1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(C)C(=O)NC(=O)CCC(=O)NCC(C)C)C(N)=O)c1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: Cc1cc(C(=O)Nc2ccc(Cl)cc2)nc2ccccc21
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   Cc1cc(C(=O)Nc2ccc(Cl)cc2)nc2ccccc12
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)C(=O)Nc1ccc(C)cc2)c1ccccc1)c1ccccc1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(C)C(=O)Nc1ccc(C)cc2)c1ccccc1)c1ccccc1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(C)C(=O)NC(=O)c1ccc(C(=O)NCC2CC2)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(C)C(=O)NC(=O)c1ccc(C(=O)NCC2CC2)cc1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCCNC(=O)CCNC(=O)c1ccc(C)cc1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCCNC(=O)CCNC(=O)c1ccc(C)cc1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 36):
================================================================================
[1] CC(C)C(=O)N1CCCCC1CC1CC1
[2] CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
[3] Nc1ccc(F)cc1F)c1ccc(F)cc1F
[4] CCCC(=O)NCC1CC1)c1ccccc1F
[5] COc1ccccc1C(=O)Nc1ccc(F)cc1
[6] Cc1ccc(C(=O)NCc2ccc(N)cc2)cc1
[7] COC(=O)CN(C)C(=O)c1ccc(Cl)cc1
[8] Cc1ccc(F)c1)NC(=O)CCC(N)=O
[9] CCNc2ccc(Br)cc2)n1
[10] c1cccc1-n1cccn1)NCC(C)C
================================================================================


================================================================================
Epoch 36 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0036
  Train - Loss: 0.9067, Recon: 0.8979, KL: 2.4560
  Val   - Loss: inf, Recon: 5773563025747542.0000, KL: inf
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 37):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CCCCC(=O)NCC(=O)NCC(=O)NCC(C)C
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCCCC(=O)NCC(=O)NCC(=O)NCC(C)C
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1Cl
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1Cl
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CCCCCCNC(=O)CCNC(=O)c1ccc(C)cc1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCCCCCNC(=O)CCNC(=O)c1ccc(C)cc1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(C)CNC(=O)CCC(=O)Nc1ccc(C)cc1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   Cc1ccc(NC(=O)CCC(=O)NCC(C)C)cc1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: Cc1ccc(C(=O)NCC(=O)Nc2ccccc2)cc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   Cc1ccc(C(=O)NCC(=O)Nc2ccccc2)cc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(C)CN(C)C(=O)CC(C)C(N)=O)c1ccc(F)cc1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(C)CN(C)C(=O)CC(C)C(N)=O)c1ccc(F)cc1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CCCCC(C)NC(=O)CC(=O)NCc1ccccc1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CCCCC(C)NC(=O)CC(=O)NCc1ccccc1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)NC(=O)c1ccc(-c2ccccc2)cc1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(C)NC(=O)c1ccc(-c2ccccc2)cc1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCCCN(C)C(=O)c1ccc(C(=O)NCCO)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCCCN(C)C(=O)c1ccc(C(=O)NCCO)cc1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCOC(=O)CNC(=O)c1ccc(-c2ccccc2)cc1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCOC(=O)CNC(=O)c1ccc(-c2ccccc2)cc1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 37):
================================================================================
[1] CCCNC(=O)C(C)NC(=O)c1ccc(C)cc1
[2] cccc1NCc1ccccc1C(=O)NCC(F)(F)F
[3] O)c2ccc(C(F)(F)F)ccc2c(C)c1C(N)=O
[4] F)(CCO)C(=O)NCC(F)(F)F)c1ccccc1
[5] =O)CCCCC(=O)NCC(C)C)cc1C
[6] C(=O)OCC(=O)N2CCOCC2)s1
[7] Oc1ccccc1C(=O)NC1CCCCC1
[8] )OCCO)c1
[9] CCC(=O)NCC(=O)NCC1CCCCC1
[10] CC(C)NC(=O)CSc1ccc(C)cc1C
================================================================================


================================================================================
Epoch 37 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0037
  Train - Loss: 0.8771, Recon: 0.8684, KL: 2.3464
  Val   - Loss: 5757219898091570196381696.0000, Recon: 9589361551529.3711, KL: 1556005344142652163913416704.0000
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 38):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)C(=O)NCC(=O)NCC(C)(C)C)C(=O)NCC(C)C
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)C(=O)NCC(=O)NCC(C)(C)C)C(=O)NCC(C)C
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)C(=O)NCC(=O)NCC(=O)NCC(C)C
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)CNC(=O)CNC(=O)CNC(=O)C(C)C
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1Cl
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1Cl
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CCCCCC(=O)NCC(=O)NCCC(C)(C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCCCCC(=O)NCC(=O)NCCC(C)(C)C
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(C)NC(=O)Nc1ccc(C(=O)NCC(F)(F)F)cc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(C)NC(=O)Nc1ccc(C(=O)NCC(F)(F)F)cc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(C)NC(=O)CNC(=O)c1ccc(Cl)cc1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(C)NC(=O)CNC(=O)c1ccc(Cl)cc1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(C)NC(=O)CNC(=O)c1ccc(C(=O)NCC(C)C)cc1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(C)CNC(=O)c1ccc(C(=O)NCC(=O)NC(C)C)cc1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(C(C)=O)cc1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(=O)c1ccc(NC(=O)CNC(=O)C(C)C)cc1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(C)C(=O)Nc1ccc(C(=O)NCC(C)C)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(C)CNC(=O)c1ccc(NC(=O)C(C)C)cc1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(C(N)=O)cc1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccc(C(N)=O)cc1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 38):
================================================================================
[1] CCCCNC(=O)CCC(N)=O)CC1
[2] Cc1ccccc1-n1cnc2ccccc21
[3] CC(=O)NCC(F)(F)F
[4] CCCNC(=O)NCC(=O)NC
[5] CCOCC1CCCC1
[6] c1ccccc1-n1nccc1-c1ccco1
[7] S334244444444nnnc(=O)c2)cc1NC(C)C
[8] O
[9] OCCNC(=O)c1ccc(F)cc1F
[10] CCCN(Cc1ccccc1)C(=O)c1ccccc1
================================================================================


================================================================================
Epoch 38 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0038
  Train - Loss: 0.8597, Recon: 0.8514, KL: 2.1811
  Val   - Loss: 7.4396, Recon: 7.0920, KL: 91.4739
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 39):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)C(=O)NCCCCC(=O)NCCCC1CCCCCC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)C(=O)NCCCCC(=O)NCCCC1CCCCCC1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)C(=O)NCCCN(C)C(=O)c1ccc(Br)cc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)C(=O)NCCCN(C)C(=O)c1ccc(Br)cc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(C)C(=O)NCC(=O)NCCCC(=O)NCC1CCCCC1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(C)C(=O)NCC(=O)NCCCC(=O)NCC1CCCCC1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(C)C(=O)NCC(C)C(=O)NCC1CCCCC1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CC(C)C(=O)NCC(C)C(=O)NCC1CCCCC1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(C)C(=O)NCCC(C)c1ccc(C(N)=O)cc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(C)C(=O)NCCC(C)c1ccc(C(N)=O)cc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(C)C(=O)NCCC(=O)NCCCC1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(C)C(=O)NCCC(=O)NCCCC1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(C)C(=O)NCC(C)C(=O)NCC1CCCC1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(C)C(=O)NCC(C)C(=O)NCC1CCCC1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)C(=O)NCCN(C)C(=O)c1ccc(Cl)cc1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(C)C(=O)NCCN(C)C(=O)c1ccc(Cl)cc1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(C)C(=O)NCC(=O)NCC(C)C)C(C)C)C(C)C)C(C
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(C)C(=O)NCC(=O)NCC(C)C)C(C)C)C(C)C)C(C
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(C)C(=O)NCCC(=O)NCCc1ccccc1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(C)C(=O)NCCC(=O)NCCc1ccccc1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 39):
================================================================================
[1] CN(C)C(=O)CCCCC(C)C)CC1
[2] CCNC(=O)c1ccccc1Cl
[3] NN22CC2)n1
[4] CCN(C)S(=O)(=O)c1ccccc1
[5] Nc(=O)c1NC(=O)c2ccccc2)cc1
[6] CC(=O)NCC(=O)NCc1ccco1
[7] OCCNC(=O)c1ccccc1
[8] CCCCCCCCCCC2)cc1
[9] ccc(NC(=O)c2ccco2)cc1
[10] CCC1CC1)c1ccccc1
================================================================================


================================================================================
Epoch 39 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0039
  Train - Loss: 0.8615, Recon: 0.8528, KL: 2.2242
  Val   - Loss: 5.1035, Recon: 5.0925, KL: 2.8205
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 40):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CCCCN(CC)C(=O)CCC(=O)Nc1ccc(F)cc1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCCCN(CC)C(=O)CCC(=O)Nc1ccc(F)cc1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)CCNC(=O)c1ccc(NC(=O)C2CC2)cc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)CCNC(=O)c1ccc(NC(=O)C2CC2)cc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CCCCNC(=O)c1ccc(NC(=O)NCC(C)C)cc1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCCCNC(=O)c1ccc(NC(=O)NCC(C)C)cc1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CCC(C)NC(=O)C(C)(C)C(=O)Nc1ccccc1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCC(C)NC(=O)C(C)(C)C(=O)Nc1ccccc1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(C)CN(C)C(=O)Cc1ccc(C(N)=O)cc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(C)CN(C)C(=O)Cc1ccc(C(N)=O)cc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CCC(C)C(=O)NCC(=O)Nc1ccc(F)cc1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCC(C)C(=O)NCC(=O)Nc1ccc(F)cc1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CCC(C)NC(=O)CCC(=O)Nc1ccc(F)cc1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CCC(C)NC(=O)CCC(=O)Nc1ccc(F)cc1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CCCC(=O)NCC(=O)Nc1ccc(C)cc1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CCCC(=O)NCC(=O)Nc1ccc(C)cc1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: Cc1ccc(C(=O)NCC(=O)NCC(C)C)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   Cc1ccc(C(=O)NCC(=O)NCC(C)C)cc1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(C)CCNC(=O)c1ccc(C(=O)NC(C)C)cc1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(C)CCNC(=O)c1ccc(C(=O)NC(C)C)cc1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 40):
================================================================================
[1] C)(C)C(=O)Nc1ccccc1
[2] 3333333333333333333333333333333333333
[3] (=O)c2ccccc2)c2ccccc21
[4] CCO)c1
[5] Nc2ccc(F)cc2c1
[6] CCC(=O)N(C)C(C)C)c1
[7] CCN(CCO)C(=O)c1ccccc1
[8] Cc1ccccc1NCc1ccccc1
[9] CCN(C(=O)c2ccccc2)C1
[10] CCCN1CCCCC1
================================================================================


================================================================================
Epoch 40 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0040
  Train - Loss: 0.7558, Recon: 0.7485, KL: 1.8268
  Val   - Loss: 6.7872, Recon: 6.7795, KL: 1.9266
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 41):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CCC(C)NC(=O)CCC(=O)NCC(C)C(C)C
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCC(C)NC(=O)CCC(=O)NCC(C)C(C)C
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)C(C)NC(=O)CNC(=O)c1ccc(Cl)cc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)C(C)NC(=O)CNC(=O)c1ccc(Cl)cc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(C)C(C)NC(=O)CNC(=O)c1ccc(Cl)cc1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(C)C(C)NC(=O)CNC(=O)c1ccc(Cl)cc1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(C)cc1C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   Cc1ccc(NC(=O)CNC(=O)C(C)C)c(C)c1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1Cl
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1Cl
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(-n2cnnn2)cc1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccc(-n2cnnn2)cc1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(C(F)(F)F)cc1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccc(C(F)(F)F)cc1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: Cc1ccc(C(=O)Nc2ccc(C(N)=O)cc2)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   Cc1ccc(C(=O)Nc2ccc(C(N)=O)cc2)cc1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: Cc1ccc(C(=O)Nc2ccc(C(N)=O)cc2)cc1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   Cc1ccc(C(=O)Nc2ccc(C(N)=O)cc2)cc1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 41):
================================================================================
[1] (()(Cl)(Cl)(=O)(Cl)(F)(F)(F)(F)(F)F)CCCC
[2] ccc(NC(=O)CNC(=O)C(C)(C)C)c1
[3] CCCCCCCCCC(=O)NCCCCCCCCCCCCCCCCCCCCCC
[4] 1Cc1ccc(F)cc1C(=O)NCC(F)(F)F
[5] CCCC(C)NC(=O)c1ccc(C)c(C)c1
[6] Cc1nnc(-c2ccc(Cl)cc2)n1C1CC1
[7] c2ccccc2NCC1
[8] CC(=O)Nc1ccc(-c2ccccc2)cc1
[9] Nc1ccccc1OCc1ccccc1
[10] CC(C)C(=O)Nc1ccc(C(=O)NCC2CC2)cc1
================================================================================


================================================================================
Epoch 41 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0041
  Train - Loss: 0.7212, Recon: 0.7141, KL: 1.7210
  Val   - Loss: 7.0064, Recon: 7.0004, KL: 1.4587
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 42):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CCCCCCC(C)NC(=O)CCC(=O)Nc1ccccc1C
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCCCCCC(C)NC(=O)CCC(=O)Nc1ccccc1C
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: O=C(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   O=C(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CCCCC(=O)NCCC(=O)Nc1ccccc1C
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCCCC(=O)NCCC(=O)Nc1ccccc1C
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CCCCCCC(C)NC(=O)CCc1ccc(C)cc1C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCCCCCC(C)NC(=O)CCc1ccc(C)cc1C
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CCCCCCCCCCCCCCCC(C)NC(=O)CCC(=O)Nc1ccccc
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCCCCCCCCCCCCCCC(C)NC(=O)CCC(=O)Nc1ccccc
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CCCCCCCCCCCCCC(=O)Nc1cccc(C(=O)NCCCCCCCC
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CCCCCCCCCCCCCC(=O)Nc1cccc(C(=O)NCCCCCCCC
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CCCCCCC(=O)NCCC(=O)Nc1ccccc1C
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CCCCCCC(=O)NCCC(=O)Nc1ccccc1C
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCCCCC(C)NC(=O)CCC(=O)Nc1ccccc1C
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCCCCC(C)NC(=O)CCC(=O)Nc1ccccc1C
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCCCCC(C)NC(=O)CCC(=O)Nc1ccccc1C
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCCCCC(C)NC(=O)CCC(=O)Nc1ccccc1C
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 42):
================================================================================
[1] cccc1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
[2] C(C)CCCCC(=O)Nc2ccccc2C)c1
[3] NC(=O)CCCCCCCCCCCCCCCOc2ccccc1)c2cccc
[4] CCCCCC(=O)NCC(=O)NCCCCCCCCCCCCCCCCCCC
[5] OCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
[6] )=OCCCC(=O)Nc1ccccc1C
[7] CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
[8] BrC(C)CCCCC1
[9] CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
[10] CCC(=O)NCCC(=O)NCCCCCCCCCCCCCCCCCCCCC
================================================================================


================================================================================
Epoch 42 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0042
  Train - Loss: 0.9096, Recon: 0.9019, KL: 1.8322
  Val   - Loss: 4.2759, Recon: 4.2681, KL: 1.8650
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 43):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(F)cc1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccc(F)cc1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)C(=O)NCC(=O)NCC(=O)Nc1ccccc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)C(=O)NCC(=O)NCC(=O)Nc1ccccc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(F)cc1F
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccc(F)cc1F
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   Cc1cc(Cl)ccc1NC(=O)CNC(=O)C(C)C
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: O=C(CCC1CCCC1)c1ccc(F)cc1F
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   O=C(CCC1CCCC1)c1ccc(F)cc1F
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(C(C)=O)cc1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(=O)c1ccc(NC(=O)CNC(=O)C(C)C)cc1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(C)NC(=O)CNC(=O)c1ccc(F)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(C)NC(=O)CNC(=O)c1ccc(F)cc1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(F)cc1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccc(F)cc1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 43):
================================================================================
[1] COC(=O)CCC(=O)NCC(C)C
[2] c1ccc(Cl)cc1
[3] OC(C)CC(=O)N1CCCC(C(=O)NCCCO)C1CC1
[4] C1CC1)CCC2(C)C1CCCCC1
[5] 3N3(CCO3)CCCC3)cc1
[6] O=C(NCC(F)(F)F)c1ccccc1
[7] N(=O)CCC(=O)NCC(F)(F)F)C1CC1
[8] c2ccccc2OCC1
[9] 111O1c(=O)n2cccnc12
[10] OOC(=O)NCCC(=O)NCC(C)C
================================================================================


================================================================================
Epoch 43 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0043
  Train - Loss: 1.1373, Recon: 1.1283, KL: 2.0721
  Val   - Loss: 6.4800, Recon: 6.4628, KL: 4.0087
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 44):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)CC(=O)NCC(=O)Nc1ccccc1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)CC(=O)NCC(=O)Nc1ccccc1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)C(=O)Nc1ccc(NC(=O)c2ccco2)cc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)C(=O)Nc1ccc(NC(=O)c2ccco2)cc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CCCNC(=O)CN(C)C(=O)c1ccc(C)cc1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCCNC(=O)CN(C)C(=O)c1ccc(C)cc1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(C)C(=O)Nc1ccc(NC(=O)c2cccs2)cc1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CC(C)C(=O)Nc1ccc(NC(=O)c2cccs2)cc1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: O=C(NCC1CC1)c1ccc(Br)cc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   O=C(NCC1CC1)c1ccc(Br)cc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(C)CC(=O)Nc1ccc(C(N)=O)cc1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(C)CC(=O)Nc1ccc(C(N)=O)cc1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(C)C(=O)Nc1ccc(NC(=O)c2cccs2)cc1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(C)C(=O)Nc1ccc(NC(=O)c2cccs2)cc1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)CC(=O)NCC(=O)Nc1ccc(C)cc1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cc1ccc(NC(=O)CNC(=O)CC(C)C)cc1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(C)CC(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(C)CC(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(C)CN(C)C(=O)CNc1ccc(F)c(F)c1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(C)CN(C)C(=O)CNc1ccc(F)c(F)c1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 44):
================================================================================
[1] Brc1ccc(F)c1
[2] c4ncn4nnnnnnnnnnnnnnnnnnnnnnnnnnn444n
[3] c1ccc(Cl)cc1Cl
[4] CCCC(=O)NCC(=O)NCC(C)(C)C)C(=O)NC(C)C
[5] CC(CNC(=O)c1cccs1)NCC1CC1
[6] c(=O)CCCNC(=O)c1cccc(Cl)c1
[7] COCCNC(=O)c1ccc(F)c(F)c1
[8] CCN(CC(=O)OC)C(C)(C)C)C(C)C(C)C
[9] (C(C)C1CCCC1)C(F)(F)F
[10] O=C(NCc1ccccc1)c1ccc(F)c(F)c1
================================================================================


================================================================================
Epoch 44 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0044
  Train - Loss: 0.9899, Recon: 0.9830, KL: 1.5808
  Val   - Loss: 6.8406, Recon: 6.8342, KL: 1.4697
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 45):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: Cc1ccc(C(C)C)C(=O)C(C)C(C)C(=O)CCCC(C)CC
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   Cc1ccc(C(C)C)C(=O)C(C)C(C)C(=O)CCCC(C)CC
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: Cc1ccc(CCCC(=O)CCC(C)CCCCCCCCCCCCCCCCCCC
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   Cc1ccc(CCCC(=O)CCC(C)CCCCCCCCCCCCCCCCCCC
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(CC(CCCC(=O)CCCC(C)C(=O)C(C)C(=O)C(C)C
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(CC(CCCC(=O)CCCC(C)C(=O)C(C)C(=O)C(C)C
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CCCCCCCCC(C)C(=O)CCCCC(=O)CCC(=O)CCCCC(=
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCCCCCCCC(C)C(=O)CCCCC(=O)CCC(=O)CCCCC(=
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: Cc1ccc(C)CCC(C)CCC(=O)CCCCCCCCC(=O)CCCCC
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   Cc1ccc(C)CCC(C)CCC(=O)CCCCCCCCC(=O)CCCCC
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: Cc1nc(C)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   Cc1nc(C)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cc1ccc(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cc1ccc(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: Cc1nc(C)CCCCC(C)CCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   Cc1nc(C)CCCCC(C)CCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: Cc1ccc(C(=O)N(C)C(=O)N(CCCCCCCCCCCCCCCCC
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   Cc1ccc(C(=O)N(C)C(=O)N(CCCCCCCCCCCCCCCCC
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 45):
================================================================================
[1] 2CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC2
[2] c1ccccc1-c1ccccc1Oc1ccccc1
[3] s2
[4] Nc1ccc(C(=O)NCC2CCCO2)cc1
[5] cccc12
[6] CC1CCC(C)NC2C(C)=O)CC1
[7] CCCC(C)C(C)OC(C)C(N)=O
[8] NC(=O)CNC(=O)c1ccc(F)cc1F
[9] CC(=O)NCC(=O)NCC(C)(C)C
[10] 2C2O3O3C2C3CC3C2CC2
================================================================================


================================================================================
Epoch 45 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0045
  Train - Loss: 0.8654, Recon: 0.8594, KL: 1.3367
  Val   - Loss: 4.2879, Recon: 4.2803, KL: 1.6891
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 46):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1Cl
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1Cl
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)(C)C(=O)NCC(=O)NCC(=O)Nc1ccc(F)cc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)(C)C(=O)NCC(=O)NCC(=O)Nc1ccc(F)cc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(C)(C)NC(=O)CCC(=O)Nc1ccc(Cl)cc1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(C)(C)NC(=O)CCC(=O)Nc1ccc(Cl)cc1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(C)CNC(=O)CCC(=O)Nc1ccc(F)cc1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CC(C)CNC(=O)CCC(=O)Nc1ccc(F)cc1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(C)(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(C)(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(C)CC(=O)NCC(=O)Nc1ccc(F)cc1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(C)CC(=O)NCC(=O)Nc1ccc(F)cc1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(C)(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(C)(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)(C)C(=O)NCC(=O)Nc1ccc(F)cc1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(C)(C)C(=O)NCC(=O)Nc1ccc(F)cc1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(C)CNC(=O)CCC(=O)Nc1ccc(F)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(C)CNC(=O)CCC(=O)Nc1ccc(F)cc1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(C)(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(C)(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 46):
================================================================================
[1] CC(=O)NCC(=O)NCC(=O)NC(C)(C)C
[2] ccccc2ccccc2N1CCO
[3] CC(C)NC(=O)CCC(=O)Nc1ccccc1
[4] CC(=O)N1CCCCC1
[5] CCCN(C)C(=O)c1ccc(C(C)=O)cc1
[6] c1ccc(F)cc1F
[7] 1)n1nnc2c(C)cccc21
[8] NC(=O)c1ccc(OCC(F)F)cc1
[9] CCCNC(=O)NCC(=O)Nc1ccccc1
[10] CC(C)C(=O)NCC(=O)Nc1ccc(F)cc1
================================================================================


================================================================================
Epoch 46 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0046
  Train - Loss: 0.7258, Recon: 0.7198, KL: 1.3008
  Val   - Loss: 7.0907, Recon: 7.0851, KL: 1.2112
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 47):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)CNC(=O)CCC(=O)Nc1ccc(F)cc1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)CNC(=O)CCC(=O)Nc1ccc(F)cc1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)CC(=O)Nc1ccc(C(=O)NCC(F)(F)F)cc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)CC(=O)Nc1ccc(C(=O)NCC(F)(F)F)cc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccccc1Cl
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccccc1Cl
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(C)CCC(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CC(C)CCC(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(C)(C)C(=O)Nc1ccc(C(=O)N2CCCC2)cc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(C)(C)C(=O)Nc1ccc(C(=O)N2CCCC2)cc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(C)(C)C(=O)Nc1ccc(NC(=O)C2CCCC2)cc1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(C)(C)C(=O)Nc1ccc(NC(=O)C2CCCC2)cc1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(C)CC(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(C)CC(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)CC(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(C)CC(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(C)(C)C(=O)NCC(=O)Nc1ccccc1Cl
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(C)(C)C(=O)NCC(=O)Nc1ccccc1Cl
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(C)(C)NC(=O)CNC(=O)c1ccccc1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(C)(C)NC(=O)CNC(=O)c1ccccc1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 47):
================================================================================
[1] CC(C)NC(=O)NCC(=O)NC1CCCC1
[2] Cc1ccc(C(=O)Nc2ccccc2)cc1
[3] cccc1OCC(=O)Nc1ccccc1
[4] n1n1C1CCCC1C(=O)NCC(C)C
[5] CC(=O)N1CCC(C(=O)NC(C)C)CC1
[6] CC(C)CN(CC(=O)NC1CCCC1)c1ccccc1
[7] Cc1ccc(C(=O)Nc2ccc(C(=O)N(C)C)CC2)cc1
[8] CC(=O)COCCO)c(=O)[nH]c1=O
[9] Cn1cccc1C(=O)NC1CCCCC1
[10] CCCNC(=O)C1CCCN1C(=O)c1ccccc1
================================================================================


================================================================================
Epoch 47 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0047
  Train - Loss: 0.7153, Recon: 0.7103, KL: 1.0494
  Val   - Loss: 7.3876, Recon: 7.3838, KL: 0.8004
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 48):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)CC(=O)NCC(=O)Nc1ccc(C(F)(F)F)cc1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)CC(=O)NCC(=O)Nc1ccc(C(F)(F)F)cc1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)CC(=O)NCC(=O)Nc1ccc(C(F)(F)F)cc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)CC(=O)NCC(=O)Nc1ccc(C(F)(F)F)cc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(C)NC(=O)CNC(=O)c1ccc(Cl)cc1Cl
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(C)NC(=O)CNC(=O)c1ccc(Cl)cc1Cl
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(C)CC(=O)NCC(=O)Nc1ccc(Cl)cc1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CC(C)CC(=O)NCC(=O)Nc1ccc(Cl)cc1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(C(F)(F)F)cc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccc(C(F)(F)F)cc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(C)CNC(=O)CNC(=O)c1ccc(OC(F)F)cc1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(C)CNC(=O)CNC(=O)c1ccc(OC(F)F)cc1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(C)NC(=O)CNC(=O)c1ccc(NC(=O)C2CC2)cc1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(C)NC(=O)CNC(=O)c1ccc(NC(=O)C2CC2)cc1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)NC(=O)CNC(=O)c1ccc(Cl)cc1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(C)NC(=O)CNC(=O)c1ccc(Cl)cc1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(C)CC(=O)NCC(=O)Nc1ccc(C(F)(F)F)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(C)CC(=O)NCC(=O)Nc1ccc(C(F)(F)F)cc1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(C)NC(=O)CNC(=O)c1ccc(Cl)cc1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(C)NC(=O)CNC(=O)c1ccc(Cl)cc1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 48):
================================================================================
[1] CCCCN(C(=O)c1ccccc1)C(C)C
[2] CN(CC)C(=O)CN(C)C(=O)c1ccc(C)c(C)c1)C
[3] C(=O)c1ccc(C)cc1
[4] NC(=O)c1ccc(NC(=O)C2CC2)cc1
[5] c1Nc2ccccc2c1-c1cccc(Cl)c1
[6] CC(=O)NCC(=O)NCC(F)(F)F
[7] C(C)N(C)C(=O)c1ccc(F)cc1F
[8] CC(=O)NCC(=O)Nc1ccccc1
[9] (=O)c1ccc(NC(=O)c2ccccc2)cc1
[10] CCC1)C1CC1
================================================================================


================================================================================
Epoch 48 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0048
  Train - Loss: 1.0835, Recon: 1.0779, KL: 1.1635
  Val   - Loss: 7.1475, Recon: 7.1394, KL: 1.6937
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 49):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1Cl
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1Cl
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1Cl
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1Cl
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1Cl
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1Cl
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CCC(C)NC(=O)CN(C)C(=O)c1ccccc1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCC(C)NC(=O)CN(C)C(=O)c1ccccc1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(C(F)(F)F)cc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccc(C(F)(F)F)cc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CCC(C)NC(=O)CNC(=O)c1ccc(F)cc1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCC(C)NC(=O)CNC(=O)c1ccc(F)cc1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CCC(C)NC(=O)CNC(=O)c1ccc(F)cc1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CCC(C)NC(=O)CNC(=O)c1ccc(F)cc1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1Cl
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1Cl
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(C)C(=O)Nc1ccc(C(=O)NCC2CC2)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(C)C(=O)Nc1ccc(C(=O)NCC2CC2)cc1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(C)(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1Cl
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(C)(C)C(=O)NCC(=O)Nc1ccc(Cl)cc1Cl
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 49):
================================================================================
[1] 111C1CCCC1)N1CCN(C(=O)C2CC2)CC1
[2] CCCC1CCCO1
[3] CCOC(=O)CCC(=O)Nc1ccccc1C
[4] Cc1ccc(C(=O)NCC(=O)NC(C)C)cc1
[5] c1-c1ccccc1
[6] CC(O)(CC(F)(F)F)C1CCCC1
[7] CC(=O)NCC(=O)Nc1cccc(C(N)=O)c1
[8] CN(C)C(=O)NCC(C)C
[9] Cc1ccc(C(=O)NCC(=O)NCC2CC2)cc1
[10] CC(C(=O)Nc1ccccc1)c1ccccc1
================================================================================


================================================================================
Epoch 49 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0049
  Train - Loss: 0.8518, Recon: 0.8446, KL: 1.4773
  Val   - Loss: 16359.2643, Recon: 360.7884, KL: 3264995.0942
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 50):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)CC(=O)Nc1ccc(NC(=O)CCC#N)cc1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)CC(=O)Nc1ccc(NC(=O)CCC#N)cc1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)CC(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)CC(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(C)CC(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(C)CC(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(C)C(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CC(C)C(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(C)CC(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(C)CC(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(C)CC(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(C)CC(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(C)CC(=O)Nc1ccc(C(=O)NCC2CC2)cc1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(C)CC(=O)Nc1ccc(C(=O)NCC2CC2)cc1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)CC(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(C)CC(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(C)CC(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(C)CC(=O)Nc1ccc(NC(=O)C2CC2)cc1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(C)C(=O)Nc1ccc(NC(=O)c2ccco2)cc1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(C)C(=O)Nc1ccc(NC(=O)c2ccco2)cc1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 50):
================================================================================
[1] Cc1cccc2CC2
[2] C1CCCC1)C1CCCCC1
[3] Cc1ccc(C(=O)NCC2CCCCC2)cc1
[4] ))ccc2c12
[5] Cc1ccc(C(=O)NCC(=O)N2CCCCC2)cc1
[6] Cln1nnc2ccccc12
[7] C1CCCN(C(=O)c2ccc(C)cc2)C1
[8] CC(C)NC(=O)CNC(=O)c1ccc(Cl)cc1
[9] 2222C(F)(F)F)cn12
[10] 2NCCC2OCCO1
================================================================================


================================================================================
Epoch 50 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0050
  Train - Loss: 0.8220, Recon: 0.8154, KL: 1.3331
  Val   - Loss: 6.7007, Recon: 6.6952, KL: 1.1128
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 51):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CCC(C)C(=O)NC(=O)NC(=O)c1ccccc1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCC(C)C(=O)NC(=O)NC(=O)c1ccccc1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)C(=O)NC(=O)NC(=O)NC(=O)NC(=O)NC(=O)
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)C(=O)NC(=O)NC(=O)NC(=O)NC(=O)NC=O
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(C)c1ccc(C(C)C(C)C(C)CC(C)CC2)cc1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(C)c1ccc(C(C)C(C)C(C)CC(C)CC2)cc1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: COc1ccc(C(=O)NCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   COc1ccc(C(=O)NCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(C)C(=O)NC(=O)c1cccc(NC(=O)CCCC2)c1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(C)C(=O)NC(=O)c1cccc(NC(=O)CCCC2)c1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CCC(=O)N(C)C(=O)NC(=O)NC(=O)NC(=O)NC(=O)
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCC(=O)N(C)C(=O)NC(=O)NC(=O)NC(=O)NC=O
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: COC(=O)NC(=O)NC(=O)NC(=O)c1ccccc1Cl
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   COC(=O)NC(=O)NC(=O)NC(=O)c1ccccc1Cl
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: COC(=O)NC(=O)NC(=O)NC(=O)c1cccc(Cl)cc1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   COC(=O)NC(=O)NC(=O)NC(=O)c1cccc(Cl)cc1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(C)C(=O)NC(=O)c1ccccc1)c1ccccc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(C)C(=O)NC(=O)c1ccccc1)c1ccccc1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(C)C(=O)NC(=O)c1ccccc1ccccc1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(C)C(=O)NC(=O)c1ccccc1ccccc1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 51):
================================================================================
[1] Oc2c3ccc3c2-#Oc3cccc3c2=O
[2] FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
[3] CCNC(=O)c1ccc(OC)c(C)c1
[4] 2222222222222222222222222222222222222
[5] CCOC(=O)CNC(=O)c1ccc(F)cc1F
[6] NC(=O)NCCNC(=O)c2cc3ccc3oc3ccc3o2
[7] FF)c1ccc(OCC(=O)NCC(N)=O)cc1
[8] ))c2cc3c(cc2c1
[9] 4c1ccc(OCC(=O)NCC(N)=O)cc1
[10] FFF)FFFCC2CCCC2C1
================================================================================


================================================================================
Epoch 51 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0051
  Train - Loss: 1.1974, Recon: 1.1900, KL: 1.4582
  Val   - Loss: 4.8997, Recon: 4.8926, KL: 1.4037
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 52):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CCCCNC(=O)CNC(=O)c1ccc(C)c(C)c1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCCCNC(=O)CNC(=O)c1ccc(C)c(C)c1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)C(=O)NCC(=O)NCCc1ccccc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)C(=O)NCC(=O)NCCc1ccccc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(C)(C)C(=O)NCC(=O)Nc1ccc(F)cc1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(C)(C)C(=O)NCC(=O)Nc1ccc(F)cc1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(C)CC(=O)NCC(=O)Nc1ccc(F)cc1F
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CC(C)CC(=O)NCC(=O)Nc1ccc(F)cc1F
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CCCCCC(=O)NCC(=O)Nc1ccc(C)c(C)c1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CCCCCC(=O)NCC(=O)Nc1ccc(C)c(C)c1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(F)cc1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccc(F)cc1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1ccc(F)cc1F
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1ccc(F)cc1F
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)CC(=O)NCC(=O)Nc1ccc(F)cc1F
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(C)CC(=O)NCC(=O)Nc1ccc(F)cc1F
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCCNC(=O)CCC(=O)Nc1ccc(C)c(C)c1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCCNC(=O)CCC(=O)Nc1ccc(C)c(C)c1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(C)CC(=O)NCC(=O)Nc1ccc(C)c(C)c1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   Cc1ccc(NC(=O)CNC(=O)CC(C)C)cc1C
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 52):
================================================================================
[1] CCC(=O)Nc1ccc(C(C)=O)cc1
[2] )F)(F)FNCc1ccccc1
[3] CC(=O)Nc1ccc(C(=O)NCC(C)C)cc1
[4] Cnn(=O)nn2n1
[5] OCCCC(=O)Nc1ccc(F)cc1
[6] )cccccccccc(C)ccc2Cl)c1)C1CC1
[7] CC(=O)NCC(=O)NC1CCCC1
[8] CCCCNC(=O)CCC(=O)NC1CCCC1
[9] FnCc1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
[10] CCCNC(=O)C1CCCCC1
================================================================================


================================================================================
Epoch 52 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0052
  Train - Loss: 0.7428, Recon: 0.7356, KL: 1.3868
  Val   - Loss: 7.0519, Recon: 7.0439, KL: 1.5490
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 53):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: Ccc((c((((((((((((((((((((((((((((((((((
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   Ccc((c((((((((((((((((((((((((((((((((((
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: C(cccccccccccccccccccccccccccccccccccccc
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   C(cccccccccccccccccccccccccccccccccccccc
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: C1cccccccccccccccccccccccccccccccccccccc
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   C1cccccccccccccccccccccccccccccccccccccc
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: C=c1C)c1cccccccccccccccccccccccccccccccc
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   C=c1C)c1cccccccccccccccccccccccccccccccc
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: Cccccccccccccccccccccccccccccccccccccccc
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   Cccccccccccccccccccccccccccccccccccccccc
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: Cccccccccccccccccccccccccccccccccccccccc
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   Cccccccccccccccccccccccccccccccccccccccc
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cccccccccccccccccccccccccccccccccccccccc
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cccccccccccccccccccccccccccccccccccccccc
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: Cccccccccccccccccccccccccccccccccccccccc
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   Cccccccccccccccccccccccccccccccccccccccc
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 53):
================================================================================
[1] Cc1c1C(c1n1C(C)(c1c1C)n11C1(c1)C1(c1)
[2] (c1c1)1C(c1c1C(C)(C)(C)1C1(c1)C1(c2c1
[3] C1(c1c1C(=O)c1c1)C1(c1cC(C(C)(C)1C1)C
[4] Cc1c1C(c1n1CC1C(c1c1CC1)C1C(=O)11C1C1
[5] 1C1(c2c1)C1(c1c1)C1(c1)C1)C1(c11)C1C1
[6] 1c11C1C1(c1c1CC(C)(C)(C)(C)(C)(C)(C)C
[7] C1C1(C)(C)n11C1(c2c1)C1(c1)C1C1(c1)C1
[8] Cn1n1c(C(c2c(C)n(C)c(C(C)(C)n2C(C)(C)
[9] c1c1C1(c1)C1(c1)C1(c2c1)C1)C1(c1)C1C1
[10] oc2c111)C11C1)c1c1C1(c1)C1(c11)C1C1(c
================================================================================


================================================================================
Epoch 53 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0053
  Train - Loss: 1.8579, Recon: 1.4841, KL: 70.5293
  Val   - Loss: 2.6746, Recon: 2.6098, KL: 12.2256
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 54):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)(C)C(=O)Nc1ccc(C(=O)NCC(C)C)cc1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)CNC(=O)c1ccc(NC(=O)C(C)(C)C)cc1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)N(C)C(=O)C(C)NC(=O)c1ccccc1Cl
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(NC(=O)c1ccccc1Cl)C(=O)N(C)C(C)C
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(C)(C)C(=O)NCC(=O)NCC1CCCC1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(C)(C)C(=O)NCC(=O)NCC1CCCC1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(C)(C)NC(=O)CNC(=O)c1ccc(C(N)=O)cc1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CC(C)(C)NC(=O)CNC(=O)c1ccc(C(N)=O)cc1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(C)NC(=O)C(C)NC(=O)c1ccc(F)cc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(C)NC(=O)C(C)NC(=O)c1ccc(F)cc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CCOC(=O)C(C)NC(=O)c1ccc(C)cc1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCOC(=O)C(C)NC(=O)c1ccc(C)cc1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(C)NC(=O)C(C)NC(=O)c1ccc(F)cc1F
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(C)NC(=O)C(C)NC(=O)c1ccc(F)cc1F
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)(C)NC(=O)CCC(=O)NCC1CCCC1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(C)(C)NC(=O)CCC(=O)NCC1CCCC1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(C)NC(=O)CNC(=O)c1ccc(F)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(C)NC(=O)CNC(=O)c1ccc(F)cc1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(C)NC(=O)C(C)NC(=O)c1ccc(F)cc1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(C)NC(=O)C(C)NC(=O)c1ccc(F)cc1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 54):
================================================================================
[1] 
[2] CC(C)CC(=O)NCC(=O)NCC(=O)NCC1CCCC1
[3] C2CC2)CCN1CC(=O)N(C)C1
[4] 3CCCC2CC2CC2)nn(C)c1=O
[5] c1cccccccccc1CCCC2
[6] CC1CCC(C(=O)NCCCCCCC1)C1CC1
[7] )O)C(C(F)(F)F)C1CC1
[8] 2CC(=O)NCC2
[9] CC(=O)Nc2ccccc2C(F)(F)F)C1CC1
[10] CC(=O)NCC(=O)NCC(=O)NC1CCCCC1
================================================================================


================================================================================
Epoch 54 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0054
  Train - Loss: 2.2079, Recon: 2.1385, KL: 12.8512
  Val   - Loss: 5.8352, Recon: 5.8248, KL: 1.9342
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 55):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)C(=O)NCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCNC(=O)C(C)C
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(=O)NC(=O)NCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCNC(=O)NC(C)=O
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(C)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(C)C
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CCC(=O)NCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCNC(=O)CC
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(=O)NC(=O)NC(=O)c1cccccccc2)c1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(=O)NC(=O)NC(=O)c1cccccccc2)c1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(C)CC(=O)NCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCNC(=O)CC(C)C
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(=O)NC(=O)c1ccccccccc2)cc1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(=O)NC(=O)c1ccccccccc2)cc1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(=O)NC(=O)c1ccccc(C)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(=O)NC(=O)C1=CC=CC=C(C)C=C1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(C(=O)NC(=O)NCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(C(=O)NC(=O)NCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 55):
================================================================================
[1] ####Fc#[nH]nnnn2
[2] 
[3] 3[nH]nnn2c-c2nno2)c4
[4] c(=FF)n2)(=F))#
[5] 2F-c3)n22)c-n22
[6] o42222c2
[7] 266666622)=NnN[nH]2[nH]2nn2c4
[8] (2)(2)(=[nH])=[nH]
[9] 2
[10] =222222222222222222222222222222222222
================================================================================


================================================================================
Epoch 55 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0055
  Train - Loss: 1.3495, Recon: 1.3390, KL: 1.9156
  Val   - Loss: 3.6736, Recon: 3.6623, KL: 2.0665
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 56):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CCC(C)C(=O)Nc1ccc(C(=O)NCCCCC)cc1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCCCCNC(=O)c1ccc(NC(=O)C(C)CC)cc1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CCC(C)(=O)c1ccc(C(=O)NCC(=O)NCCCCCC2)cc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CCC(C)(=O)c1ccc(C(=O)NCC(=O)NCCCCCC2)cc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CCC(C)NC(=O)c1ccc(C(=O)NCCCCC2)cc1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCC(C)NC(=O)c1ccc(C(=O)NCCCCC2)cc1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CCC(C)N(C)C(=O)C(C)CC1CCCC1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCC(C)N(C)C(=O)C(C)CC1CCCC1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CCC(=O)c1ccc(C(=O)NC(C)C)cc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CCC(=O)c1ccc(C(=O)NC(C)C)cc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CCC(=O)c1ccc(C(=O)NCCC(C)(C)C)cc1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCC(=O)c1ccc(C(=O)NCCC(C)(C)C)cc1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CCC(C)N(C)C(=O)NCCCCCCCCC1CCCC1)CC1CCCC1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CCC(C)N(C)C(=O)NCCCCCCCCC1CCCC1)CC1CCCC1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C(C)c1ccc(C(=O)NCCCCC2)cc1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(C(C)c1ccc(C(=O)NCCCCC2)cc1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCC(=O)c1ccc(C(=O)NCCCCC2CCCC2)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCC(=O)c1ccc(C(=O)NCCCCC2CCCC2)cc1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCCCC(C)N(C)C(=O)c1ccc(C(C)C)cc1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCCCC(C)N(C)C(=O)c1ccc(C(C)C)cc1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 56):
================================================================================
[1] OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
[2] CCC(=O)NCC1CCCC1
[3] O=O)c(NC(=O)c1ccccc1F)c1ccccc1
[4] CC(=O)NC(C)CC(=O)NC(C)C
[5] Cc1ccccc1NC(=O)c1ccccc1
[6] C1CCCC1CC1CCCC1
[7] 2c(c2ccccc2)c1C
[8] CC(C)C(=O)NC(C)(C)C)C(=O)N(C)C
[9] CCCCN(CC(=O)NC(=O)CC1CCCC1)C(C)C
[10] cc32)c(F)ccc12
================================================================================


================================================================================
Epoch 56 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0056
  Train - Loss: 1.2126, Recon: 1.2032, KL: 1.6756
  Val   - Loss: 5.3063, Recon: 5.2978, KL: 1.5249
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 57):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)C(=O)Nc1ccc(C)c2cccc1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)C(=O)Nc1ccc(C)c2cccc1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(=O)Nc1ccc(C(=O)NCCCCCCCCCCCCCCCCCCCC2
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(=O)Nc1ccc(C(=O)NCCCCCCCCCCCCCCCCCCCC2
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(C)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(C)C
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(C)C(=O)NC(=O)CCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCC(=O)NC(=O)C(C)C
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(C)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(C)C
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(C)C(=O)NC(=O)NC(=O)NCCCCC1CCCCCCCCCCC
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(C)C(=O)NC(=O)NC(=O)NCCCCC1CCCCCCCCCCC
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(C)C(=O)NC(=O)CCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCC(=O)NC(=O)C(C)C
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)C(=O)C(=O)C(=O)C(=O)CCC1CCCCCCCCCCC
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(C)C(=O)C(=O)C(=O)C(=O)CCC1CCCCCCCCCCC
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(C)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(C)C
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(CCCCC(=O)CCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(CCCCC(=O)CCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 57):
================================================================================
[1] c333333333333333333333333333333333333
[2] 3c3oc3[nH]nc3s4conoc2o3
[3] 1Oc1[nH]nc2c3c3c3c3sc3sc3c=cn[nH]c2ooo1
[4] =3
[5] s1NC1N1ccccc2=O
[6] sssss2sc3ccccc3ooc4o12c4ooc3c=c=oo3c1
[7] oo6666666666o666o666o666o666o66o66o66
[8] [nH]c3nnc3c=[nH]nc3oooo1
[9] =O)c(sn1F)cc1=O
[10] 3O)c1ccc(Br)cs1
================================================================================


================================================================================
Epoch 57 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0057
  Train - Loss: 1.4575, Recon: 1.4481, KL: 1.6428
  Val   - Loss: 4.3196, Recon: 4.3092, KL: 1.8273
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 58):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)C(=O)NCC(=O)NCCc1ccccc1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)C(=O)NCC(=O)NCCc1ccccc1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)(C)C(=O)NCC(=O)NCC(C)C)c1ccccc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)(C)C(=O)NCC(=O)NCC(C)C)c1ccccc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(C)C(=O)NCC(=O)NCc1ccccc1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(C)C(=O)NCC(=O)NCc1ccccc1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(C)C(C)NC(=O)C1CCCCC1)N1CCCCC1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CC(C)C(C)NC(=O)C1CCCCC1)N1CCCCC1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(C)C(C)NC(=O)CNC(=O)C1CCC1)c1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(C)C(C)NC(=O)CNC(=O)C1CCC1)c1ccccc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CCC(C)(C)NC(=O)C(C)C(=O)NCC(C)C
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCC(C)(C)NC(=O)C(C)C(=O)NCC(C)C
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(C)CC(C)NC(=O)C1CCCC1)c1ccc(F)cc1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(C)CC(C)NC(=O)C1CCCC1)c1ccc(F)cc1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)CNC(=O)CCC(=O)Nc1ccc(C)cc1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cc1ccc(NC(=O)CCC(=O)NCC(C)C)cc1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(C)C(=O)NCC(=O)NCc1ccccc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(C)C(=O)NCC(=O)NCc1ccccc1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(C)C(=O)NCC(=O)NCc1ccccc1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(C)C(=O)NCC(=O)NCc1ccccc1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 58):
================================================================================
[1] 5555555555555555555555555555555555555
[2] C(=O)c1ccccc1OC
[3] O)c1ccccc1F)c1ccccc1F
[4] (=O)c1ccc(Cl)cc1Cl
[5] 11CCN1CC(=O)NCC1
[6] CC(C)(C)C(=O)NCC(=O)NCc1ccccc1
[7] FN1CCCCCC1
[8] C1
[9] CCCC1)c1ccc(F)cc1
[10] CC(=O)NCC(=O)NCC(F)(F)F)ccc1F
================================================================================


================================================================================
Epoch 58 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0058
  Train - Loss: 1.1929, Recon: 1.1828, KL: 1.7511
  Val   - Loss: 5.7837, Recon: 5.7735, KL: 1.7653
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 59):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)c1ccccc1NC(=O)c1ccccc1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)c1ccccc1NC(=O)c1ccccc1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)c1ccc(C(=O)NCC(=O)NC2CC2)cc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)c1ccc(C(=O)NCC(=O)NC2CC2)cc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(=O)c1ccc(NC(=O)CC2CC2)cc1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(=O)c1ccc(NC(=O)CC2CC2)cc1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(C)c1ccc(NC(=O)c2ccccc2)cc1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CC(C)c1ccc(NC(=O)c2ccccc2)cc1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(C)C(=O)NC(=O)CC1CCCC1)C(=O)NCC(C)C
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(C)C(=O)NC(=O)CC1CCCC1)C(=O)NCC(C)C
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(CC(=O)CC1CCCCCC1)C(=O)Nc1ccccc1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(CC(=O)CC1CCCCCC1)C(=O)Nc1ccccc1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CCC(CC(=O)Cc1ccccc1NC(=O)c1ccccc1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CCC(CC(=O)Cc1ccccc1NC(=O)c1ccccc1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(CCOc1ccc(C(=O)NCC(C)C)cc1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(CCOc1ccc(C(=O)NCC(C)C)cc1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(C(=O)c1ccc(F)cc1NC(=O)c1ccccc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(C(=O)c1ccc(F)cc1NC(=O)c1ccccc1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(C)C(=O)NC(=O)NC(C)C(=O)Nc1ccc(C(N)=O)
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(C)C(=O)NC(=O)NC(C)C(=O)Nc1ccc(C(N)=O)
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 59):
================================================================================
[1] )c1nnc(-c2ccccc2)c1C#N
[2] CC(C)NC(=O)NCC(=O)NCC(C)C)C(C)C)C(C)C
[3] CC(C)C(=O)NCC(C)(C)C(C)C)c1ccccc1
[4] C
[5] CCCNC(=O)c1ccccc1
[6] COCCCCC#N)NCC1CCCC1
[7] c1ccc(-c2ccc(-c3ccccc3)n2)cc1
[8] n1ccc(-c2ccccc2)cc1
[9] CCNC(=O)CCC(=O)NCC1
[10] NNC(NC(=O)CC)CC1
================================================================================


================================================================================
Epoch 59 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0059
  Train - Loss: 1.3252, Recon: 1.3138, KL: 1.9299
  Val   - Loss: 5.5852, Recon: 5.5781, KL: 1.1976
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 60):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(=O)NC(=O)c1cccc(C)cc1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(=O)NC(=O)c1cccc(C)cc1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)c1ccc(C(=O)NCC2)c1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)c1ccc(C(=O)NCC2)c1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(C)c1ccc(C)cc1)c1ccccc1)c1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(C)c1ccc(C)cc1)c1ccccc1)c1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(=O)NCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCNC(C)=O
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(C)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(C)C
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(=O)NCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCNC(C)=O
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(C)c1ccc(C)cc1)c1ccccc1)c1ccccc1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(C)c1ccc(C)cc1)c1ccccc1)c1ccccc1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(=O)NCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCNC(C)=O
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(=O)NCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCNC(C)=O
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(C)c1ccc(C(=O)NCCCCC2)c1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(C)c1ccc(C(=O)NCCCCC2)c1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 60):
================================================================================
[1] c1cccc1C(=O)NCCCCCCCCC1
[2] (C(N(C(N(C(N(C(N(C(N(C(N(C(N(C(N(C(N(
[3] CCOC(C)C(C)C)CCOCCCCCC1
[4] cc1ccccc1C(=O)NCCCCC1
[5] CCCCCCC(C(C(C(C(C((O)(C(C(C(C(C(C(C(C
[6] ccc1cccc(NC(=O)CCCCCCCCCCCCCCCCCCCCCC
[7] 
[8] Nc1ccccc1)c1ccc(C(N)C(C)C)cc1
[9] ccccc1C(=O)NCCCCCCCCCCCCCCC1
[10] =O
================================================================================


================================================================================
Epoch 60 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0060
  Train - Loss: 1.3478, Recon: 1.3373, KL: 1.7589
  Val   - Loss: 3.5238, Recon: 3.5131, KL: 1.7708
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 61):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: COC(=C(C(=O)N(C(=O)NC(=O)NC(=O)NC(C)C(=O
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   COC(=C(C(=O)N(C(=O)NC(=O)NC(=O)NC(C)C(=O
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: COCC(C(=O)c1cccc(NCCCC)cc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   COCC(C(=O)c1cccc(NCCCC)cc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CCc1cccccccccccccccccccccccccccccccccccc
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCc1cccccccccccccccccccccccccccccccccccc
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CCCCCC(C(=O)CCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCCCCC(C(=O)CCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CCC(C(=C(C(=O)N(C)C(=O)NC(C(=O)NC(C(=O)N
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CCC(C(=C(C(=O)N(C)C(=O)NC(C(=O)NC(C(=O)N
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: COC(=C(C(=O)N(C)C(=O)C(C)C(=O)C(C(=O)N(C
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   COC(=C(C(=O)N(C)C(=O)C(C)C(=O)C(C(=O)N(C
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CCc1cccccccccccccccccccccccccccccccccccc
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CCc1cccccccccccccccccccccccccccccccccccc
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: COC(=C(=C)C(=O)C(C)C(=O)N(C(C(=O)C(=O)c1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   COC(=C(=C)C(=O)C(C)C(=O)N(C(C(=O)C(=O)c1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: COC(C(=O)NC(=O)N(C)c1ccc(F)c2CCCC1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   COC(C(=O)NC(=O)N(C)c1ccc(F)c2CCCC1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCCCCCCCCCCC(=O)c1ccc(C(=O)Nc2ccc(C)c2c
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCCCCCCCCCCC(=O)c1ccc(C(=O)Nc2ccc(C)c2c
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 61):
================================================================================
[1] [nH][nH][nH]snnnnnnnnnnnnnnnnnnnnnnnnsnnnnsnnn
[2] OOOOOOOOOOOOOOOOOOOO=OOOOOOOOOOOOOOOO
[3] (=F2)c1ccccc1F
[4] C=CCNC(=O)c1ccc(F)cc1
[5] c1ccc(-n2cccn2)cc1
[6] C=CCN1CCNC(=O)N1CCOCC1
[7] C2
[8] n2cnnn2)nn1
[9] NNO=O)Clcccc1-c1ccccc1F
[10] C==O)n2ccc1C(=O)N1CCCC1
================================================================================


================================================================================
Epoch 61 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0061
  Train - Loss: 1.0164, Recon: 1.0080, KL: 1.3711
  Val   - Loss: 4.8276, Recon: 4.8145, KL: 2.1418
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 62):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)C(=O)NCC(=O)Nc1cccc(C(=O)NCC2CC2)c1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)C(=O)NCC(=O)Nc1cccc(C(=O)NCC2CC2)c1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)C(=O)NCC(=O)NCC(=O)Nc1ccc(C(N)=O)cc
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)C(=O)NCC(=O)NCC(=O)Nc1ccc(C(N)=O)cc
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(=O)NCC(=O)c1ccc(C(=O)NCC(C)C)cc1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(=O)NCC(=O)c1ccc(C(=O)NCC(C)C)cc1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(C)CNC(=O)CN(C)C(=O)c1ccc(F)cc1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CC(C)CNC(=O)CN(C)C(=O)c1ccc(F)cc1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(C)NC(=O)CCN(C)C(=O)c1ccc(F)cc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(C)NC(=O)CCN(C)C(=O)c1ccc(F)cc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(C)NC(=O)CCC(=O)Nc1ccc(C#N)cc1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(C)NC(=O)CCC(=O)Nc1ccc(C#N)cc1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(C)CNC(=O)C1CCCCCCCCCCCCCCCCCCCC1)C(=O
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(C)CNC(=O)C1CCCCCCCCCCCCCCCCCCCC1)C(=O
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)C(=O)Nc1ccc(C(=O)NCC(=O)NCC2CC2)cc1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(C)C(=O)Nc1ccc(C(=O)NCC(=O)NCC2CC2)cc1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCC(=O)Nc1ccc(C(=
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCC(=O)Nc1ccc(C(=
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCCCCCCNC(=O)CCC(=O)NCC(=O)Nc1ccc(C)cc1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCCCCCCNC(=O)CCC(=O)NCC(=O)Nc1ccc(C)cc1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 62):
================================================================================
[1] (C(N4CC55C5C5C55C55C55C55C55C55C55C55
[2] C(=O)NC)c1ccccc1
[3] CCNC(=O)CC(C)C(C)(C)C
[4] c(NC(=O)C3CC3)ccc1C
[5] cc1ccc(C(=O)NCC(=O)NCC(C)C)cc1
[6] CC(C)C(=O)NCC(=O)NCc1ccccc1
[7] O)CNC(=O)CCC2)cc1
[8] c1)c1ccccc1N
[9] CCC(C)C(=O)NCC(=O)NC1CCCCC1
[10] Cc2ccc(C(=O)OC)cc2C(=O)N2CC2CC2)c1=O
================================================================================


================================================================================
Epoch 62 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0062
  Train - Loss: 1.0525, Recon: 1.0429, KL: 1.5471
  Val   - Loss: 6.2904, Recon: 6.2829, KL: 1.2030
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 63):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(=O)NC(=O)NC(=O)NC(=O)NC(=O)NC(=O)NC(=
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(=O)NC(=O)NC(=O)NC(=O)NC(=O)NC(=O)NC(=
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(=O)NC(=O)c1cccc(C)cc1CCCCCCCCCCCCCCCC
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(=O)NC(=O)c1cccc(C)cc1CCCCCCCCCCCCCCCC
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(=O)c1ccc(C)cc1ccccc(C)cc1ccccc(C)cc1c
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(=O)c1ccc(C)cc1ccccc(C)cc1ccccc(C)cc1c
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(=O)NC(=O)NC(=O)NC(=O)NC(=O)NC(=O)CCCC
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCCCC(=O)NC(=O)NC(=O)NC(=O)NC(=O)NC(C)=O
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(=O)NC(=O)NC(=O)NC(=O)NC(=O)NC(=O)N1CC
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(=O)NC(=O)NC(=O)NC(=O)NC(=O)NC(=O)N1CC
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(=O)c1ccc(C)cc1ccccc(C)cc(C)cc(C)cc1cc
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(=O)c1ccc(C)cc1ccccc(C)cc(C)cc(C)cc1cc
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(=O)c1ccc(C)cc1CCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCc1cc(C)ccc1C(C)=O
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(=O)NC(=O)NC(=O)NC(=O)NC(=O)NC(=O)N1CC
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(=O)NC(=O)NC(=O)NC(=O)NC(=O)NC(=O)N1CC
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(=O)NC(=O)c1cccc(C)cc1CCCCCCCCCCCCCCCC
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(=O)NC(=O)c1cccc(C)cc1CCCCCCCCCCCCCCCC
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(=O)NC(=O)NC(=O)NC(=O)NC(=O)NC(=O)NC(=
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(=O)NC(=O)NC(=O)NC(=O)NC(=O)NC(=O)NC(=
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 63):
================================================================================
[1] NNNNNNNFNNFNNNNNFNNFNNFNNFNNNNNNNNNNN
[2] F)N==F)=F)F)F)F)F)F)F)F)F)F)F)F)F)F)F
[3] sN
[4] s=sssssssssssssssssssssssssssssssssss
[5] N)N=F=========c
[6] ssn
[7] ==)F)F)F)F)F)F)F)F)F)F)F)F)F)F)F)F)F)
[8] #
[9] N#==)F=F=============================
[10] N#=F)================================
================================================================================


================================================================================
Epoch 63 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0063
  Train - Loss: 1.3848, Recon: 1.3732, KL: 1.8329
  Val   - Loss: 4.9250, Recon: 4.9117, KL: 2.1194
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 64):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CCc1ccc(C(=O)NC(=O)c2ccccc1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCc1ccc(C(=O)NC(=O)c2ccccc1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CCc1ccccc1C(=O)NC(=O)NC(=O)NC(=O)NC(=O)N
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CCc1ccccc1C(=O)NC(=O)NC(=O)NC(=O)NC(N)=O
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CCC(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCC(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: C=ccccc1c1cccc1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   C=ccccc1c1cccc1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: Cc1cc1c1cccc1CC(=O)NC(=O)NCCCC
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   Cc1cc1c1cccc1CC(=O)NC(=O)NCCCC
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CCc1cc1c(c1c1c11111111111111111111111111
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCc1cc1c(c1c1c11111111111111111111111111
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CCc1ccccc1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CCc1ccccc1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC11c1c1cccc1c1cccc1cc1C(=O)NCCCCCCCCCC1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC11c1c1cccc1c1cccc1cc1C(=O)NCCCCCCCCCC1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCC(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCC(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCCC(CCCc1ccccc1c1ccccc1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCCC(CCCc1ccccc1c1ccccc1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 64):
================================================================================
[1] nn4N444555555555555555555555555555545
[2] s1ccccc1CCNC(=O
[3] (o1(O((((((((((((((((((((((((((((((((
[4] =N((OO(O11)CCC3
[5] Cc1ccccc1
[6] 1o[nH]ns1)N
[7] 2444444444444444444444444444444444444
[8] n1ncccc1Br
[9] NS(S((((((((((((((((((((((((((((((((
[10] BrS(Sc1nccs1)N1
================================================================================


================================================================================
Epoch 64 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0064
  Train - Loss: 1.2250, Recon: 1.2129, KL: 1.8910
  Val   - Loss: 3.6191, Recon: 3.6062, KL: 2.0180
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 65):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CCC(====================================
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCC(====================================
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CCC(Cc1ccc(C)c1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CCC(Cc1ccc(C)c1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CCCCCCc1cccccccccccccccccccccccccccccccc
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCCCCCc1cccccccccccccccccccccccccccccccc
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CCC(=Occ(=O)C)C)CCCCCCC(C)C)CCCC)COC)C)C
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCC(=Occ(=O)C)C)CCCCCCC(C)C)CCCC)COC)C)C
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CCCC(==C(=C(=C(=C(=C(=C(=C(=C(=C(=C(=C(=
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CCCC(==C(=C(=C(=C(=C(=C(=C(=C(=C(=C(=C(=
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CCcccccccccccccccccccccccccccccccccccccc
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCcccccccccccccccccccccccccccccccccccccc
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CCCcc1cc(C)c1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CCCcc1cc(C)c1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CCCCc1ccc(C)cc1ccc1cccc2cccccccccccccccc
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CCCCc1ccc(C)cc1ccc1cccc2cccccccccccccccc
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: C=Cccccccccccccccccccccccccccccccccccccc
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   C=Cccccccccccccccccccccccccccccccccccccc
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCcc1ccccccccccc(C)c1cc(C)c1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCcc1ccccccccccc(C)c1cc(C)c1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 65):
================================================================================
[1] Fscsssssssssssssss1
[2] 1
[3] C))n2cc3
[4] =cccc3)
[5] )))))))))))))))))))))))))))))))))))))
[6] F
[7] F)c2csc3s4)o3)on3)sc3nc4s3)oc3s3)sc3s
[8] F))c2no3)no2)n1
[9] )ss24)nn3=O
[10] =))c2csc3s3
================================================================================


================================================================================
Epoch 65 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0065
  Train - Loss: 1.4217, Recon: 1.4100, KL: 1.8086
  Val   - Loss: 3.7850, Recon: 3.7739, KL: 1.7094
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 66):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CCC(=O)CCCN(C)C(=O)c1ccc(C)cc1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCC(=O)CCCN(C)C(=O)c1ccc(C)cc1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CCC(=O)NC(=O)NCC(=O)NCc1ccccc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CCC(=O)NC(=O)NCC(=O)NCc1ccccc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(=O)NCCNC(=O)c1ccc(C(F)(F)F)cc1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(=O)NCCNC(=O)c1ccc(C(F)(F)F)cc1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(C)C(C)NC(=O)NCC(=O)NCC(C)C)c1ccccc1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CC(C)C(C)NC(=O)NCC(=O)NCC(C)C)c1ccccc1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CCC(C)C(=O)NCC(=O)NCC(C)C)c1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CCC(C)C(=O)NCC(=O)NCC(C)C)c1ccccc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CCC(=O)N1CCCCC1C(=O)Nc1ccc(C)cc1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCC(=O)N1CCCCC1C(=O)Nc1ccc(C)cc1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: COC(=O)Nc1ccc(C(=O)NCCOC)cc1C
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   COCCNC(=O)c1ccc(NC(=O)OC)c(C)c1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CCC(=O)NCCC(=O)NCC1CCCCC1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CCC(=O)NCCC(=O)NCC1CCCCC1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(=O)NCCNC(=O)c1ccc(C(N)=O)cc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(=O)NCCNC(=O)c1ccc(C(N)=O)cc1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(=O)NCCCC(=O)Nc1ccc(C)cc1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(=O)NCCCC(=O)Nc1ccc(C)cc1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 66):
================================================================================
[1] COc1ccc(C(=O)NCC(=O)N2CCCC2)cc1
[2] COc1ccc(N)cc1C(=O)NCC1CCCC1
[3] ccc(C)cc1NC(=O)c1ccccc1
[4] F)(=O)Nc1ccccc1NCc1ccccc1
[5] Oc1ccc(C(=O)N2CCCC2)cc1
[6] O1)c1CCCC2
[7] N=O)(=O)CCNC(=O)c1ccccc1C
[8] Oc1cccc1)c1ccccc1
[9] CC1CCCCCCC1)NC(=O)c1ccccc1
[10] COC(=O)CNC(=O)c2ccccc21
================================================================================


================================================================================
Epoch 66 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0066
  Train - Loss: 1.1403, Recon: 1.1305, KL: 1.4774
  Val   - Loss: 5.4687, Recon: 5.4587, KL: 1.5252
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 67):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)C(=O)N1CC(=O)NCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)C(=O)N1CC(=O)NCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(C)C(=O)c1ccccc1CCCC1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(C)C(=O)c1ccccc1CCCC1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(C)C(=O)C(=O)(=O)C(=O)(=O)C1CCCCCCCCCC
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(C)C(=O)C(=O)(=O)C(=O)(=O)C1CCCCCCCCCC
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: OC(C)C(=O)NC(=O)c1cccccc1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   OC(C)C(=O)NC(=O)c1cccccc1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(C(=O)NC(=O)C(=O)NC(=O)CCC(=O)C(=O)CC(
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(C(=O)NC(=O)C(=O)NC(=O)CCC(=O)C(=O)CC(
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(C)C(=O)c1ccccc1CCCC1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(C)C(=O)c1ccccc1CCCC1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(=O)c1cccc(C(=O)NC(=O)CCCCCCCCCCCCCCCC
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(=O)c1cccc(C(=O)NC(=O)CCCCCCCCCCCCCCCC
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C)C(=O)N1CCCC(=O)CCCCCCCCCCCCCCCCCCCC
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(C)C(=O)N1CCCC(=O)CCCCCCCCCCCCCCCCCCCC
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(C)C(=O)N1CCC1CC1CC1CC1CCCC1)C1CC1CC1C
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(C)C(=O)N1CCC1CC1CC1CC1CCCC1)C1CC1CC1C
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(C)C(=O)NC(=O)C(=O)C(=O)C(=O)C(=O)C(=O
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(C)C(=O)NC(=O)C(=O)C(=O)C(=O)C(=O)C(=O
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 67):
================================================================================
[1] (N(-)c2ccccc212
[2] C22==========================O2
[3] O)c1CC12
[4] (N(C)S(=O)(=O)(=O)N2CC3
[5] CC(=O)N1CCCC1C(=O)N1CCCC1
[6] (([nH][nH]
[7] C2C(C)CCC1C1CCC1)1
[8] C2=O2CC22CC21
[9] CCNc1cccc1SC1
[10] c1cccc1NC(=O)N1CC1
================================================================================


================================================================================
Epoch 67 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0067
  Train - Loss: 1.4841, Recon: 1.4714, KL: 1.9044
  Val   - Loss: 4.0328, Recon: 4.0215, KL: 1.6784
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 68):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(=O)c1c1cccccccccccccccccccccccccccccc
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(=O)c1c1cccccccccccccccccccccccccccccc
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CO=O)O)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CO=O)O)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CCCCC1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCCCC1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: Cc1ccccccccccccccccccccccccccccccccccccc
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   Cc1ccccccccccccccccccccccccccccccccccccc
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: COCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCOC
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: COCO)C)C)C)C)C)C)C)C)C)C)C)C)C)C)C)C)C)C
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   COCO)C)C)C)C)C)C)C)C)C)C)C)C)C)C)C)C)C)C
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CO)c1ccccccccccccccccccccccccccccccccccc
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CO)c1ccccccccccccccccccccccccccccccccccc
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 68):
================================================================================
[1] )-
[2] F)F
[3] 
[4] )
[5] -SSS-SS(C)SS(C)SS(-S(
[6] --
[7] CS-C
[8] C)SC(-
[9] 
[10] SS-c2[nH])[nH][nH]
================================================================================


================================================================================
Epoch 68 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0068
  Train - Loss: 1.3360, Recon: 1.3221, KL: 2.0458
  Val   - Loss: 3.1008, Recon: 3.0855, KL: 2.2488
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 69):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: 1ccc(C(=O)NCC(=O)NCCC(=O)NCCCCCCC2CC2)cc
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   1ccc(C(=O)NCC(=O)NCCC(=O)NCCCCCCC2CC2)cc
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CCCC(C)C1CCCCCCCC1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CCCC(C)C1CCCCCCCC1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(C)C(=O)NCC(=O)NCC(=O)NCCC(=O)NCCCC(=O
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(C)C(=O)NCC(=O)NCC(=O)NCCC(=O)NCCCC(=O
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(C)C(=O)NCCCCCC1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CC(C)C(=O)NCCCCCC1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CCCC(=O)NCCCCCCCCC1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CCCC(=O)NCCCCCCCCC1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(C)C(=O)NCC(=O)NCC(=O)NCCCC(=O)NCCCC1C
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(C)C(=O)NCC(=O)NCC(=O)NCCCC(=O)NCCCC1C
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(C)C(=O)NCC(=O)NCCCCCC1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(C)C(=O)NCC(=O)NCCCCCC1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(=O)NCC(=O)CCC(=O)NCC1CCCC1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(=O)NCC(=O)CCC(=O)NCC1CCCC1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC1CC(C)C(=O)NCC1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC1CCNC(=O)C(C)C1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCCNC(=O)CCC(=O)NCCCCCCCC1CCCCCCCCCCCCC
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCCNC(=O)CCC(=O)NCCCCCCCC1CCCCCCCCCCCCC
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 69):
================================================================================
[1] CC(=O)NCCC(=O)NCC(=O)CCC(=O)NCCCC1
[2] Cc1ccc(C(=O)NC(=O)NCC(=O)NC(C)C)cc1
[3] (=O)(C(=O)CCCCCCCCCCCCCCCCCCCCCCCCCCC
[4] COC(=O)NCC(=O)NCCCCC(=O)NC(=O)CCC(=O)
[5] CCCCCCC(=O)NCC(C)C(C)C(=O)CCC1
[6] CCCCCCCC2CCCCCCCCCCCCCCCCCCCCCCCCCCCC
[7] CCCCCCCCCCCCCCC3)C(=O)NCC1CCCCCC2)c1c
[8] c(NC(=O)CCC(=O)NCCCCCC3CCCC2)c1
[9] CC(=O)NCCC(=O)NCCCC1CCCCCCCCCCCCCCCCC
[10] ==O)c1cccc1
================================================================================


================================================================================
Epoch 69 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0069
  Train - Loss: 1.4239, Recon: 1.4103, KL: 1.9751
  Val   - Loss: 4.4533, Recon: 4.4456, KL: 1.1120
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 70):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: Cc1ccc(CCCC(=O)CCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   Cc1ccc(CCCC(=O)CCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: Cc1ccc(C(=O)CCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   Cc1ccc(C(=O)CCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: C1ccccccccc1ccc1ccc1ccc1ccc1ccc1ccc1ccc1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   C1ccccccccc1ccc1ccc1ccc1ccc1ccc1ccc1ccc1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: C1cccc(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   C1cccc(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: C1cc(CCC(=O)CCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   C1cc(CCC(=O)CCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: Cccccccccccccccccccccccccccccccccccccccc
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   Cccccccccccccccccccccccccccccccccccccccc
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: Ccccccccccccccccccccc1cc(C(=O)C(=O)NCCCC
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   Ccccccccccccccccccccc1cc(C(=O)C(=O)NCCCC
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cc1cccccc(CCC(C)C(=O)NC(=O)NCCCCCCCCCCCC
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cc1cccccc(CCC(C)C(=O)NC(=O)NCCCCCCCCCCCC
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: C1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   C1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: Cccccccccccccccccccccccccccccccccccccccc
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   Cccccccccccccccccccccccccccccccccccccccc
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 70):
================================================================================
[1] Sc2cccccccccccccccccccccccccccccccccc
[2] Occ2(N)S))c1ccccc1
[3] CC(=O)N1CCC(=O)NC(=O)C1CCCCC1CC1ccccc
[4] CS(=O)CCC(=O)CCCCCCCCCCCCCCCCCCCCCCCC
[5] 
[6] SC(=O)CC(=O)NC(=O)CCC(=O)CCCCCCCCCCCC
[7] SC(=O)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
[8] C(S)c1ccccc1C)cc1C)c1ccccc1Cl
[9] Sc1cccccc1CCCCCCCCCCCCCCCCCCCCCCCCCCC
[10] SC()O)N)c1ccccc1
================================================================================


================================================================================
Epoch 70 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0070
  Train - Loss: 1.7313, Recon: 1.7186, KL: 1.8227
  Val   - Loss: 4.3464, Recon: 4.3347, KL: 1.6723
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 71):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CCC(C(C(C)c1cccccccccccccccccccccccccccc
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCC(C(C(C)c1cccccccccccccccccccccccccccc
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CCC(C(=)c1cccccccccccccccccccccccccccccc
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CCC(C(=)c1cccccccccccccccccccccccccccccc
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CCC(C(C)c1)c1ccccccccccccccccccccccccccc
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCC(C(C)c1)c1ccccccccccccccccccccccccccc
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CCC(C(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCC(C(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CCC(C)C(=O)C(=O)C(=O)C(=O)C(=O)C(=O)C(=O
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CCC(C)C(=O)C(=O)C(=O)C(=O)C(=O)C(=O)C(=O
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CCC(=C(=C(=C(=C(=C(=C(=C(=C(=C(=C(=C(=C(
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCC(=C(=C(=C(=C(=C(=C(=C(=C(=C(=C(=C(=C(
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: COCC(C(=C(CC(CCC(CCC(CCCCCCCCC(CC(CCC(CC
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   COCC(C(=C(CC(CCC(CCC(CCCCCCCCC(CC(CCC(CC
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CCCC(=O)C(=O)C(=O)C(=O)CCCC(C)c1cccccccc
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CCCC(=O)C(=O)C(=O)C(=O)CCCC(C)c1cccccccc
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCC(C(=)CC)CC)CC)CC)CC)CC)CC)CC)CC)CC)CC
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCC(C(=)CC)CC)CC)CC)CC)CC)CC)CC)CC)CC)CC
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCc1ccccccccccccccccccccccccccccccccccc
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCc1ccccccccccccccccccccccccccccccccccc
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 71):
================================================================================
[1] O)3sss2ss3ssc3sss3)sss2)ss2)ss2N((O)(
[2] )(F)3
[3] O(F4444444444444444444444444444444444
[4] O[nH]3
[5] 
[6] 
[7] 
[8] CO22
[9] ==33333333
[10] FF33
================================================================================


================================================================================
Epoch 71 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0071
  Train - Loss: 1.4315, Recon: 1.4181, KL: 1.8767
  Val   - Loss: 2.8563, Recon: 2.8396, KL: 2.3471
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 72):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CCCCCCC(C)C(=O)NC(=O)CC1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCCCCCC(C)C(=O)NC(=O)CC1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: Cc1ccc(C)cc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   Cc1ccc(C)cc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CCCC(C)N(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCCC(C)N(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CCCCC(C)c1ccc(C)c(C)cc1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCCCC(C)c1ccc(C)c(C)cc1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CCCCCC(C)C(=O)NC(=O)CCCCCCCCCCCCCCCCCCCC
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCC(=O)NC(=O)C(C)CCCCC
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CCCCC(C)CC(C)CCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCC(C)CC(C)CCCC
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CCCCCCCCCCCCC(C)c1ccc(C)cc(C)cc1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CCCCCCCCCCCCC(C)C1=CC=C(C)C=C(C)C=C1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCCCCCC(C)N(C)CCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCN(C)C(C)CCCCCCC
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCCC(C)c1ccc(C)cc(C)cc1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCCC(C)C1=CC=C(C)C=C(C)C=C1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 72):
================================================================================
[1] 6666666666666666666666666666666666666
[2] -S(-44S44S44S444444444444444444444444
[3] NSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
[4] S4S4S4S4S4-S4S(=O)O=O)CCCCCCCCCCCCCCC
[5] SC(=OOOO)NC(=O)CCC1CCCCCCC1
[6] Nc2ccc(-(=O)c3ccc(NC(=O)N3CCC3)cc2)c1
[7] 6666666666666666666666666666666O66O)O
[8] -SSS4S(-S4S444444444444444444S4-(=O-(
[9] CC(=O)NCC1
[10] 4444444444444444444444444444444444444
================================================================================


================================================================================
Epoch 72 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0072
  Train - Loss: 1.5945, Recon: 1.5780, KL: 2.2996
  Val   - Loss: 4.0962, Recon: 4.0811, KL: 2.1005
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 73):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CCC(=O)N(C)C(=O)NC(=O)c1ccccc1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCC(=O)N(C)C(=O)NC(=O)c1ccccc1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CCCC(=O)c1ccc(C)c(C)cc1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CCCC(=O)c1ccc(C)c(C)cc1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CCCC(=O)c1ccccc1C(=O)N1CCCC1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCCC(=O)c1ccccc1C(=O)N1CCCC1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: COC(=O)C(=O)N(C)c1cccccc1CC1CCCC1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   COC(=O)C(=O)N(C)c1cccccc1CC1CCCC1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(C(=O)C(=O)N1CCC(=O)NC(=O)NC(=O)NCC1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(C(=O)C(=O)N1CCC(=O)NC(=O)NC(=O)NCC1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(=OCC(=O)N(=O)c1ccccc1C(=O)NC(=O)NC(C)
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(=OCC(=O)N(=O)c1ccccc1C(=O)NC(=O)NC(C)
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CCC(=O)c1ccccc1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CCC(=O)c1ccccc1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(=O)C(=O)N(C)C(=O)c1ccccc1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(=O)C(=O)N(C)C(=O)c1ccccc1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCC(=O)C(=O)NC(=O)NC(=O)NC(=O)c1ccccc1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCC(=O)C(=O)NC(=O)NC(=O)NC(=O)c1ccccc1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: COCCNC(=O)c1cccc(NC(=O)NCCCCCCCCCCCCCCCC
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   COCCNC(=O)c1cccc(NC(=O)NCCCCCCCCCCCCCCCC
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 73):
================================================================================
[1] 44)4444444444444444444444444444444444
[2] 4444444444444444444444444444444444444
[3] 6666666666666666666666666666666666666
[4] ))1CC1)C1)C1CC1)C1CC1)c11)c1c11
[5] 444)Nc4ccc4[nH]1
[6] 4444444444444444444444444444444444444
[7] OS(=O)(=O)(=O)(=O)(=O)(=O)(=O)(=O)(=O
[8] 44
[9] (O)(=O)(=O)(F)F)c1ccccc1
[10] N)(=O)(=O)N1CCCC1
================================================================================


================================================================================
Epoch 73 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0073
  Train - Loss: 1.2768, Recon: 1.2653, KL: 1.5723
  Val   - Loss: 4.6483, Recon: 4.6365, KL: 1.6143
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 74):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(=O)c1cccc1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(=O)c1cccc1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(=O)c1cccc1CCCC1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(=O)c1cccc1CCCC1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(=O)N1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CC(=O)N1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(=O)C(=O)NC(=O)c1ccccc1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CC(=O)C(=O)NC(=O)c1ccccc1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(=O)c1ccccc1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(=O)c1ccccc1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(=O)c1ccccc1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(=O)c1ccccc1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(=O)N1CCCCCCCCC(C)CC1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(=O)N1CCCCCCCCC(C)CC1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(C(=O)NC(=O)NC(=O)NC(C)C1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(C(=O)NC(=O)NC(=O)NC(C)C1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(=O)N1CCC(C)CCC1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(=O)N1CCCC(C)CC1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(=O)c1cccc(C(=O)NC(C)CCCCCCCCCCCCCCCCC
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(=O)c1cccc(C(=O)NC(C)CCCCCCCCCCCCCCCCC
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 74):
================================================================================
[1] )N(CCCCCCCCCCCCC(=O)(=O)c2cccccc2c2cc
[2] ccccccccccccccccccccccccccccccccccccc
[3] Fc1cccc(C(C)c2ccccc2c2)ccc1
[4] COC(=O)c1cccc(C(C)NC(C)C)c1
[5] ccc(-c2cccc(NC(C)C)c2)c1
[6] Cc1ccc(C(=O)NC(C)C)c1C
[7] )O)O)c1ccc(-(=O)NC(=O)NC(C)C(C)(C)C)C
[8] CC(C)NC(=O)CCCCCCCCCCCCCCCCCCCCCCCCCC
[9] nc1cccc(-c2ccccc2ccccc2ccccc2ccccc2cc
[10] c1cccc(-c2ccccc2ccccc2c2)ccc1C(C)CCCC
================================================================================


================================================================================
Epoch 74 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0074
  Train - Loss: 1.4388, Recon: 1.4232, KL: 2.1103
  Val   - Loss: 4.6316, Recon: 4.6176, KL: 1.8979
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 75):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(=O)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(C)=O
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CCCCCCCCCCCCCCCCC(=O)CCCCCCCCCCCCCCCCCCC
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCC(=O)CCCCCCCCCCCCCCCC
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(=O)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(C)=O
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(C)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(C)C
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(=O)C(=O)CCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)C(C)=O
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CC(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(=O)C(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(=O)C(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(=O)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(C)=O
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 75):
================================================================================
[1] 5C5
[2] 
[3] F
[4] F5F5F5F5F6FFFF6F6F6F6F6F6F6F6F6F
[5] -------------------------------------
[6] FF5F5F5F5F5F5F5F5F5F5F5F5F5F5F5F5F5F
[7] SSSS)F)F
[8] 
[9] F
[10] 
================================================================================


================================================================================
Epoch 75 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0075
  Train - Loss: 1.7563, Recon: 1.7402, KL: 2.1492
  Val   - Loss: 3.2373, Recon: 3.2176, KL: 2.6201
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 76):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CCCC(=O)cc(C)cc(C(=O)N2CCCCCCCCCCCCCCCCC
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCCC(=O)cc(C)cc(C(=O)N2CCCCCCCCCCCCCCCCC
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CCCC(=O)N(=O)C(=O)CCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CCCC(=O)N(=O)C(=O)CCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CCC(C(=O)c1ccc(C)cc2ccccccccccc1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCC(C(=O)c1ccc(C)cc2ccccccccccc1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CCCC(=O)ccc1C(=O)NC(=O)CC1CCC1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCCC(=O)ccc1C(=O)NC(=O)CC1CCC1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CCCC(C)N(=O)N(=O)C(=O)NC(=O)CCC(=O)NC(=O
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CCCC(C)N(=O)N(=O)C(=O)NC(=O)CCC(=O)NC(=O
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CCCC(C(=O)c1cccc1CCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCCC(C(=O)c1cccc1CCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CCC(=O)c1ccc2C(=O)NC(=O)C1CCCC1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CCC(=O)c1ccc2C(=O)NC(=O)C1CCCC1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CCCC(=O)cc(C(=O)NC(C)CCCCCCCCCCCCCCCCCCC
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CCCC(=O)cc(C(=O)NC(C)CCCCCCCCCCCCCCCCCCC
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCC(=O)c1ccc1C(=O)NC(=O)C1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCC(=O)c1ccc1C(=O)NC(=O)C1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCC(=O)c1CCCCC(=O)NCC(=O)NC(C)CCCCCCCCC
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCC(=O)c1CCCCC(=O)NCC(=O)NC(C)CCCCCCCCC
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 76):
================================================================================
[1] CC(C)C(=O)NC(=O)c2ccc(Cl)cc2)c1
[2] C#CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
[3] Cl
[4] ClSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
[5] CCOCC=OOOOOOOOOOOOOOOOOCOOOCCOCC
[6] #=)c2cc(N)c(N)c3)cc32)c1
[7] 2CCC(=O)NCC1
[8] FFFF)F)c2cc(Cl)c(F)cc2Cl)cc1
[9] 
[10] 2=Cl=ClClClCl=ClClClCl=ClCl=ClCl=ClClCl2=ClClClCl2=ClCl2=ClCl
================================================================================


================================================================================
Epoch 76 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0076
  Train - Loss: 1.6221, Recon: 1.6033, KL: 2.4758
  Val   - Loss: 4.2373, Recon: 4.2195, KL: 2.3359
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 77):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: FCcccccccccccccccccccccccccccccccccccccc
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   FCcccccccccccccccccccccccccccccccccccccc
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: ccccCCCCCCCCCCCC(CCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   ccccCCCCCCCCCCCC(CCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CCcccccccccccccccccccccccccccccccccccccc
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCcccccccccccccccccccccccccccccccccccccc
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: Cccccccccccccccccccccccccccccccccccccccc
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   Cccccccccccccccccccccccccccccccccccccccc
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: OCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCO
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: C(cccccccccccccccccccccccccccccccccccccc
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   C(cccccccccccccccccccccccccccccccccccccc
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cccccccccccccccccccccccccccccccccccccccc
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cccccccccccccccccccccccccccccccccccccccc
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC()cccccccccccccccccccccccccccccccccccc
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC()cccccccccccccccccccccccccccccccccccc
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: 3CCCCCcccccccccccccccccccccccccccccccccc
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   3CCCCCcccccccccccccccccccccccccccccccccc
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 77):
================================================================================
[1] 
[2] C)c2cc(
[3] 44
[4] N))c1nocc1
[5] C
[6] 3
[7] c1ccccc1cc1ccccc1c1cocc1
[8] C
[9] oooo2nooo2
[10] c1cc((
================================================================================


================================================================================
Epoch 77 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0077
  Train - Loss: 1.6657, Recon: 1.6473, KL: 2.4020
  Val   - Loss: 2.6454, Recon: 2.6280, KL: 2.2553
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 78):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: =CCCCCCCCC=CC==CCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   =CCCCCCCCC=CC==CCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: 2CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   2CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: ((CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   ((CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CCc1ccc1cc1c1CCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CCc1ccc1cc1c1CCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: Cccccccccccccccccccccccccccccccccccccccc
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   Cccccccccccccccccccccccccccccccccccccccc
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: Cccccccccccccccccccccccccccccccccccccccc
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   Cccccccccccccccccccccccccccccccccccccccc
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: (Ccccccccccccccccccccccccccccccccccccccc
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   (Ccccccccccccccccccccccccccccccccccccccc
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 78):
================================================================================
[1] ccccccccccccccccccccccccccccccccccccc
[2] Ccccccccccccccccccccccccccccccccccccc
[3] CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
[4] ccccccccccccccccccccccccccccccccccccc
[5] Cc1cccccccccccccccccccccccccccccccccc
[6] cc1cccccccccccccccccccccccccccccccccc
[7] cc1cccccccccccccccccccccccccccccccccc
[8] Ccccccccccccccccccccccccccccccccccccc
[9] C)ccccccccccccccccccccccccccccccccccc
[10] (C)cccccccccccccccccccccccccccccccccc
================================================================================


================================================================================
Epoch 78 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0078
  Train - Loss: 3.7148, Recon: 2.4556, KL: 161.4358
  Val   - Loss: 2.7429, Recon: 2.7024, KL: 5.1899
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 79):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(=O)NCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCNC(C)=O
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(=O)c1cccccccccccccccccccccccccccccccc
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(=O)c1cccccccccccccccccccccccccccccccc
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CC(=O)NC(=O)NC(=O)CCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCC(=O)NC(=O)NC(C)=O
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CCC(=O)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)CC
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CC(C)c1ccccccccccccccccccccccccccccccccc
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CC(C)c1ccccccccccccccccccccccccccccccccc
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CC(=O)c1cccccccccccccccccccccccccccccccc
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CC(=O)c1cccccccccccccccccccccccccccccccc
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CC(=O)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(C)=O
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CC(=O)c1cccccccccccccccccccccccccccccccc
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CC(=O)c1cccccccccccccccccccccccccccccccc
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CC(=O)C(=O)N(=O)NC(=O)NC(=O)c1cccccccccc
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CC(=O)C(=O)N(=O)NC(=O)NC(=O)c1cccccccccc
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(=O)cc(C(=O)cccccccccccc(C(=O)cccccccc
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(=O)cc(C(=O)cccccccccccc(C(=O)cccccccc
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 79):
================================================================================
[1] 4
[2] 
[3] 
[4] =
[5] 5=
[6] 4444444444444444444444444444444444444
[7] ooo
[8] 
[9] 
[10] o
================================================================================


================================================================================
Epoch 79 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0079
  Train - Loss: 2.2950, Recon: 2.0647, KL: 29.1519
  Val   - Loss: 4.7923, Recon: 4.7791, KL: 1.6792
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 80):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: OO)c))))))))))))))))))))))))))))))))))))
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   OO)c))))))))))))))))))))))))))))))))))))
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: 
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: 2)))))OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   2)))))OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: 22CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   22CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: Cccccccccccccccccccccccccccccccccccccccc
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   Cccccccccccccccccccccccccccccccccccccccc
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: O(((2221111111111111111111((((((((((((((
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   O(((2221111111111111111111((((((((((((((
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: 
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: 
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CcCc1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CcCc1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: Cc112111c1111111111111111111111111111111
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   Cc112111c1111111111111111111111111111111
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 80):
================================================================================
[1] c2ccc2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c
[2] (=)(c1ccccc22)c1
[3] C(c1c(c22)(c22)c12
[4] 22
[5] C(=2)(22)(22)(22222)(2222)(2222(c22)(
[6] c2ccc2c22
[7] c2cc(c222
[8] 2c(c2ccccc(c22)c1
[9] c1c222c222c222
[10] C1)(c2c2c2c22)c1
================================================================================


================================================================================
Epoch 80 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0080
  Train - Loss: 1.7463, Recon: 1.7314, KL: 1.8644
  Val   - Loss: 3.8742, Recon: 3.8224, KL: 6.4745
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 81):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CC(C)c1ccccccccccccccccccccccccccccccccc
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CC(C)c1ccccccccccccccccccccccccccccccccc
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CC(=O)c1cccccccccccccccccccccccccccccccc
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CC(=O)c1cccccccccccccccccccccccccccccccc
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CCCC(=O)c1cccccccccccccccccccccccccccccc
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCCC(=O)c1cccccccccccccccccccccccccccccc
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC(CC(C)c1cccccccccccccccccccccccccccccc
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CC(CC(C)c1cccccccccccccccccccccccccccccc
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CCC(=O)N1ccccccccccccccccccccccccccccccc
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CCC(=O)N1ccccccccccccccccccccccccccccccc
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: C(=O)c1ccccccccccccccccccccccccccccccccc
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   C(=O)c1ccccccccccccccccccccccccccccccccc
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CCCCC(=O)c1ccccccccccccccccccccccccccccc
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CCCCC(=O)c1ccccccccccccccccccccccccccccc
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 81):
================================================================================
[1] ccccccccccccccccccccccccccccccccccccc
[2] CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
[3] =O)c1
[4] C#=O)ccc(C)c(C)c(C)c(C)c(C)c(C)c(C)c(
[5] O)cccccc1
[6] ccccccccccccccccccccccccccccccc1ccccc
[7] ==OOccccccccccccccccccccccccccccccccc
[8] ccc1
[9] CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
[10] cc1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
================================================================================


================================================================================
Epoch 81 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0081
  Train - Loss: 3.1247, Recon: 2.3107, KL: 100.4934
  Val   - Loss: 3.8554, Recon: 3.8412, KL: 1.7465
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 82):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CCC(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCC(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CCC(=O)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)CC
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CCCC(=O)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)CCC
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CCC(=O)C(=O)CCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)C(=O)CC
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CCC(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCC(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CCCC(=O)c1cccccccccccccccccccccccccccccc
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CCCC(=O)c1cccccccccccccccccccccccccccccc
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCC(=O)C(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCC(=O)C(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCCCC(CCC(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCCCC(CCC(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 82):
================================================================================
[1] Cl[nH]c
[2] ClCl1
[3] =O)1
[4] 6cCl
[5] Brc1ccClccClCl-c
[6] =OC1
[7] =#==O1cccc1-cc
[8] ClClClClClClClClClClClClClClClClClClClClClClClClClClClClClClClClClClClClCl
[9] ClCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
[10] #Cccccccccccccccccccccccccccccccccccc
================================================================================


================================================================================
Epoch 82 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0082
  Train - Loss: 1.8273, Recon: 1.8120, KL: 1.8724
  Val   - Loss: 3.1108, Recon: 3.0925, KL: 2.2278
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 83):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CCCC(C)C(=O)NCC1CCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCCC(C)C(=O)NCC1CCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CCCC(=O)NC(=O)CC1CCCCC1
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CCCC(=O)NC(=O)CC1CCCCC1
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CCC(C)NC(=O)CC1CCCC1
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCC(C)NC(=O)CC1CCCC1
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CC1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CC1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CCCC(C)C(=O)NCCC1CCCC1
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CCCC(C)C(=O)NCCC1CCCC1
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CCCC(C)C(=O)NCC1CCCCCCCCCCCCCCCCCCCCCCCC
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCCC(C)C(=O)NCC1CCCCCCCCCCCCCCCCCCCCCCCC
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: N#C(C)CC1CCC1
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   N#C(C)CC1CCC1
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: N#C(C)C(=O)NCCC1CCC1
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   N#C(C)C(=O)NCCC1CCC1
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: Nc1ccccc1NC(=O)CC1CCCC1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   Nc1ccccc1NC(=O)CC1CCCC1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CC(C)C(=O)NCC(=O)NCCC1CCC1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CC(C)C(=O)NCC(=O)NCCC1CCC1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 83):
================================================================================
[1] cccc2ccc(C)ccc2ccc(C(F)F)cc2c1ccccc2c
[2] CC1CC1)C(=F)F)F)c1cccccc1C(=O)NCC1CC1
[3] CCc1cccc(C)c1ccc(NC(=O)NC(=O)c2ccccc1
[4] O=O)c2cccc(C(C)(C)C)c2c(C)cc2c(c2)ccc
[5] CC(=O)NCCC(C)C(=O)NCC1
[6] =O)(=O)(=O)(=O)(O)c1cccccc1C(N)C(C)C1
[7] cssss1-1cccccc1Cl
[8] O)c1cccccc1Cl
[9] =C(=O)Oc1cccccccc1C(=O)OCC1C
[10] ccccc1NC(=O)N1CC1CC1CC1CC1CC1CC1CC1CC
================================================================================


================================================================================
Epoch 83 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0083
  Train - Loss: 1.2263, Recon: 1.2123, KL: 1.6821
  Val   - Loss: 5.0858, Recon: 5.0738, KL: 1.4425
  Val Recon Acc:     0.00% (0/51745)


================================================================================
Reconstruction Examples (Epoch 84):
================================================================================

[1] Target:        CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Reconstructed: CCC(=O)NC(=O)C(C)CC(=O)NCCC(C)C)c1ccccc1
    Target (Can):  CC(C)(C)OC(=O)NCCNC(=O)NCC1CC1
    Recon (Can):   CCC(=O)NC(=O)C(C)CC(=O)NCCC(C)C)c1ccccc1
    Match: ✗

[2] Target:        COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Reconstructed: CCCC(=O)NC(=O)CC(=O)NC(=O)CCCCCCCCCCCCCC
    Target (Can):  COC(CNC(=O)Nc1cccs1)c1cccc(F)c1
    Recon (Can):   CCCCCCCCCCCCCCC(=O)NC(=O)CC(=O)NC(=O)CCC
    Match: ✗

[3] Target:        C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Reconstructed: CCC(=O)NC(=O)CCC1CCC(=O)NC(=O)CCCCCC(C)C
    Target (Can):  C=CCSc1nnc(NC2OC(=O)c3ccccc32)s1
    Recon (Can):   CCC(=O)NC(=O)CCC1CCC(=O)NC(=O)CCCCCC(C)C
    Match: ✗

[4] Target:        CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Reconstructed: CCC(=O)Nc1ccc(C(=O)NC(=O)c2ccccc2)c1
    Target (Can):  CNC(=O)c1c(OC2CC2)cccc1C(=O)N(C)C
    Recon (Can):   CCC(=O)Nc1ccc(C(=O)NC(=O)c2ccccc2)c1
    Match: ✗

[5] Target:        NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Reconstructed: CCCC(=O)NC(=O)CC(=O)NCCCCCCCCCCCCCCCCCCC
    Target (Can):  NC(=O)NC(Cc1ccccc1)Cc1ccccc1
    Recon (Can):   CCCCCCCCCCCCCCCCCCCNC(=O)CC(=O)NC(=O)CCC
    Match: ✗

[6] Target:        COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Reconstructed: CCC(C)C(=O)Nc1ccccc1C(=O)NCC1
    Target (Can):  COc1ccc(Br)c(C(=O)N(C)CC(C)C)c1
    Recon (Can):   CCC(C)C(=O)Nc1ccccc1C(=O)NCC1
    Match: ✗

[7] Target:        CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Reconstructed: CCC(=O)NC(=O)C(C)C(=O)NCCCCCCCCCCCCCCCCC
    Target (Can):  CN1CC(O)N(c2nnc(C(F)(F)F)s2)C1=O
    Recon (Can):   CCCCCCCCCCCCCCCCCNC(=O)C(C)C(=O)NC(=O)CC
    Match: ✗

[8] Target:        Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Reconstructed: CCC(=O)NC(=O)CC(C)CC(=O)NCCCC(C)C)c1cccc
    Target (Can):  Cn1cc(CCC(=O)NC(C)(C)CO)c2ccccc21
    Recon (Can):   CCC(=O)NC(=O)CC(C)CC(=O)NCCCC(C)C)c1cccc
    Match: ✗

[9] Target:        CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Reconstructed: CCCC(=O)Nc1ccccc1C(=O)NCC1
    Target (Can):  CCCn1nccc1C(=O)NCCc1ccc(O)cc1
    Recon (Can):   CCCC(=O)Nc1ccccc1C(=O)NCC1
    Match: ✗

[10] Target:        CCCNc1nc(-c2scnc2COC)cs1
    Reconstructed: CCC(C)C(=O)Nc1cccc(C(=O)NCCC2)c1
    Target (Can):  CCCNc1nc(-c2scnc2COC)cs1
    Recon (Can):   CCC(C)C(=O)Nc1cccc(C(=O)NCCC2)c1
    Match: ✗

================================================================================
Sampled Molecules from Latent Space (Epoch 84):
================================================================================
[1] Cc1cccc(N)c1
[2] O)[nH]2)c1
[3] c1ccccc1C1CC1CCCCCCCCCCCCCCCCCCCCCCCC
[4] CCCCC1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC1
[5] ccccccc2
[6] =O)o2ccc(F)c2)c1
[7] CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
[8] OOccccccccccccc2cc3[nH]c(C)nc1
[9] C#O)c1
[10] =O)ccccc1
================================================================================


================================================================================
Epoch 84 Summary:
================================================================================
  Learning Rate:     0.000750
  KL Weight:         0.0084
  Train - Loss: 1.3932, Recon: 1.3757, KL: 2.0770
  Val   - Loss: 5.2464, Recon: 5.2314, KL: 1.7915
  Val Recon Acc:     0.00% (0/51745)

