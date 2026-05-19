# Merge – Konsolidierte Ergebnisse

Dieser Ordner enthält die finale Zusammenführung des Teams: die konsolidierte **EDA** sowie das **Best-of-Model** des Projekts (V5-Ensemble).

---

## Dateien

| Datei | Inhalt |
|---|---|
| [merged_01_EDA.ipynb](merged_01_EDA.ipynb) | Konsolidierte Explorative Datenanalyse über alle drei Datensätze (Train1, Train2, Test). Enthält: Klassenverteilung, Imbalance-Analyse (14,3x), Kamera-Distribution-Shift (Train 58% cam1 / Test 36% cam1), cam1 vs. cam2 Beispielbilder pro Klasse, Bounding-Box-Analyse, Zusammenfassung. |
| [v5_train.ipynb](v5_train.ipynb) | **Best-of-Model Training (V5)** – Ensemble aus DINOv2 Large + ConvNeXt V2 Large. Bestes Ergebnis im Leaderboard.  |
| [v5_inference.ipynb](v5_inference.ipynb) | **Best-of-Model Inferenz (V5)** – Erstellt die finale Kaggle-Submission. |
| [v5_inference_compare.ipynb](v5_inference_compare.ipynb) | Vergleichs-Inferenz für V5: Gegenüberstellung verschiedener Inferenz-Varianten (z. B. TTA, Ensemble-Gewichtung).|
| [model_stats.ipynb](model_stats.ipynb) | **Auswertung des finalen V5-Modells:** Strict-CLO-Validierung aus den gespeicherten V5-Checkpoints inkl. Confusion Matrix, Per-Class-Scores (Precision/Recall/F1) und Per-Camera-Breakdowns. Zusätzlich Inspektion der finalen Submission (Klassenverteilung, Train-vs-Test-Vergleich). |

---

## Best-of-Model (V5) – Übersicht

**Ensemble-Ansatz: DINOv2 Large (ViT) + ConvNeXt V2 Large (CNN)** – maximale Architektur-Diversität.

| Komponente | Konfiguration |
|---|---|
| Architektur A | `vit_large_patch14_dinov2.lvd142m` @ **518px** |
| Architektur B | `convnextv2_large.fcmae_ft_in22k_in1k_384` @ **384px** |
| Loss | CrossEntropy + Class Weights (Label Smoothing 0,05) |
| Optimierung | AdamW, LR 1e-4, Cosine Schedule mit 3 Warmup-Epochs |
| Regularisierung | MixUp (α=0,2) + CutMix (α=1,0), EMA (decay 0,9995) |
| Layer-wise LR Decay | 0,75 (nur ViT) |
| Augmentation | RandomResizedCrop(0.7,1.0), Perspective, GaussianBlur, ColorJitter |
| Validation | Strict CLO (nur Test-Kamera-Folds, 3 Folds) |
| Hardware | 4× Tesla V100-SXM2 32GB (128 GB VRAM gesamt) |
| Batch | DINOv2: 16 × Grad-Accum 2 (effektiv 32) / ConvNeXtV2: 24 |
| Epochs | 10 pro Architektur |

### Verbesserungen gegenüber V4
1. V3-Hyperparams zurück: LR=1e-4 (kein lineares Scaling), CE + Class Weights statt Focal Loss
2. DINOv2 **Large @ 518px** statt Base @ 392 – mehr Kapazität für Domain-Shift
3. ConvNeXt **V2 Large @ 384** (FCMAE pretrained) statt ConvNeXt V1 Base – deutlich stärker
4. EMA Weights stabilisieren Vorhersagen
5. MixUp + CutMix via `timm.data.Mixup` – starke Regularisierung
6. Heavier Augmentations (Perspective, GaussianBlur)
7. Layer-wise LR Decay (0,75) für ViT statt einfachem backbone_mult
8. Strict CLO: nur Test-Kamera-Folds (keine Extra-Train-Cams) → bessere Schätzung der Test-Performance
9. Gradient Accumulation (effektiv Batch 32 trotz 16er Mini-Batch)

---
