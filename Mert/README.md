# Multi-View Pig Posture Recognition

Kaggle-Wettbewerb zur Klassifizierung von Schweinehaltungen (5 Klassen) mittels Computer Vision und PyTorch.  
Metrik: **Macro-averaged F1-Score**

---

## Projektstruktur

### Explorative Datenanalyse (EDA)

| Datei | Inhalt |
|---|---|
| [01_Baseline.ipynb](01_Baseline.ipynb) | Erste EDA auf Train1: Klassenverteilung, Bounding-Box-Visualisierung, Seitenverhältnisse, Beispielbilder pro Klasse, Kameraverteilung |
| [01b_EDA_Vertieft.ipynb](01b_EDA_Vertieft.ipynb) | Tiefergehende EDA über alle drei Datensätze (Train1, Train2, Test): Imbalance-Ratio (14,3x), Kamera-Shift (Train 58% cam1 / Test 36% cam1), klassenspezifische Bounding-Box-Analyse, Cross-Dataset-Vergleich |

### Modellierung – Individuelle Experimente

> **Hinweis zur Versionierung:** Dateien mit dem Suffix `b` (z. B. `03b_T1_trainiert.ipynb`) sind überarbeitete Versionen der gleichnamigen Originale (z. B. `03_T1_trainiert.ipynb`). Die b-Versionen enthalten u. a. den Fix der Modell-Speicherlogik von Validation Accuracy auf **Macro F1-Score** sowie weitere Bugfixes. Die Originaldateien wurden zur Nachvollziehbarkeit des Iterationsprozesses beibehalten.

| Datei | Modell | Datensatz | Besonderheiten |
|---|---|---|---|
| [02_first_model.ipynb](02_first_model.ipynb) | ResNet50 | Train1 (500 Samples) | Schneller Pipeline-Test ohne Validierungssplit |
| [03_T1_trainiert.ipynb](03_T1_trainiert.ipynb) | ResNet50 | Train1 (~22k Samples) | Ursprüngliche Version – Modell-Speicherung via Validation Accuracy |
| [03b_T1_trainiert.ipynb](03b_T1_trainiert.ipynb) | ResNet50 | Train1 (~22k Samples) | Überarbeitet: 80/20-Split, BBox-Crop, Augmentation, 2-Phasen-Training, Modell-Speicherung via Macro F1 |
| [04_T2_trainiert.ipynb](04_T2_trainiert.ipynb) | ResNet50 | Train2 (~23k Samples) | Ursprüngliche Version – Modell-Speicherung via Validation Accuracy |
| [04b_T2_trainiert.ipynb](04b_T2_trainiert.ipynb) | ResNet50 | Train2 (~23k Samples) | Überarbeitet: Transfer auf Train2-Domäne, Modell-Speicherung via Macro F1 |
| [05_convnext.ipynb](05_convnext.ipynb) | ConvNeXt-Base | Train1 | StratifiedGroupKFold (kamerabasiert), MixUp/CutMix |
| [06_convnext_hyperparameter.ipynb](06_convnext_hyperparameter.ipynb) | ConvNeXt-Base | Train1 | Hyperparameter-Tuning, WeightedRandomSampler, Cosine-Warmup |
| [07_eva02.ipynb](07_eva02.ipynb) | EVA02-Base (timm) | Train1 | Ursprüngliche Version – Modell-Speicherung via Validation Accuracy |
| [07b_eva02.ipynb](07b_eva02.ipynb) | EVA02-Base (timm) | Train1 | Überarbeitet: 224px, WeightedRandomSampler, 4 Trainingsphasen |
| [08_convnextv2.ipynb](08_convnextv2.ipynb) | ConvNeXt-V2-Huge | Train1+2 | Ursprüngliche Version – Modell-Speicherung via Validation Accuracy |
| [08b_convnextv2.ipynb](08b_convnextv2.ipynb) | ConvNeXt-V2-Huge | Train1+2 | Überarbeitet: 384px, 2x Tesla V100 (DataParallel), Ensemble (cam1/cam2), TTA |


## Klassen

| ID | Bezeichnung |
|---|---|
| 0 | Lateral_lying_left |
| 1 | Lateral_lying_right |
| 2 | Sitting |
| 3 | Standing |
| 4 | Sternal_lying |
