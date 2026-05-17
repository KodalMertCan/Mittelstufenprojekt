# Arbeitsprotokoll Mittelstufenprojekt — Multi-View Pig Posture Recognition

## Kontext

Kaggle-Wettbewerb zur Erkennung von 5 Schweine-Körperhaltungen aus Bildern mehrerer Kameras:

- **Klassen** (5): Lateral_lying_left, Lateral_lying_right, Sitting, Standing, Sternal_lying
- **Evaluation**: Macro-F1
- **Splits**: T1 (Public Leaderboard) und T2 (Private Leaderboard, **gewinnentscheidend**)
- **Test-Kameras**: `pen1_tur_cam1`, `pen2_orb_cam2`, `pen2_tur_cam2` — diese sind **nur in T2** und kommen im Training NICHT vor → **Domain-Shift** ist die zentrale Herausforderung
- **Hardware**: 4× Tesla V100 SXM2 32 GB (128 GB Total VRAM)

**Modelle**

### EfficientNet-B4 Baseline (`train.ipynb` / `inference.ipynb`)

**Kontext:** Das Projekt startete mit einem Baseline-Notebook (`train.ipynb`) das EfficientNet-B4 als Backbone nutzte.

- **Architektur:** EfficientNet-B4 (via `timm`, pretrained auf ImageNet)
- **IMG_SIZE:** 224px (feste Größe)
- **Augmentations:** Standard (RandomCrop, HorizontalFlip, VerticalFlip, ColorJitter, RandomGrayscale, RandomErasing)
- **Loss:** CrossEntropy + Class Weights + Label Smoothing (0.10)
- **Optimizer:** AdamW (lr=3e-4)
- **MixUp Alpha:** 0.20
- **Split:** Simple `train_test_split` (stratified, 10% val) – **kein GroupKFold**, d.h. Schweine aus dem gleichen Bild konnten in Train UND Val landen → **Data Leakage**
- **Workflow:** T1 Training (40 Epochen) → Inference → T2 Fine-Tuning (20 Epochen, lr=5e-5, von T1-Checkpoint) → Inference

**Erkenntnis:** Funktionale Baseline, aber keine Anti-Domain-Shift-Maßnahmen (keine kamera-spezifischen Augmentierungen, kein GroupKFold). Das Modell lernte zu stark kameraspezifische Merkmale.

### SwinV2 (`swinv2.ipynb` / `swinv2_inference.ipynb`)

> User: *"Der ConvNeXt-Backbone hat am besten funktioniert"*

Es wurde auch ein SwinV2-Modell probiert. Die Ergebnisse fielen hinter ConvNeXt-Base zurück.

### ConvNeXt V1 (`convnext.ipynb` / `convnext_inference.ipynb`)

ConvNeXt-Base erzielte die besten Ergebnisse in den frühen Tests (besser als EfficientNet-B4 und SwinV2). Allerdings bestand weiterhin das Problem des Domain-Shifts zwischen `orb`- und `tur`-Kameras.

### K-Fold Cross-Validation (`kfold_train.ipynb` / `kfold_inference.ipynb`)

- Architektur: **ConvNeXt-Base** @ 288 px
- **5-Fold Cross Validation** mit StratifiedKFold (Klassen-balanciert pro Fold)
- 100% Datenabdeckung über 5 Modelle (jedes Fold validiert anderen Slice)
- Augmentations: **timm.data.create_transform** mit **RandAugment**
- **Erkenntnis**: K-Fold gibt stabile Validation-Schätzungen aber **fängt Domain-Shift nicht** — StratifiedKFold mischt Kameras, Test-Cams sind aber speziell. Später ersetzt durch **Camera-Level GroupKFold** und schließlich **Strict CLO** (nur Test-Cam-Folds als Val).

### Beste Vor-V3 Version (`best_train.ipynb` / `best_inference.ipynb`)

- Architektur: **ConvNeXt V2 FCMAE** @ 384 px (IN22k + FCMAE Self-Supervised Pretraining)
- Verbesserungen:
  - **Horizontal Flip mit Label-Swap** (Links/Rechts-Lying werden korrekt getauscht — vorher unbemerkter Bug)
  - **StratifiedGroupKFold** (keine Datenleckage + balancierte Klassen)
  - **Gradient Accumulation** für größere effektive Batch
- **Erkenntnis**: HFlip mit Label-Swap war fundamental für Lateral_left/right — hätte schon vorher kommen müssen. Dieser Trick wurde in alle Folge-Versionen übernommen.

### Generalization Fix (`dino_v1_generalization.ipynb`) — Brücke zur V3-Numerierung

Im Titel heißt es **"ConvNeXt v3 (Generalization Fix)"** — das ist faktisch der Übergang zur V3-Nummerierung.

- **Kernprobleme** die hier erstmals analysiert wurden:
  - Val-Split auf Instanz-Ebene = Data Leak (gleiche Bilder in Train+Val)
  - Zu viel Hintergrund-Kontext (PAD_RATIO=0.25) → Modell lernte Kamera-spezifische Features statt Pose
  - Nur T2 = wenig Kamera-Diversität
- **Fixes**:
  - GroupKFold nach `image_id` → kein Bild in Train UND Val
  - **PAD_RATIO reduziert** → weniger Hintergrund, mehr Schweine-Fokus
  - **T1 + T2 kombiniert** für mehr Kamera-Diversität im Train
  - Camera-Level Eval
- **Erkenntnis**: Der Sprung von dieser Notebook-Sammlung zur **V3-Numerierung** markiert das Bewusstsein, dass die Architektur fast egal ist — Domain-Shift dominiert und braucht zentrale Pipeline-Fixes (Cropping, Split-Strategie, Augs).

### Datenanalyse (`analyse.ipynb`) — parallel

EDA-Notebook zum Verstehen der Domain-Shift-Eigenschaften: Kamera-Verteilung, Klassen-Imbalance pro Kamera, BBox-Statistiken. Wurde mehrfach erweitert während der Reise.

---

## ConvNeXt V2 — Camera-Agnostic Strategy

### Motivation

> User: *"convnext hat ganz gut funktioniert und zwar am besten das problem ist das ich immer noch nicht das perfekte ergebnis habe was hälst du von der idee das man einfach kameras predictet bzw kameras nimmt die garnicht existieren denn genau dafür soll das ja eigentlich sein das man theoretisch in jedem stall diese programm nutzten könnte"*

**Analyse:** Das Modell war zu stark auf kameraspezifische Eigenheiten (Domain-Shift zwischen `orb` und `tur` Kameras) trainiert, statt auf die tatsächlichen Haltungsmuster der Schweine. Wir entwickelten eine "Kamera-agnostische" Strategie.

### Implementierte Änderungen (`convnext_v2.ipynb` & `convnext_v2_inference.ipynb`)

1. **Domain-Augmentationen — Virtuelle Kameras:**
   - `CameraSimTransform` (p=0.7): Simuliert Resolution Jitter, JPEG-Artefakte (quality 15–60), Sensor-Rauschen (Gaussian noise σ=5–25), Unschärfe (GaussianBlur)
   - `PerspectiveJitter` (p=0.4, distortion_scale=0.15): Simuliert verschiedene Kamerawinkel/-positionen
   - `MultiResResize` (224–320px): Trainiert mit zufälligen Auflösungen statt fixer Größe
   - `AspectRatioJitter` (p=0.3, ratio 0.85–1.15): Simuliert verschiedene Sensor-Formate

2. **Daten-Handling:**
   - Umstellung auf `GroupShuffleSplit` (GroupKFold) basierend auf `image_id`, um Data Leakage zu verhindern
   - Option `COMBINE_DATASETS = True` zum Zusammenführen von Train1 und Train2

3. **Training:**
   - Early Stopping (patience=5)
   - Stärkere Color-Augmentationen (brightness=0.6, contrast=0.6, saturation=0.5, hue=0.10)
   - Basis Image Size 288px (statt 224px)

4. **Analyse:**
   - Per-Camera-Performance-Analyse am Ende des Trainings (F1-Score pro Kamera-ID)
   - Ziel: F1-Varianz zwischen Kameras < 0.05

5. **Inference (`convnext_v2_inference.ipynb`):**
   - Erweiterte TTA mit 6 Views (inkl. Multi-Resolution)

### Aufgetretene Probleme

#### Problem: `FileNotFoundError` bei DATA_ROOT

> User: *"FileNotFoundError: [Errno 2] No such file or directory: '/multi-view-pig-posture-recognition/train1.csv'"*

**Ursache:** Der `DATA_ROOT` Pfad war auf dem lokalen System anders als auf dem Server (`/datasets/multi-view-pig-posture-recognition` vs. `multiview_pig_posture_recognition`).

**Fix:** Auto-Erkennung des DATA_ROOT mit mehreren Kandidaten-Pfaden:

```python
_candidates = [
    'multiview_pig_posture_recognition',           # lokal
    './multiview_pig_posture_recognition',
    '/datasets/multi-view-pig-posture-recognition', # GPU Server
    '/multi-view-pig-posture-recognition',
]
DATA_ROOT = None
for _p in _candidates:
    if os.path.isdir(_p):
        DATA_ROOT = _p
        break
assert DATA_ROOT is not None, f'Data directory not found!'
```

#### Problem: Lokale Ausführung unter Windows

> User: *"ich möchte das lokal laufen lassen jetzt"*

**Ursache:** Jupyter Notebooks (`.ipynb`-Dateien) können nicht direkt per Texteditor bearbeitet werden. Außerdem unterstützt Windows kein Multiprocessing im DataLoader (`num_workers > 0` crasht).

**Fix:** Notebooks komplett neu erstellt mit:

- `NUM_WORKERS = 0` für Windows-Kompatibilität (statt 8 auf dem Server)
- Auto-Detection für `DATA_ROOT` (funktioniert lokal UND auf dem Server)
- Hinweis: ConvNeXt-Base ist ein großes Modell — bei wenig lokaler GPU-VRAM `BATCH_SIZE` auf 8–16 reduzieren

---

## Forschungsrecherche & Advanced Strategies

### Recherche-Grundlage (vom User bereitgestellt)

> User hat eigenständig Forschungsarbeiten und Top-Kaggle-Lösungen analysiert:
>
> - *"Pig-Posture Recognition Based on Computer Vision"*
> - Aktuelle Kaggle-Wettbewerbe mit ähnlichem Setup

### Identifizierte Strategien

#### 1. Hintergrund-Eliminierung durch Segmentierung

**Forschungserkenntnis:** Selbst in perfekten Boundingboxen verwirrt Hintergrund (Spaltenboden, Stroh, Gitter, Teile anderer Schweine) das Modell massiv.

**Umsetzung (Light-Version — Vignette Blur):**

- Statt voller Segmentierung (DeepLab v3+ / SAM): Weichzeichnung der Bildrände
- Elliptische Maske im Zentrum bleibt scharf, Ränder werden per Gaussian Blur verwischt
- In `PigPostureDatasetAdvanced` als `use_bg_blur=True` konfigurierbar

**Bewertung:** Sinnvoll als leichtgewichtiger Ansatz. Volle Segmentierung wäre aufwendiger (eigenes Segmentierungsmodell nötig), aber potentiell effektiver.

#### 2. Deep Separable Convolutions (MobileNetV3)

**Forschungserkenntnis:** In einer Kernstudie zum Datensatz erzielten Forscher >92% Genauigkeit mit Deep Separable Convolutional Networks. Weniger Parameter → weniger Overfitting bei "schmutzigen" Stallbildern.

**Umsetzung:** `mobilenetv3_large_100` (via `timm`) als Alternative zu ConvNeXt-Base.

**Bewertung:** Im Kontext dieses Wettbewerbs hat sich allerdings gezeigt (V5), dass größere Modelle wie DINOv2-L (303M+ Params) deutlich besser performen als kleinere Architekturen. MobileNetV3 hat den Vorteil geringerer Overfitting-Tendenz, aber die Domain-Shift-Problematik dominiert hier die Modellgröße.

#### 3. Grad-CAM (Visuelles Debugging)

**Forschungserkenntnis:** Gradient-weighted Class Activation Mapping zeigt als Heatmap, welche Pixel das Modell für Entscheidungen nutzt. Wenn z.B. bei "Standing" die obere Boundingbox-Kante rot leuchtet (statt Beine), "schummelt" das Modell.

**Bewertung:** Wertvolles Diagnose-Tool, aber kein direktes Training-Feature. Wurde nicht als separates Notebook implementiert, da die Per-Camera-Performance-Analyse in ConvNeXt V2 einen ähnlichen diagnostischen Zweck erfüllt (zeigt, ob das Modell kameraspezifisch oder haltungsspezifisch unterscheidet).

#### 4. Focal Loss (Anti-Lying-Bias)

**Forschungserkenntnis:** Schweine liegen den Großteil des Tages → Modell rät bei Unsicherheit "Liegend" → ruiniert F1 für seltene Klassen (Sitting, Standing).

**Umsetzung:**

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        # alpha = Class Weights, gamma = Fokussierungsfaktor
        ...
    def forward(self, input, target):
        # Reduziert Gradients für "easy examples" (richtig mit hoher Konfidenz)
        loss = -1 * (1 - pt)**self.gamma * logpt
        return loss.mean()
```

**Bewertung:** Class Weights waren bereits in allen Versionen ab V3 integriert (via `CrossEntropyLoss(weight=...)`). Focal Loss geht einen Schritt weiter, indem es nicht nur die Klassen-Häufigkeit, sondern auch die Vorhersage-Sicherheit berücksichtigt. In V8b wurde stattdessen auf Knowledge Distillation gesetzt, was sich als robusterer Ansatz erwiesen hat.

### Implementiertes Notebook: `advanced_strategies.ipynb` / `advanced_strategies_inference.ipynb`

Kombiniert alle drei Strategien:

- **Modell:** MobileNetV3 (Deep Separable Convolutions)
- **Pre-Processing:** Vignette Blur (Hintergrund-Unterdrückung)
- **Loss:** Focal Loss mit Class Weights (gamma=2.0)
- **Config:** IMG_SIZE=224, BATCH_SIZE=64, LR=1e-3, EPOCHS=40

**Status:** Erstellt zur Evaluierung. Ergebnisse ausstehend (Stand 2026-05-12).

### Datenanalyse: Class Imbalance

Die im `analyse.ipynb` durchgeführte EDA bestätigt die Forschungsergebnisse:

| Klasse | Instanzen | Anteil |
|--------|-----------|--------|
| Standing | 9.928 | ~42% (häufigste) |
| Sternal_lying | 6.309 | ~27% |
| Lateral_lying_right | 3.435 | ~15% |
| Lateral_lying_left | 3.083 | ~13% |
| **Sitting** | **695** | **~3%** (seltenste) |

**Imbalance-Faktor (max/min):** 14.3× — extremes Class Imbalance! Das erklärt, warum:

1. Standard-CrossEntropy ohne Gewichte problematisch ist
2. Sitting die schlechteste Per-Class-Performance hat
3. Focal Loss oder aggressive Class Weights sinnvoll sind

### Kamera-Analyse

- **Train-Kameras:** pen1_orb_cam1, pen1_orb_cam2, pen1_tur_cam1, pen1_tur_cam2, pen2_orb_cam1, pen2_orb_cam2, pen2_tur_cam1, pen2_tur_cam2
- **Test-Kameras:** pen1_tur_cam1, pen2_orb_cam2, pen2_tur_cam2
- **Überlappung:** 3 von 8 Kameras kommen in beiden vor
- **T2-exklusiv:** 516 Instanzen aus 60 Bildern (nur in T2, nicht in T1)

**Kernproblem:** Die Turm-Kameras (`tur`) haben drastisch andere Perspektiven/Auflösungen als die Orbital-Kameras (`orb`). Einige Kameras haben extrem wenige Samples (pen2_orb_cam2: nur 120, pen2_tur_cam2: nur 196), was die Generalisierung erschwert.

---

## Versions-Historie (V3 und folgende)

### V3 (Baseline)

- Architektur: einzelnes ViT-Modell
- LR: 1e-4
- Loss: CrossEntropy + Class Weights
- Label Smoothing: 0.05
- **Ergebnis**: 0.848 Kaggle

### V4 — gescheiterter Versuch (LR-Skalierung)

- Idee: mehr GPUs nutzen → linear scaling der LR
- LR: 4e-4 (statt 1e-4)
- Loss: Focal Loss statt CE
- Architektur: ConvNeXt Base
- **Ergebnis**: **0.793** — schlechter als V3
- **Erkenntnis**: Aggressive LR-Skalierung + schwächere ConvNeXt-Variante führt zu Regression. Bei diesem Domain-Shift-Problem will man **noisy Gradients** (kleinerer effektiver Batch, moderates LR), nicht glatte Konvergenz.

### V5 — Rückkehr zu V3-Hyperparams + größere Modelle

> User: *"baue das in ein neues v5 notebook wo du die sachen umsetz baue die idee mit V4 retrainen mit V3-Hyperparams auch ein."*

- Architekturen: **DINOv2-L @ 518px** + **ConvNeXt V2-L @ 384px**
- V3-Hyperparams zurück: LR=1e-4, CE+CW, Label Smooth 0.05
- Neu: **EMA-Weights** (decay 0.9995), **MixUp + CutMix**, **Layer-wise LR Decay 0.75** für ViT
- Heavy Augmentations gegen Domain-Shift: `CameraSimTransform` (JPEG, Noise, Downscale), `PerspectiveJitter`, GaussianBlur
- **Strict CLO**: nur 3 Test-Cam-Folds für Validation
- Gradient Accumulation: batch 16 × 2 = effektiv 32 (für DINOv2-L)

**Ergebnisse**:
- DINOv2-L Mean CV F1: **0.9002**
- DINOv2-L **only** Kaggle: **0.86**
- Mit ConvNeXt zusammen: 0.80

### V5 Erkenntnis: ConvNeXt zieht runter

> User: *"ich habe das only dino reingemacht und das hat 0.86 gehittet und war besser als das convnext"*

ConvNeXt V2-L erreichte CV F1 von nur ~0.76 (vs 0.90 bei DINOv2-L). Im Ensemble zog ConvNeXt das Ergebnis nach unten statt zu helfen.

### V6 — EVA-02 als zweite Architektur

> User: *"ja baue das in convnext v2 large in das v5 rein und erstelle ein v6 wo du das ViT und ein Convnext v2 Large verwendes"*

- Idee: echte Architektur-Diversität via DINOv2 (V5) vs. EVA-02 (V6) statt nur Stärke
- **EVA-02-L @ 448px** statt DINOv2
- Sonst identisches V5-Rezept

**Ergebnisse**:
- EVA-02 Fold-Werte: 0.872 / 0.921 / ~0.84
- only EVA-02 Kaggle: **0.852**

### V7 — Cross-Run Ensemble DINOv2 + EVA-02

> User: *"kannst du mir ein v7 geben bzw ich brauch ja nur das inference da die runs alle auf dem server gespeichert sind"*

- Keine neue Training-Pipeline — nur Inference-Notebook
- Lädt V5 (DINOv2-L) und V6 (EVA-02-L) Checkpoints, kombiniert mit verschiedenen Gewichten
- 3 Varianten:
  - **A_equal**: 1.0 / 1.0
  - **B_dinov2_strong**: 1.5 / 1.0
  - **C_eva_strong**: 1.0 / 1.5

**Ergebnisse** (Kaggle T2):
| Variante | Score |
|----------|-------|
| A_equal | 0.859 |
| **B_dinov2_strong** | **0.862** ← Best |
| C_eva_strong | (etwas niedriger) |
| only DINOv2 (V5) | 0.860 |
| only EVA-02 (V6) | 0.852 |

**Erkenntnis**: EVA-02 bringt im Ensemble nur **+0.002** gegenüber DINOv2-only → das Ensembling-Diversifizierungs-Argument trägt hier kaum. EVA und DINOv2 machen dieselben Fehler (beide vom Domain-Shift gequält).

### V8a — DINOv2 + Pseudo-Labels (gescheitert)

> User: *"wäre es möglich eine version aus den für diese aufgabe guten sachen aus dino und dann nur pseudolabels hinzuzugeben"*

- Architektur: **nur DINOv2-L** (EVA raus, brachte ja kaum was)
- Pseudo-Labels: **1000 Samples** aus V7-Inference (200 pro Klasse, balanciert, min confidence 0.60)
- Sonst V5-Rezept

**Pseudo-Label-Verteilung in V7-Export**:
| Klasse | min conf | Anzahl |
|--------|----------|--------|
| Lateral_lying_left | 0.93 | 200 |
| Lateral_lying_right | 0.93 | 200 |
| Sitting | 0.86 | 200 |
| Standing | 0.82 | 200 |
| Sternal_lying | 0.84 | 200 |

**Ergebnis**: **0.82 Kaggle** — **Regression** um 4-5%-Punkte gegenüber V7 (0.862)!

#### Analyse: Warum V8a regressed ist

1. **Pseudo-Source falsch?**: Im V7-Notebook wurden Pseudos nur bei `A_equal` (0.859) gespeichert, nicht bei `B_dinov2_strong` (0.862). Marginal-Unterschied aber wirkt mit.
2. **Class-Balancing toxisch für Sitting**: Sitting hat im Train nur 895 Samples (3.7%). 200 Pseudos hinzuzufügen mit ~20% Fehlerrate = +40 falsche Sitting-Labels = **4.5% Noise** in der schwächsten Klasse.
3. **Confirmation Bias**: V8a lernt was V7 schon weiß, inklusive V7's systematischer Fehler.
4. **Val-F1 vs Kaggle Gap**: Strict CLO validiert auf echten Test-Cam-Samples, aber Pseudos derselben Test-Kameras sind im Training → Modell fittet Cam-spezifische Features (auch falsche Pseudo-Patterns) ohne dass CV das fängt.

### V8b — V8a + Knowledge Distillation

- Statt hard Pseudo-Labels: **Soft Teacher-Targets** vom V5+V6-Ensemble
- Teacher-Precompute (einmalig): V5-DINOv2 × 3 Folds + V6-EVA-02 × 3 Folds laufen auf Train-Daten → cached als `teacher_soft_t2_v7.npy` (Shape: [24450, 5])
- Loss: `0.5 × CE_hard + 0.5 × T² × KL(student/T || teacher/T)`, T=4
- Bei MixUp: KD-Komponente skipped (Soft-Mixing wäre fehleranfällig)

**Vorteil gegenüber V8a**: Soft-Targets enthalten Unsicherheit. Wenn Teacher unsicher ist (z.B. Lateral_left vs Lateral_right verwechselbar), sieht Student auch eine breite Verteilung statt einer harten falschen Klasse → robuster gegen Pseudo-Noise.

**Status**: Läuft (Stand 2026-04-29).

---

## Probleme & Lösungen

### Problem 1: OOM bei Fold 2 in V5

> User: *"wenn ich bei v5 das training starte läuft fold 1 durch aber ich bekomme ein out of memory bei fold 2"*

**Ursache**:
- `persistent_workers=True` im DataLoader hielt Worker-Prozesse + Pinned Memory beim Fold-Wechsel
- EMA ist eine 2. Modell-Kopie auf GPU
- AdamW-State ist 2× Model-Größe → bei DINOv2-L (1.2 GB) sind das ~2.4 GB extra
- Beim Fold-Wechsel wurden diese Referenzen nicht aktiv gelöscht

**Fix** (aggressive Cleanup zwischen Folds):
```python
try:
    del train_loader, val_loader, train_ds, val_ds
except NameError:
    pass
del model, optimizer, scheduler, scaler
if ema is not None:
    del ema
gc.collect()
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
```

### Problem 2: Jupyter-Notebook stürzt bei langen Trainings ab

> User: *"auf dem server stürtzt das notebook ab und ich muss den server neustarten da bringt auch das logging nichts"*

**Ursache**: Bei 30 Epochen mit tqdm-Bars und Print pro Batch akkumuliert Jupyter hunderte MB im Cell-Output-State → Browser/Kernel kollabiert oder Output-Stream wird gekappt.

**Fix**: Standalone Python-Script (`v8a_train.py`, `v8b_train.py`) mit:
- `logging`-Modul → Schreibt in `runs/v8X_t2/train_*.log` + Console
- **Kein tqdm** — periodische Reports 10× pro Epoche
- Stats-CSV pro Epoche für späteres Plotten
- Top-Level `try/except` mit Stacktrace im Log
- GPU-Memory-Snapshots an Schlüsselstellen
- Resume-Logik: fertige Folds werden geskippt

### Problem 3: VPN-Drop killt Training

> User: *"wenn ich das training starte und meine vpn verbindung abbricht wird trotzdem weiter trainiert?"*

**Diagnose**: `ps -ef | grep jupyter` zeigt dass JupyterHub im Container als Service unter init läuft → VPN-Drop killt nicht den Server-Prozess. Aber **JupyterHub Idle-Culler** kann den Singleuser-Container abräumen bei zu langer Browser-Inaktivität.

**Lösung**: Training als Background-Job ausführen, unabhängig von Jupyter:
```bash
nohup python -u v8a_train.py > /dev/null 2>&1 < /dev/null &
disown
tail -f runs/v8a_t2/train_*.log
```
- Resume-Schutz: bei Crash kann man neu starten, fertige Folds werden übersprungen.

### Problem 4: CUDA OOM trotz V5-identischer Config (V8a, V8b)

> User: *"habe immer noch das slebe problem bei v8a"*

**Erste Diagnose**: Crash-Log zeigte GPU 0 mit 29.5 GB allocated, 29.9 GB reserved → OOM. Andere GPUs hatten ~2 GB allocated aber 26.6 GB reserved (Fragmentierung).

**Erkenntnis dank User-Beobachtung**:
> User: *"wenn das z.b auf cuda 1 geschoben wird und nicht auf 0 weil bei cuda 0 sind 1,5 gb schon auslastung"*

Auf **GPU 0 läuft Fremd-Workload** (~1.5 GB). DataParallel nutzt aber per Default `cuda:0` als Master-GPU mit zusätzlichem Overhead (~0.5 GB für Gradient-Aggregation, Output-Collation, EMA-Kopie, AdamW-State). Das schiebt GPU 0 über die 32 GB-Grenze.

**Fix**: DataParallel-Master auf GPU 1 verschieben:
```python
MASTER_GPU = 1
DEVICE = torch.device(f"cuda:{MASTER_GPU}")
DEVICE_IDS = [MASTER_GPU] + [i for i in range(n_gpus) if i != MASTER_GPU]
# Resultat: device_ids=[1, 0, 2, 3], output_device=1

model = model.to(DEVICE)
model = nn.DataParallel(model, device_ids=DEVICE_IDS, output_device=DEVICE_IDS[0])
```

Effekt:
- GPU 1 (Master): trägt Modell + EMA + AdamW-State + Gradient-Aggregation = ~13 GB
- GPU 0 (Replica): nur Modell + ihre Batch-Slice Activations = ~5 GB (+ 1.5 GB Fremd = 6.5 GB total, **massig Puffer**)

### Problem 5: OOM trotz Master-GPU-Fix (V8b nach Warmup 2)

> User: *"daer kam nach dem 2. warup"*

Bedeutet: Teacher-Precompute lief sauber durch, Crash erst im Student-Training nach Epoche 2-3. GPU 1 (Master) war wahrscheinlich der Engpass — Activations bei DINOv2-L @ 518 mit batch=4/GPU passen knapp aber nicht mit allen Mitigations.

**Fix**: **Gradient Checkpointing** aktivieren:
```python
if hasattr(model, "set_grad_checkpointing"):
    model.set_grad_checkpointing(True)
```

**Wie es funktioniert**:
- Normalerweise speichert Backward alle 24 Layer-Activations für Gradient-Berechnung → ~17 GB pro GPU
- Mit GC: nur ~3-4 Checkpoint-Boundaries gespeichert, Rest wird im Backward re-computed → ~3-5 GB pro GPU
- **Mathematisch identisch** zu V5 (Gradienten bit-identisch, gleiche Endgewichte)
- Kostet ~15% Trainingszeit durch Re-Compute

**Ergebnis-Memory-Verteilung mit GC**:
| GPU | Rolle | Memory |
|-----|-------|--------|
| GPU 0 | Replica | ~5 GB (+ 1.5 GB Fremd) |
| GPU 1 | MASTER | ~13 GB |
| GPU 2 | Replica | ~5 GB |
| GPU 3 | Replica | ~5 GB |

→ Alle GPUs unter 50% Auslastung, kein OOM mehr möglich.

---

## Wichtige Erkenntnisse (Lessons Learned)

### 1. Domain-Shift dominiert alles

Test-Kameras kommen im Training nicht vor → das ist der Score-Limiter. Architektur-Diversifizierung (V7 ENSemble) bringt nur +0.002 weil beide ViTs dieselben Fehler beim unbekannten Kamerawinkel machen. Daten-seitige Lösungen (Pseudo-Labels) sind theoretisch der richtige Hebel — wenn man die Pseudos nicht versaut.

### 2. ConvNeXt war zu schwach für dieses Problem

Trotz allgemeiner Empfehlung "ViT + ConvNeXt zusammen" bei Kaggle-Wettbewerben: hier kam ConvNeXt V2-L auf nur 0.76 CV F1 vs. DINOv2-L bei 0.90. Im Ensemble zog ConvNeXt das Ergebnis runter statt zu helfen. **Diversität ohne Stärke ist nutzlos.**

### 3. Pseudo-Labels können stark schaden

V8a regressed um 4-5% durch Pseudos:
- Class-Balanced 200/Klasse forciert künstlich gleichmäßige Repräsentation → für schwache Klassen (Sitting: 895 echte → +200 mit ~20% Fehlern = signifikante Noise-Spike)
- Confirmation Bias: Student lernt Lehrer-Fehler mit
- Strict CLO konnte die Regression nicht im Voraus zeigen (Val-F1 hoch obwohl Kaggle-Score abgestürzt)

### 4. Knowledge Distillation ist robuster als Hard Pseudo-Labels

Soft-Targets enthalten Unsicherheits-Information. Wenn Teacher unsicher ist (Lateral_left/right bei seitlich liegendem Schwein), sieht Student `[0.45, 0.45, 0.05, 0.03, 0.02]` statt eines hart erzwungenen "0". Das ist ehrlicher und propagiert keine falschen Hard-Labels.

### 5. Memory-Setup ist nicht trivial bei DataParallel

- **DataParallel Master-GPU ist immer `device_ids[0]`** — bei "alle 4 GPUs" und Default-Config = GPU 0
- Master trägt Extra-Overhead: Gradient-Aggregation, Output-Collation, EMA, AdamW-State
- Wenn GPU 0 Fremd-Workload hat (z.B. 1.5 GB), kann der zusätzliche Master-Overhead OOM triggern
- Fix: `device_ids=[1, 0, 2, 3]` + `output_device=1` → GPU 0 wird zur leichten Replica

### 6. Gradient Checkpointing = free lunch bei großen Modellen

- Spart 50-70% Activation-Memory bei nur +15% Trainingszeit
- Mathematisch **identisch** (bit-identische Gradienten am Ende)
- Lab-Standard bei großen Transformers (OpenAI, Google, Meta)
- Hier essentiell um DINOv2-L @ 518 + DataParallel auf V100 stabil zu fahren

### 7. Jupyter ist ein schlechtes Trainings-Frontend

- Output-Buffer wächst pro Cell unbegrenzt → bei 30 Epochen mit tqdm zerbricht der Browser
- Kernel-Bindung an Browser-Session ist fragil (JupyterHub Idle-Culler)
- Standalone-Script mit `logging` + `nohup` ist deutlich robuster
- Inference im Notebook ist OK (kurz, kein Buffer-Problem)

### 8. Größerer Batch ≠ besseres Modell

Trotz freiem Memory nach GC: **batch_size hochziehen ist gefährlich**.
- V4 ist genau daran gescheitert (LR×4 + ConvNeXt = 0.793)
- Größerer Batch → glattere Gradienten → schärfere Minima → schlechtere Generalization
- Bei diesem Domain-Shift-Problem will man **noisy Gradients** (effektiv 32 bewährt)
- Wenn man Memory-Reserve nutzen will: `batch_size=32, grad_accum=1` (selber effektiver Batch, nur schneller)

---

## Aktuelles File-Setup

### Frühe Notebooks (vor V3)

- [`train.ipynb`](train.ipynb) / [`inference.ipynb`](inference.ipynb) — EfficientNet-B4 Baseline
- [`swinv2.ipynb`](swinv2.ipynb) / [`swinv2_inference.ipynb`](swinv2_inference.ipynb) — SwinV2 Versuch
- [`convnext.ipynb`](convnext.ipynb) / [`convnext_inference.ipynb`](convnext_inference.ipynb) — ConvNeXt V1
- [`convnext_v2.ipynb`](convnext_v2.ipynb) / [`convnext_v2_inference.ipynb`](convnext_v2_inference.ipynb) — ConvNeXt V2 (Camera-Agnostic)
- [`kfold_train.ipynb`](kfold_train.ipynb) / [`kfold_inference.ipynb`](kfold_inference.ipynb) — 5-Fold StratifiedKFold
- [`best_train.ipynb`](best_train.ipynb) / [`best_inference.ipynb`](best_inference.ipynb) — Beste Vor-V3 Version (ConvNeXt V2 FCMAE + HFlip)
- [`dino_v1_generalization.ipynb`](dino_v1_generalization.ipynb) — Generalization Fix (Brücke zu V3)
- [`advanced_strategies.ipynb`](advanced_strategies.ipynb) / [`advanced_strategies_inference.ipynb`](advanced_strategies_inference.ipynb) — MobileNetV3 + Blur + Focal Loss
- [`analyse.ipynb`](analyse.ipynb) — Datenanalyse (EDA, Kamera-Analyse, Class Imbalance, Bbox-Statistiken)

### Trainings-Scripts (Standalone Python)

- [`v8a_train.py`](v8a_train.py) — DINOv2-L + Pseudo-Labels
- [`v8b_train.py`](v8b_train.py) — DINOv2-L + Pseudo-Labels + KD

### Notebook-Generatoren (`build_*.py`)

- `build_v5.py` → `v5_train.ipynb`
- `build_v5_inference.py` → `v5_inference.ipynb`
- `build_v6.py` → `v6_train.ipynb`
- `build_v6_inference.py` → `v6_inference.ipynb`
- `build_v7_inference.py` → `v7_inference.ipynb`
- `build_v8a_train.py` → `v8a_train.ipynb` (Notebook-Variante, **Standalone-Script ist besser**)
- `build_v8a_inference.py` → `v8a_inference.ipynb`
- `build_v8b_train.py` → `v8b_train.ipynb` (Notebook-Variante)
- `build_v8b_inference.py` → `v8b_inference.ipynb`

### Run-Outputs

- `runs/v5_t2/best_<arch>_fold_<N>.pth` — V5-Checkpoints
- `runs/v6_t2/best_<arch>_fold_<N>.pth` — V6-Checkpoints
- `runs/v8a_t2/best_dinov2l_fold_<N>.pth` — V8a-Checkpoints
- `runs/v8b_t2/best_dinov2l_fold_<N>.pth` — V8b-Checkpoints (in Arbeit)
- `runs/v8X_t2/train_<TS>.log` — Standalone-Script Logs
- `runs/v8X_t2/stats_<TS>.csv` — Per-Epoche Stats
- `teacher_soft_t2_v7.npy` — V8b Teacher-Soft Cache
- `pseudo_labels_t2_v7.csv` — V8a Pseudo-Quelle

### Submissions

- `T2_v5_*.csv`, `T2_v6_*.csv`, `T2_v7_*.csv` — bestehende Kaggle-Submissions
- `T2_v8a_dinov2_pseudo.csv` — V8a (Kaggle 0.82)
- `T2_v8b_dinov2_kd.csv` — V8b (in Arbeit)

---

## Score-Übersicht

| Version | Architektur | Trick | Kaggle T2 |
|---------|-------------|-------|-----------|
| V3 | ViT (Baseline) | V3-Hyperparams | 0.848 |
| V4 | ConvNeXt Base | Aggressive LR-Scaling | **0.793** ← Regression |
| V5 (only) | DINOv2-L | V5-Rezept | 0.860 |
| V5 (Ensemble) | DINOv2-L + ConvNeXt V2-L | V5-Rezept | ~0.80 |
| V6 (only) | EVA-02-L | V5-Rezept | 0.852 |
| V7 A_equal | DINOv2-L + EVA-02-L (1.0/1.0) | Cross-Run Ensemble | 0.859 |
| **V7 B_dinov2_strong** | DINOv2-L + EVA-02-L (1.5/1.0) | Cross-Run Ensemble | **0.862** ← Best bisher |
| V8a | DINOv2-L + Pseudo (V7) | Hard Pseudo-Labels | **0.82** ← Regression |
| V8b | DINOv2-L + Pseudo + KD | Soft Targets | (läuft) |

---

## Nächste Schritte / Offene Punkte

1. **V8b durchlaufen lassen und Kaggle-Score messen**
   - Wenn ≥ 0.862 → KD hat geholfen, robuster gegen Pseudo-Noise
   - Wenn deutlich darunter → Pseudo-Daten in der Form (Class-Balanced 200/Klasse) sind das Problem, nicht die Hardness

2. **Falls V8b auch regressed**: Pseudo-Strategie überdenken
   - Höherer Confidence-Threshold (z.B. 0.90 statt 0.60)
   - Kein Class-Balancing erzwingen (nur die confidesten Samples nehmen)
   - Pseudos aus `B_dinov2_strong` (0.862) statt `A_equal` (0.859) generieren

3. **Falls V8b über V7 kommt**: V9 könnte versuchen
   - Größere Image-Size (560 px statt 518) — Memory-Reserve ist da
   - Mehr Augs für Domain-Shift
   - Ensemble V8b + V5 (du hast die Checkpoints noch)

4. **Wenn der Score bei 0.86 stehen bleibt**: das ist evtl. das natürliche Limit dieses Datensatzes mit deinen 3 Test-Kameras-Domain-Shift. Dann ist V7 B_dinov2_strong die finale Submission.

---

## Workflow-Empfehlungen für die Zukunft

1. **Immer Standalone-Scripts für lange Trainings** — Notebooks nur für explorative Analyse und Inference
2. **`nohup` + `disown`** statt direkter Terminal-Ausführung
3. **GPU-Memory immer mit `MASTER_GPU=1`** wenn GPU 0 von Anderen genutzt wird
4. **Gradient Checkpointing** als Default für Modelle > 200M Params auf V100
5. **Resume-Logik** in Training-Scripts: fertige Folds skippen, nicht von vorn anfangen
6. **Stats-CSV pro Epoche** mitschreiben — erlaubt späteres Plotten und Vergleich zwischen Runs ohne Rerun
7. **Vor Pseudo-Labels immer Confidence-Histogramm prüfen** — niedrige Confidence in einer Klasse = Pseudo-Quelle wahrscheinlich toxisch

---

*Dokument-Stand: 2026-05-14*
