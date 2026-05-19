# Mittelstufenprojekt – Multi-View Pig Posture Recognition

Deep Learning Projekt im Rahmen der **Precision Livestock Farming (PLF)**: Klassifizierung von Schweinehaltungen in **5 Klassen** auf Basis von 2D-Bilddaten (Datensätze der Michigan State University).

**Kaggle-Wettbewerb:** [Multi-View Pig Posture Recognition](https://www.kaggle.com/competitions/multi-view-pig-posture-recognition)
**Metrik:** Macro-averaged F1-Score
**Framework:** PyTorch
**Hardware:** IBM Power9 / Tesla V100 (bis zu 4× SXM2 32GB)

---

## Repository-Struktur

Das Repository ist nach der in der Aufgabenstellung geforderten Struktur aufgebaut: ein Unterordner pro Teammitglied für die persönlichen Experimente, sowie ein `Merge/`-Ordner für die finale Zusammenführung.

```
Mittelstufenprojekt-dev/
├── Daniel/      → Persönliche Experimente von Daniel
├── Fabian/      → Persönliche Experimente von Fabian
├── Mert/        → Persönliche Experimente von Mert
├── merged/       → Konsolidierte EDA + Best-of-Model
├── README.md    → Diese Datei
└── .gitignore
```

### Ordner-Übersicht

| Ordner | Inhalt | README |
|---|---|---|
| [`Daniel/`](Daniel/) | Individuelle Experimente von Daniel | – |
| [`Fabian/`](Fabian/) | Individuelle Experimente von Fabian (V3–V8, DANN, KFold, Ensemble-Strategien) | – |
| [`Mert/`](Mert/) | Individuelle Experimente von Mert (Baseline → ResNet50 → ConvNeXt → EVA02 → ConvNeXt-V2-Huge) | [README](Mert/README.md) |
| [`Merge/`](merged/) | **Einstiegspunkt für die finalen Ergebnisse:** Konsolidierte EDA + Best-of-Model (V5 Ensemble) | [README](Merge/README.md) |

---

## Best-of-Model

Das finale Modell liegt im [`Merge/`](Merge/)-Ordner: ein **Ensemble aus DINOv2 Large (ViT) + ConvNeXt V2 Large (CNN)** (Version V5), trainiert auf Train2 mit Strict CLO-Validierung über die Test-Kamera-Folds. Details siehe [Merge/README.md](Merge/README.md).

---

## Klassen

| ID | Bezeichnung |
|---|---|
| 0 | Lateral_lying_left |
| 1 | Lateral_lying_right |
| 2 | Sitting |
| 3 | Standing |
| 4 | Sternal_lying |

---

## Datensätze

| Datensatz | Charakteristik |
|---|---|
| **Train1 (T1)** | Erfordert Generalisierung auf völlig unbekannte Kameraperspektiven und Tiergruppen |
| **Train2 (T2)** | Erlaubt begrenzte Domänenadaption durch einen kleinen Anteil bekannter Kameraperspektiven |
| **Test** | Bewertung über Macro F1 auf der Kaggle-Plattform |

**Zentrale Herausforderung:** Distribution Shift – sowohl Klassenungleichgewicht (Imbalance-Ratio ~14,3x) als auch Kameraverteilungs-Shift (Train ~58% cam1 vs. Test ~36% cam1).


---

## Einstieg

Für einen schnellen Überblick → [`Merge/README.md`](Merge/README.md) lesen. Dort liegen die konsolidierte EDA und das finale Modell mit ausführlicher Dokumentation der Hyperparameter und der Ensemble-Strategie.
