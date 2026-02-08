# MetaMatcher Code-Analyse & Verbesserungen

## Kurzfazit
Der aktuelle MetaMatcher ist bereits gut strukturiert (Kalibrierung, Gating, mehrere Backbones), hat aber einige **methodische Risiken** (z. B. Test-Set für Model-Selection), **architektonische Inkonsistenzen** (Attention-Parameter werden nur teilweise genutzt) und **Trainingsstabilitäts-Themen** (fehlende Scheduler/Early-Stopping/Gradient-Clipping).

## 1) Methodik / Evaluation (höchste Priorität)

### Problem A: Checkpoint-Auswahl über Test-F1
In `trainer.py` wird das beste Modell bevorzugt über `test_f1` gewählt, falls Test-Metriken verfügbar sind. Das führt zu Leakage und optimistischen Ergebnissen.

**Empfehlung**
- Für Modellwahl ausschließlich Validation-Metriken nutzen.
- Test-Set nur einmalig am Ende reporten.
- `cfg.best_metric` konsequent auswerten (z. B. `val_f1`, `val_auc`, `val_loss`).

**Nutzen**
- Saubere wissenschaftliche Auswertung.
- Bessere Vergleichbarkeit zwischen Runs.

### Problem B: `best_metric`-Optionen nicht konsistent umgesetzt
Konfigurationsoptionen enthalten auch `test_*`, die in der Selektion nicht vollständig und sauber verarbeitet werden.

**Empfehlung**
- `best_metric` strikt auf `val_*` begrenzen.
- Falls `test_*` gewünscht, explizit als „Reporting only“ behandeln.

## 2) Modellarchitektur (hoch)

### Problem C: Aktivierungsfunktion ist faktisch hart auf ReLU
Der Code setzt `act = nn.ReLU if cfg.activation == "relu" else nn.ReLU`.

**Empfehlung**
- Echte Auswahl implementieren (z. B. `relu`, `gelu`, `silu`).
- Optional: zentraler Resolver `get_activation(name)`.

### Problem D: `attn_mode` existiert, wird aber nicht wirklich verwendet
In `config.py` gibt es `attn_mode` (`gating` | `token`), aber im Modell wird nur Gating benutzt.

**Empfehlung**
- Entweder `attn_mode` entfernen (wenn nicht geplant),
- oder wirklich zwei Pfade implementieren:
  - `gating`: aktuelles alpha-basiertes Score-Mixing,
  - `token`: Self-Attention über Modell-Tokens.

### Problem E: RNN-Branch nutzt nur letzten Zeitschritt
Bei bidirektionalem RNN/LSTM ist `out[:, -1, :]` oft suboptimal.

**Empfehlung**
- Stattdessen finalen Hidden-State verwenden (`h_n`) oder Pooling (`mean/max`) über alle Token.
- Optional mit Attention-Pooling kombinieren.

## 3) Training & Stabilität (hoch)

### Problem F: Fehlende Trainingsstabilisierer
Aktuell fehlen u. a. LR-Scheduler, Early-Stopping und Gradient-Clipping.

**Empfehlung**
- `ReduceLROnPlateau` oder Cosine Scheduler integrieren.
- Early-Stopping mit Patience auf Validation-Metrik.
- Gradient-Clipping (`clip_grad_norm_`) für RNN/LSTM.

### Problem G: Entropie-Regularisierung ist statisch
`alpha_entropy_weight` ist konstant; zu Beginn sinnvoll, später evtl. hinderlich.

**Empfehlung**
- Gewicht über Epochen annealen (z. B. linear/cosine decay).
- Zusätzlich Monitoren: Korrelation zwischen Entropie und Val-F1.

## 4) Kalibrierung & Loss (mittel)

### Problem H: Score-Kalibrierung als volle lineare Matrix kann Overfitting fördern
`Linear(M,M)` mit voller Kopplung kann bei wenigen Daten zu instabilen Wechselwirkungen führen.

**Empfehlung**
- Start mit restriktiver Kalibrierung:
  - diagonal (pro Modell `a_i * s_i + b_i`),
  - oder low-rank + Residual.
- Optional temperaturbasierte Kalibrierung pro Modell.

### Problem I: Klassengewichtung/Focal Loss fehlt
Bei Imbalance kann BCE/CE alleine unterperformen.

**Empfehlung**
- Für `BCEWithLogitsLoss`: `pos_weight` aus Trainingsdaten ableiten.
- Optional Focal Loss als konfigurierbare Alternative.

## 5) Daten/Features (mittel)

### Problem J: Pair-Embedding wird in RNN für jedes Modelltoken repliziert
Das kann modell-spezifische Information verwässern.

**Empfehlung**
- Modell-ID-Embedding je Token ergänzen.
- Pro Basismodell zusätzliche Meta-Features (z. B. historische Kalibrierungsfehler).

### Problem K: Unscharfe Input-Kontrakte
`scores` werden als [0,1] angenommen, aber keine Assertions/Checks vorhanden.

**Empfehlung**
- Validierungschecks in Debug-Modus:
  - Shape-Checks,
  - Range-Checks,
  - NaN/Inf-Schutz.

## 6) Reproduzierbarkeit & Engineering (mittel)

### Problem L: Seed-Management nicht zentral sichtbar
Für faire Experimente fehlen deterministische Settings im Trainingspfad.

**Empfehlung**
- `set_seed(seed)` zentral einbauen (Python, NumPy, Torch, CUDA).
- In Outputs Seed + Git-Commit + Config persistieren.

### Problem M: Logging/Tracking begrenzt
Bisher primär Konsolen-Logs + JSON.

**Empfehlung**
- Optionales Experiment-Tracking (TensorBoard/W&B/MLflow).
- Zusätzliche Artefakte: PR-Kurven, Kalibrierungsdiagramme, Confusion Matrix.

## 7) Konkreter Umsetzungsplan (2 Wochen)

1. **Woche 1 – Saubere Evaluation & Stabilität**
   - Best-Checkpoint nur nach Validation.
   - Early-Stopping + Scheduler + Gradient-Clipping.
   - Seed-Fixierung + klarere Config-Validierung.

2. **Woche 2 – Architekturverbesserung**
   - Aktivierungs-Auswahl + optionale restriktive Kalibrierung.
   - RNN-Pooling verbessern.
   - Optional `attn_mode=token` implementieren oder entfernen.

## Quick Wins (geringer Aufwand, hoher Impact)
- Test-Set aus Checkpoint-Selektion entfernen.
- Activation-Bug beheben.
- Early-Stopping + ReduceLROnPlateau ergänzen.
- `pos_weight` für BCE einführen.

