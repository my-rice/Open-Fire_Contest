<div align="center">

# ONFIRE Contest 2023 – Early Fire Detection from Video

Lightweight, modular PyTorch pipeline for rapid wildfire / smoke onset detection in surveillance video streams. Originally built for the 2023 **ONFIRE (Open Fire) Contest**.

</div>

---

## 🔥 Key Features
* Multiple CNN backbones: FireNet, FireNetV2, MobileNetV2, MobileNetV3 Small, ResNet18, ResNet50.
* Frame-level sparse temporal sampling with segment-based dataset loader.
* K-fold cross validation utilities (implemented inside `train.ipynb`).
* Unified metrics (precision, recall, accuracy, delay & normalized delay).
* Training / validation curve aggregation and plotting from raw TensorBoard event exports.
* Inference script (`test.py`) for test set video evaluation.
* CSV summarization + aggregated statistics per experiment.

---

## 🏗 Project Structure (Refactored)
```
scripts/
    summary.py           # Fold aggregation (mean/std/min/max/median)
    plots.py             # fold plot generator
    to_csv.py            # event export → CSV
    converter.py         # .avi → .mp4 helper (ffmpeg wrapper)
    graphs.py            # Plot training vs validation curves from CSV logs
    metrics.py           # Evaluation utilities (callable functions)
custom_models/
    FireNet.py          # Original Keras → PyTorch port
    FireNetV2.py        # Improved FireNetV2 architecture
    models.py           # Generic builders and optimizer helpers
best_model/
    best_model.pth      # Best model weights (MobileNetV3)
train.ipynb             # End‑to‑end training workflow (data prep → K-fold training)
test.ipynb              # Inference / result generation

```

---

## 🚀 Quick Start
### 1. Install Dependencies
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Data
Extract frames & annotations as per contest specification. Frame extraction & dataset assembly utilities are inside the initial cells of `train.ipynb` (sections: download → extract → dataset build).

### 3. Train Models
Open `train.ipynb` and run sequentially. Each attempt section ("Attempt X: ModelName") defines:
* Model instantiation
* Optimizer / scheduler setup
* K-fold loop
* Logging to TensorBoard

---

## 📊 Metrics Implemented
* TP / TN / FP / FN
* Accuracy, Precision, Recall
* Average Delay (frames)
* Normalized Average Delay (rewarding early detection)

Delay is computed only on true positives; early guesses before allowed tolerance (`delta_t`) count as false positives.

---

## 🧪 K-Fold Strategy
Implemented manually inside notebook; each fold logs separate TensorBoard scalars which are post-processed into CSV → summary stats.

---

## 📚 Dependencies
See `requirements.txt` for a concrete list of dependencies.

