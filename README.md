# Urban Change Detection : Feasibility Probes

Feasibility experiments for a research project on **monitoring unplanned
construction and urban sprawl from satellite imagery**, with an
uncertainty-aware, human-in-the-loop review step (uncertain areas are flagged
for a human rather than auto-decided).

These are **go/no-go probes** that verify the data carries a strong, learnable
signal before committing to the full study. They deliberately use simple models
(frozen CNN embeddings + classical classifiers) — dedicated models would do
better; the point is to show the signal is real.

## Results (reproducible, fixed seeds)

### 1. Built-up vs natural land classification — `src/probe_classification.py`
- **Dataset:** EuroSAT (Sentinel-2 satellite imagery, 10 land-use classes; via `torchvision`).
- **Task (binary):** built-up `{Residential, Industrial, Highway}` vs natural `{Forest, Pasture, HerbaceousVegetation, AnnualCrop, PermanentCrop, River, SeaLake}`.
- **Probe sample:** 8,000 images (≤800 per original class, seed 0); 2,400 built-up / 5,600 natural.
- **Features:** frozen MobileNetV3-Small (ImageNet) embeddings — no fine-tuning.
- **Split:** 70/30 stratified, `random_state=0`.

| Model | Accuracy | Balanced acc | F1 | AUROC |
|-------|----------|--------------|-----|-------|
| Logistic Regression | 0.948 | 0.941 | 0.914 | 0.985 |
| Random Forest | 0.932 | 0.904 | 0.880 | 0.983 |

### 2. Change detection (new construction between two dates) — `src/probe_changedetection.py`
- **Dataset:** LEVIR-CD (real before/after building change-detection pairs with change masks; `ericyu/LEVIRCD_Cropped256` test shard on Hugging Face, auto-downloaded).
- **Task (binary proxy):** "did substantial change occur in this patch?" (change-mask foreground > 2%). 2,048 patches, 36% change-positive.
- **Features:** `|MobileNetV3 embedding difference|` of before/after + pixel-difference statistics.
- **Split:** 70/30 stratified, `random_state=0`.

| Accuracy | Balanced acc | F1 | AUROC |
|----------|--------------|-----|-------|
| 0.836 | 0.796 | 0.740 | 0.922 |

## Honest caveats
- **EuroSAT is European** — used here only as a fast feasibility proxy. The full
  study would use region-specific imagery (e.g. the India-focused *IndiaSat* and
  NASA India built-up datasets) and bi-temporal pairs.
- The change-detection number is from a **simple proxy**; proper segmentation
  models (BIT-CD, TinyCD, etc.) are expected to do considerably better.
- These probes answer *"is the signal real?"* — not *"here is the final model."*

## Setup & run
```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt

python src/probe_classification.py       # downloads EuroSAT to ./data on first run
python src/probe_changedetection.py      # downloads LEVIR-CD test shard to ./data
```
Both are CPU-only. Datasets download automatically into `./data/` (git-ignored).
Metrics are written to `results/*.json`.

## Repository layout
```
src/probe_classification.py       built-up vs natural classification probe
src/probe_changedetection.py      change-detection feasibility probe
results/classification_feasibility.json
results/changedetection_feasibility.json
requirements.txt
```
Datasets are **not** committed (see `.gitignore`); they download on first run.
