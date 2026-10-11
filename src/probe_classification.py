#!/usr/bin/env python3
"""
Feasibility probe 1 — Built-up vs natural land classification.

Question: can a model separate BUILT-UP (construction/urban) land from NATURAL
land in real Sentinel-2 satellite imagery (EuroSAT)?

  built-up = Residential, Industrial, Highway
  natural  = Forest, Pasture, HerbaceousVegetation, AnnualCrop,
             PermanentCrop, River, SeaLake

Method: frozen MobileNetV3-Small (ImageNet) embeddings + Logistic Regression
and Random Forest, with an honest stratified train/test split. A balanced
subsample is used for a fast go/no-go probe (not the final model).

EuroSAT is downloaded automatically by torchvision into ./data on first run.
Run:  python src/probe_classification.py
"""
import warnings; warnings.filterwarnings("ignore")
import json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import EuroSAT
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score, f1_score)

# repo-relative paths (portable: works wherever the repo is cloned)
REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results"
DATA.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)

device = torch.device("cpu")
torch.manual_seed(0); np.random.seed(0)
BUILTUP = {"Residential", "Industrial", "Highway"}

tf = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print("Loading EuroSAT (auto-downloads to ./data on first run)...")
ds = EuroSAT(root=str(DATA), download=True, transform=tf)
classes = ds.classes
builtup_idx = {i for i, c in enumerate(classes) if c in BUILTUP}

# balanced subsample for a fast probe: up to 800 per original class
labels_all = np.array([lbl for _, lbl in ds.samples])
rng = np.random.RandomState(0)
keep = []
for c in range(len(classes)):
    idx = np.where(labels_all == c)[0]
    rng.shuffle(idx)
    keep += list(idx[:800])
keep = np.array(keep)
y_bin = np.array([1 if labels_all[i] in builtup_idx else 0 for i in keep])
print(f"Probe set: {len(keep)} images | built-up={y_bin.sum()} natural={len(keep)-y_bin.sum()}")

# frozen CNN embeddings
net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
net.classifier = nn.Identity(); net.eval().to(device)

@torch.no_grad()
def embed():
    dl = DataLoader(Subset(ds, keep.tolist()), batch_size=64, shuffle=False)
    feats, t0 = [], time.time()
    for i, (x, _) in enumerate(dl):
        feats.append(net(x.to(device)).cpu().numpy())
        if (i + 1) % 20 == 0:
            print(f"   embedded {min((i+1)*64,len(keep))}/{len(keep)} ({time.time()-t0:.0f}s)")
    return np.vstack(feats)

print("Extracting embeddings...")
X = embed()
Xtr, Xte, ytr, yte = train_test_split(X, y_bin, test_size=0.3,
                                       stratify=y_bin, random_state=0)

results = {}
for name, clf in [
    ("logreg", make_pipeline(StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"))),
    ("rf", make_pipeline(StandardScaler(),
        RandomForestClassifier(n_estimators=400, random_state=0,
                               class_weight="balanced", n_jobs=-1))),
]:
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    pred = (p >= .5).astype(int)
    results[name] = {
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(yte, pred)), 4),
        "f1": round(float(f1_score(yte, pred)), 4),
        "auroc": round(float(roc_auc_score(yte, p)), 4),
    }
    print(f"  {name}: {results[name]}")

out = {"dataset": "EuroSAT (Sentinel-2, torchvision)",
       "n_probe": int(len(keep)), "builtup": int(y_bin.sum()),
       "classes": classes, "builtup_classes": sorted(BUILTUP),
       "split": "70/30 stratified, random_state=0",
       "features": "frozen MobileNetV3-Small embeddings (ImageNet)",
       "results": results}
json.dump(out, open(RESULTS / "classification_feasibility.json", "w"), indent=2)
print("\nSaved results/classification_feasibility.json")
