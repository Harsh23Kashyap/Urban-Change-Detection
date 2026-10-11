#!/usr/bin/env python3
"""
Feasibility probe 2 — Change detection (new construction between two dates).

Question: do before/after satellite image pairs carry a learnable signal for
"new construction appeared here"? Tested on LEVIR-CD (real bi-temporal building
change-detection pairs with ground-truth change masks).

The full task is pixel-level segmentation of new buildings. For a fast go/no-go
we reduce it to a per-patch binary task: "did substantial change occur?"
(change-mask foreground > 2% of the patch), using |embedding difference| of the
before/after images plus simple pixel-difference statistics. If a simple model
already recovers the signal, dedicated segmentation models (e.g. BIT-CD, TinyCD)
do considerably better.

The LEVIR-CD test shard is downloaded from Hugging Face into ./data on first run.
Run:  python src/probe_changedetection.py
"""
import warnings; warnings.filterwarnings("ignore")
import io, json, urllib.request
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch, torch.nn as nn
from torchvision import models, transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score, f1_score)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results"
DATA.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
PARQUET = DATA / "levircd_test.parquet"
URL = ("https://huggingface.co/datasets/ericyu/LEVIRCD_Cropped256/"
       "resolve/main/data/test-00000-of-00001-31d7c3e3444e5b5d.parquet")

device = torch.device("cpu")
torch.manual_seed(0); np.random.seed(0)

if not PARQUET.exists():
    print("Downloading LEVIR-CD test shard (~70 MB) from Hugging Face...")
    urllib.request.urlretrieve(URL, PARQUET)

tbl = pq.read_table(PARQUET).to_pydict()
N = len(tbl["imageA"])
print(f"LEVIR-CD samples: {N}")

def load_img(d, size=112):
    return Image.open(io.BytesIO(d["bytes"])).convert("RGB").resize((size, size))

def load_mask(d, size=112):
    return np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("L").resize((size, size)))

change_frac = np.array([(load_mask(tbl["label"][i]) > 127).mean() for i in range(N)])
y = (change_frac > 0.02).astype(int)      # >2% of patch changed = "change"
print(f"Change-positive patches: {y.sum()}/{N} ({100*y.mean():.0f}%)")

net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
net.classifier = nn.Identity(); net.eval().to(device)
tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])])

@torch.no_grad()
def embed_pair(i):
    a = tf(load_img(tbl["imageA"][i])); b = tf(load_img(tbl["imageB"][i]))
    ea = net(a.unsqueeze(0)).squeeze(0).numpy()
    eb = net(b.unsqueeze(0)).squeeze(0).numpy()
    arrA = np.asarray(load_img(tbl["imageA"][i]), np.float32) / 255
    arrB = np.asarray(load_img(tbl["imageB"][i]), np.float32) / 255
    d = np.abs(arrA - arrB)
    pix = [d.mean(), d.std(), d.max(), (d > 0.2).mean()]
    return np.concatenate([np.abs(ea - eb), pix]).astype(np.float32)

print("Extracting change features...")
X = np.vstack([embed_pair(i) for i in range(N)])
print("Feature dim:", X.shape[1])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                      stratify=y, random_state=0)
clf = make_pipeline(StandardScaler(),
                    RandomForestClassifier(n_estimators=400, random_state=0,
                                           class_weight="balanced", n_jobs=-1))
clf.fit(Xtr, ytr)
p = clf.predict_proba(Xte)[:, 1]; pred = (p >= .5).astype(int)
res = {
    "dataset": "LEVIR-CD (ericyu/LEVIRCD_Cropped256 test shard, Hugging Face)",
    "n": int(N), "change_positive_frac": round(float(y.mean()), 3),
    "label_rule": "mask foreground > 2% of patch = change",
    "features": "|MobileNetV3 embedding diff| + pixel-difference stats",
    "split": "70/30 stratified, random_state=0",
    "accuracy": round(float(accuracy_score(yte, pred)), 4),
    "balanced_accuracy": round(float(balanced_accuracy_score(yte, pred)), 4),
    "f1": round(float(f1_score(yte, pred)), 4),
    "auroc": round(float(roc_auc_score(yte, p)), 4),
}
json.dump(res, open(RESULTS / "changedetection_feasibility.json", "w"), indent=2)
print("\n=== CHANGE-DETECTION FEASIBILITY ===")
print(json.dumps({k: v for k, v in res.items()
                  if k in ("accuracy", "balanced_accuracy", "f1", "auroc")}, indent=2))
print("Saved results/changedetection_feasibility.json")
