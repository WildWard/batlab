"""Compare classifiers with a recording-level split (no window from a test recording is seen in training).

Features:
  baseline - mean log spectrum of the loudest frames, 12-192 kHz (no deep learning)
  perch    - Perch 2.0 embeddings of the time-expanded window (from embed_perch.py)
Usage: python tools/evaluate.py [country]
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SR = 384_000


def baseline_features(X: np.ndarray) -> np.ndarray:
    feats = []
    for w in X:
        f, _, S = signal.spectrogram(w, fs=SR, nperseg=1024, noverlap=512)
        band = (f >= 12_000) & (f <= 192_000)
        S = np.log10(S[band] + 1e-12)
        loud = S.sum(axis=0) >= np.quantile(S.sum(axis=0), 0.9)
        spec = S[:, loud].mean(axis=1)
        spec = spec - spec.max()
        feats.append(np.concatenate([spec, [f[band][np.argmax(spec)] / 1000]]))
    return np.array(feats)


def evaluate(F: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 5) -> dict:
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=0)
    win_hits, rec_true, rec_pred = [], [], []
    for tr, te in cv.split(F, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
        clf.fit(F[tr], y[tr])
        proba = clf.predict_proba(F[te])
        pred = clf.classes_[proba.argmax(axis=1)]
        win_hits += list(pred == y[te])
        for g in np.unique(groups[te]):
            idx = groups[te] == g
            rec_true.append(y[te][idx][0])
            rec_pred.append(clf.classes_[proba[idx].mean(axis=0).argmax()])
    rec_true, rec_pred = np.array(rec_true), np.array(rec_pred)
    per_class = {c: float((rec_pred[rec_true == c] == c).mean()) for c in np.unique(rec_true)}
    return {"window_acc": float(np.mean(win_hits)), "recording_acc": float((rec_true == rec_pred).mean()),
            "n_recordings": int(len(rec_true)), "chance": 1 / len(np.unique(y)), "per_class": per_class}


def main(country: str = "Malaysia") -> None:
    d = np.load(ROOT / "data" / "windows" / f"{country}.npz")
    species, genus, rec = d["species"], d["genus"], d["rec"]
    feature_sets = {"baseline": baseline_features(d["X"])}
    perch_path = ROOT / "data" / "windows" / f"{country}_perch.npy"
    if perch_path.exists():
        feature_sets["perch"] = np.load(perch_path)
    results = {}
    for name, F in feature_sets.items():
        for level, y in (("species", species), ("genus", genus)):
            r = evaluate(F, y, rec)
            results[f"{name}/{level}"] = r
            print(f"{name:8s} {level:7s} recording acc {r['recording_acc']:.0%}  window acc {r['window_acc']:.0%}"
                  f"  (chance {r['chance']:.0%}, {r['n_recordings']} recordings)")
    out = ROOT / "results" / f"eval_{country}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf8")
    print("class counts:", Counter(species[np.unique(rec, return_index=True)[1]]))


if __name__ == "__main__":
    main(*sys.argv[1:2])
