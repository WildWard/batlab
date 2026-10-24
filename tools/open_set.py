"""Can the model say "I don't know this species"?

Leave-one-species-out: train on every species but one, then score how unfamiliar each test
recording looks. Known-species test recordings come from sites not used in training for that
fold's split; the hidden species' recordings are all "unknown". Compares three scores:
  maxprob  - 1 minus the classifier's top probability
  centroid - cosine distance to the nearest species' mean embedding
  knn      - mean cosine distance to the 3 nearest training recordings
Reports AUROC (1.0 = perfect separation of unknown from known, 0.5 = chance) and, for a
threshold that flags a given share of unknowns, how many known recordings get wrongly flagged.
Usage: python tools/open_set.py [name]
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def main(name: str = "SEA") -> None:
    d = np.load(ROOT / "data" / "windows" / f"{name}.npz")
    P = np.load(ROOT / "data" / "windows" / f"{name}_perch.npy")
    man = {m["uid"]: m for m in json.loads((ROOT / "data" / "dataset" / name / "manifest.json").read_text(encoding="utf8"))}
    recs = sorted(set(d["rec"]))
    X = np.stack([P[d["rec"] == r].mean(axis=0) for r in recs])
    y = np.array([d["species"][d["rec"] == r][0] for r in recs])
    site = np.array([man[r]["locality"] for r in recs])
    scores = {"maxprob": [], "centroid": [], "knn": []}
    is_unknown, known_correct = [], []
    for hidden in sorted(set(y)):
        known = y != hidden
        # hold out a fifth of the known sites as known test data, train on the rest
        tr_k, te_k = next(GroupKFold(5).split(X[known], y[known], site[known]))
        Xk, yk = X[known], y[known]
        Xtr, ytr = Xk[tr_k], yk[tr_k]
        Xte = np.vstack([Xk[te_k], X[~known]])
        unk = np.r_[np.zeros(len(te_k)), np.ones((~known).sum())]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)).fit(Xtr, ytr)
        pr = clf.predict_proba(Xte)
        scores["maxprob"] += list(1 - pr.max(axis=1))
        U_tr, U_te = unit(Xtr), unit(Xte)
        cents = unit(np.stack([U_tr[ytr == c].mean(axis=0) for c in clf.classes_]))
        scores["centroid"] += list(1 - (U_te @ cents.T).max(axis=1))
        sims = np.sort(U_te @ U_tr.T, axis=1)[:, -3:]
        scores["knn"] += list(1 - sims.mean(axis=1))
        is_unknown += list(unk)
        known_correct += list(np.r_[clf.classes_[pr[: len(te_k)].argmax(axis=1)] == yk[te_k], np.zeros((~known).sum(), bool)])
    is_unknown = np.array(is_unknown, bool)
    known_correct = np.array(known_correct, bool)
    out = {}
    for k, v in scores.items():
        v = np.array(v)
        auc = roc_auc_score(is_unknown, v)
        line = f"{k:9s} AUROC {auc:.2f}"
        for catch in (0.5, 0.7):
            th = np.quantile(v[is_unknown], 1 - catch)       # flags `catch` share of unknowns
            false_flag = (v[~is_unknown] >= th).mean()
            kept_right = known_correct[~is_unknown][v[~is_unknown] < th].mean()
            line += f" | catch {catch:.0%} unknowns: {false_flag:.0%} of known flagged, unflagged known {kept_right:.0%} right"
        print(line)
        out[k] = auc
    (ROOT / "results" / f"open_set_{name}.json").write_text(json.dumps(out, indent=1), encoding="utf8")


if __name__ == "__main__":
    main(*sys.argv[1:2])
