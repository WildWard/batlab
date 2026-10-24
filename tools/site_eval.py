"""Unseen-site evaluation: every fold tests on localities the classifier never trained on.

Compares feature variants and site normalisation tricks on the same split.
Only test recordings whose species also appears in that fold's training data are scored.
Usage: python tools/site_eval.py [country ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import baseline_features

ROOT = Path(__file__).resolve().parent.parent


def load(countries: list[str]):
    parts = []
    for c in countries:
        d = np.load(ROOT / "data" / "windows" / f"{c}.npz")
        manifest = {m["uid"]: m for m in json.loads((ROOT / "data" / "dataset" / c / "manifest.json").read_text(encoding="utf8"))}
        perch = np.load(ROOT / "data" / "windows" / f"{c}_perch.npy")
        site = np.array([f"{c}:{manifest[u]['locality']}" for u in d["rec"]])
        parts.append(dict(X=d["X"], perch=perch, species=d["species"], genus=d["genus"], rec=d["rec"], site=site))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def site_centre(F: np.ndarray, site: np.ndarray) -> np.ndarray:
    """Subtract each site's mean feature vector. Uses no labels, so it's fair at test time:
    a real survey has all of its own recordings available, just not their species."""
    out = F.copy()
    for s in np.unique(site):
        i = site == s
        out[i] -= F[i].mean(axis=0)
    return out


def site_zscore(F: np.ndarray, site: np.ndarray) -> np.ndarray:
    out = F.copy()
    for s in np.unique(site):
        i = site == s
        out[i] = (F[i] - F[i].mean(axis=0)) / (F[i].std(axis=0) + 1e-6)
    return out


def unseen_site_score(F, y, rec, site, C=0.5, folds=5):
    hits, n_classes_tested = [], set()
    for tr, te in GroupKFold(n_splits=folds).split(F, y, site):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=C, class_weight="balanced"))
        clf.fit(F[tr], y[tr])
        proba = clf.predict_proba(F[te])
        known = set(clf.classes_)
        for r in np.unique(rec[te]):
            i = rec[te] == r
            label = y[te][i][0]
            if label not in known:
                continue
            n_classes_tested.add(label)
            hits.append(clf.classes_[proba[i].mean(axis=0).argmax()] == label)
    return float(np.mean(hits)), len(hits), len(n_classes_tested)


def main(countries: list[str]) -> None:
    d = load(countries)
    base = baseline_features(d["X"])
    variants = {
        "perch": d["perch"],
        "perch + site centre": site_centre(d["perch"], d["site"]),
        "perch + site z-score": site_zscore(d["perch"], d["site"]),
        "baseline": base,
        "baseline + site centre": site_centre(base, d["site"]),
        "perch+baseline + site centre": np.hstack([site_centre(d["perch"], d["site"]), site_centre(base, d["site"])]),
    }
    print(f"{len(np.unique(d['rec']))} recordings, {len(np.unique(d['site']))} sites, {len(np.unique(d['species']))} species")
    results = {}
    for name, F in variants.items():
        for level in ("species", "genus"):
            acc, n, k = unseen_site_score(F, d[level], d["rec"], d["site"])
            results[f"{name}/{level}"] = acc
            print(f"{name:32s} {level:7s} {acc:5.0%}  ({n} recordings, {k} classes testable)", flush=True)
    out = ROOT / "results" / f"site_eval_{'_'.join(countries)}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf8")


if __name__ == "__main__":
    main(sys.argv[1:] or ["Malaysia"])
