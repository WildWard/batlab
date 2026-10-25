"""Calibrate WildWard Listen's confidence numbers so that "80%" means right about 80% of the time.

Rebuilds Listen's classifier (Perch embedding + measured call parameters x3, logistic regression
C=0.05, balanced classes), scores every recording at recording sites the model never trained on
(5-fold grouped by site, 3 shuffles), then fits an isotonic mapping from the raw top probability
to how often that answer was right. Same for the genus (summed probability of its species).
Writes the mapping to ../wildward-listen/listen/calibration.json. No audio or data leaves batlab:
the file holds only pairs of numbers.
Usage: python tools/calibrate_listen.py [name]
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
LISTEN = ROOT.parent / "wildward-listen"
sys.path.insert(0, str(LISTEN))
from listen.calls import feature_vector  # noqa: E402


def listen_model():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         LogisticRegression(max_iter=4000, C=0.05, class_weight="balanced"))


def collect(name: str, repeats: int = 3, per_species: int | None = None):
    d = np.load(ROOT / "data" / "windows" / f"{name}.npz")
    P = np.load(ROOT / "data" / "windows" / f"{name}_perch.npy")
    man = {m["uid"]: m for m in json.loads((ROOT / "data" / "dataset" / name / "manifest.json").read_text(encoding="utf8"))}
    calls = {r["uid"]: r for r in json.loads((ROOT / "data" / "windows" / f"{name}_calls.json").read_text(encoding="utf8"))}
    order = sorted(set(d["rec"]))
    emb = np.stack([P[d["rec"] == r].mean(axis=0) for r in order])
    F = np.stack([feature_vector(calls.get(r) or {}) for r in order])
    X = np.hstack([emb, np.repeat(F, 3, axis=1)])   # exactly as Survey.suggest builds its features
    y = np.array([man[r]["taxon"] for r in order])
    site = np.array([f"{man[r]['country']}/{man[r]['locality']}" for r in order])
    sp_conf, sp_ok, g_conf, g_ok = [], [], [], []
    sites = np.unique(site)
    for rep in range(repeats):
        perm = dict(zip(sites, np.random.default_rng(rep).permutation(len(sites))))
        groups = np.array([perm[s] for s in site])
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            if per_species:   # a small reference library: keep only a few training recordings per species
                rng = np.random.default_rng(100 + rep)
                tr = np.concatenate([rng.permutation(tr[y[tr] == c])[:per_species] for c in np.unique(y[tr])])
            clf = listen_model().fit(X[tr], y[tr])
            cls = clf.classes_
            genus_of = np.array([c.split()[0] for c in cls])
            genera = np.unique(genus_of)
            for i, pr in zip(te, clf.predict_proba(X[te])):
                if y[i] not in cls:   # a species never seen in training can't be named; that's the open-set problem
                    continue
                k = pr.argmax()
                sp_conf.append(pr[k]); sp_ok.append(cls[k] == y[i])
                gp = np.array([pr[genus_of == g].sum() for g in genera])
                g_conf.append(gp.max()); g_ok.append(genera[gp.argmax()] == y[i].split()[0])
    return np.array(sp_conf), np.array(sp_ok, float), np.array(g_conf), np.array(g_ok, float)


def table(conf, ok, points=21):
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(conf, ok)
    xs = np.linspace(0, 1, points)
    return [[round(float(x), 3), round(float(v), 3)] for x, v in zip(xs, iso.predict(xs))], iso


def main(name: str = "SEA_full") -> None:
    out = {"made": date.today().isoformat(),
           "source": f"batlab calibrate_listen.py on {name}: recording sites never seen in training",
           "note": "tables by training passes per species; Listen interpolates between them by its library size"}
    for size, per in (("small", 3), ("large", None)):
        print(f"--- {size} library ({per or 'all'} recordings per species)")
        sc, so, gc, go = collect(name, per_species=per)
        out[size] = report(sc, so, gc, go)
        out[size]["per_species"] = per or 12
    path = LISTEN / "listen" / "calibration.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf8")
    print("wrote", path)


def report(sc, so, gc, go):
    sp_table, sp_iso = table(sc, so)
    g_table, g_iso = table(gc, go)
    for label, conf, ok, iso in (("species", sc, so, sp_iso), ("genus", gc, go, g_iso)):
        cal = iso.predict(conf)
        print(f"{label}: {len(conf)} answers, right {ok.mean():.1%}")
        for lo, hi in ((0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
            m = (conf >= lo) & (conf < hi)
            if m.any():
                print(f"  raw {lo:.0%}-{min(hi, 1):.0%}: n={m.sum():4d}  mean raw {conf[m].mean():.0%}  actually right {ok[m].mean():.0%}  calibrated {cal[m].mean():.0%}")
    return {"species": sp_table, "genus": g_table}


if __name__ == "__main__":
    main(*sys.argv[1:2])
