"""Ways to make species ID more accurate at recording sites the model has never heard.

Recording-level features, 5-fold cross-validation grouped by site (as site_eval.py), on species
recorded at 3+ sites. Only test recordings whose species is in that fold's training data count.
  perch           mean Perch 2.0 embedding of the recording's call windows
  calls           measured call parameters (Listen's calls.py), log-scaled, plus a CF flag
  perch+calls xW  both, call features repeated W times so they weigh more (Listen uses W=3)
  +aug            training set doubled with field-like copies (augment_windows.py), test untouched
Also reports selective accuracy: answer only when confident, otherwise fall back to the genus.
Usage: python tools/accuracy_eval.py [name]
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CALL_KEYS = ["f_char_khz", "f_start_khz", "f_end_khz", "duration_ms", "bandwidth_khz"]


def rec_means(P, rec, order):
    return np.stack([P[rec == r].mean(axis=0) for r in order])


def call_features(rows_by_uid, order):
    out = []
    for r in order:
        m = rows_by_uid.get(r)
        if not m:
            out.append([np.nan] * (len(CALL_KEYS) + 2))
            continue
        out.append([np.log(max(float(m[k]), 0.5)) for k in CALL_KEYS]
                   + [float(m["slope_khz_ms"]), 1.0 if m["shape"] in ("CF", "CF-FM") else 0.0])
    return np.array(out, float)


def model(C=0.5):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=5000, C=C, class_weight="balanced"))


def run(F, y, genus, site, F_aug=None, folds=5):
    """Returns per-test-recording (true, predicted, top probability, genus-summed top genus)."""
    out = []
    for tr, te in GroupKFold(n_splits=folds).split(F, y, site):
        Xtr, ytr = F[tr], y[tr]
        if F_aug is not None:
            Xtr, ytr = np.vstack([Xtr, F_aug[tr]]), np.r_[ytr, ytr]
        clf = model().fit(Xtr, ytr)
        pr = clf.predict_proba(F[te])
        cls = clf.classes_
        cls_genus = np.array([c.split()[0] for c in cls])
        for k, i in enumerate(te):
            if y[i] not in cls:
                continue
            g_names = np.unique(cls_genus)
            g_prob = np.array([pr[k][cls_genus == g].sum() for g in g_names])
            out.append((y[i], cls[pr[k].argmax()], pr[k].max(), genus[i], g_names[g_prob.argmax()], g_prob.max()))
    return out


def summarise(res):
    true = np.array([r[0] for r in res]); pred = np.array([r[1] for r in res]); p = np.array([r[2] for r in res])
    tg = np.array([r[3] for r in res]); pg = np.array([r[4] for r in res]); gp = np.array([r[5] for r in res])
    s = {"n": len(res), "species_acc": float((true == pred).mean()), "genus_acc": float((tg == pg).mean())}
    for t in (0.5, 0.7):
        sure = p >= t
        # unsure: fall back to the genus if the genus is confident, otherwise say "unknown"
        g_ok = (~sure) & (gp >= 0.7)
        s[f"conf{t}"] = {"answered_species": float(sure.mean()), "species_acc_when_answered": float((true[sure] == pred[sure]).mean()) if sure.any() else None,
                         "genus_fallback": float(g_ok.mean()), "genus_acc_fallback": float((tg[g_ok] == pg[g_ok]).mean()) if g_ok.any() else None,
                         "wrong_overall": float(((sure & (true != pred)) | (g_ok & (tg != pg))).mean())}
    return s


def main(name: str = "SEA_full") -> None:
    d = np.load(ROOT / "data" / "windows" / f"{name}.npz")
    P = np.load(ROOT / "data" / "windows" / f"{name}_perch.npy")
    man = {m["uid"]: m for m in json.loads((ROOT / "data" / "dataset" / name / "manifest.json").read_text(encoding="utf8"))}
    calls = {r["uid"]: r for r in json.loads((ROOT / "data" / "windows" / f"{name}_calls.json").read_text(encoding="utf8"))}
    order = sorted(set(d["rec"]))
    y = np.array([man[r]["taxon"] for r in order])
    genus = np.array([man[r]["genus"] for r in order])
    site = np.array([f"{man[r]['country']}/{man[r]['locality']}" for r in order])
    perch, cf = rec_means(P, d["rec"], order), call_features(calls, order)
    aug_path = ROOT / "data" / "windows" / f"{name}_aug_perch.npy"
    variants = {"perch": (perch, None), "calls": (cf, None)}
    for w in (1, 3, 10):
        variants[f"perch+calls x{w}"] = (np.hstack([perch, np.repeat(cf, w, axis=1)]), None)
    if aug_path.exists():
        da = np.load(ROOT / "data" / "windows" / f"{name}_aug.npz")
        pa = rec_means(np.load(aug_path), da["rec"], order)
        variants["perch +aug"] = (perch, pa)
        variants["perch+calls x3 +aug"] = (np.hstack([perch, np.repeat(cf, 3, axis=1)]), np.hstack([pa, np.repeat(cf, 3, axis=1)]))
    results = {}
    for k, (F, Fa) in variants.items():
        s = summarise(run(F, y, genus, site, Fa))
        results[k] = s
        c = s["conf0.7"]
        print(f"{k:22s} unseen-site species {s['species_acc']:.0%}  genus {s['genus_acc']:.0%}  (n={s['n']})"
              f" | confident (>=70%): answers {c['answered_species']:.0%} at {c['species_acc_when_answered']:.0%} right,"
              f" genus fallback {c['genus_fallback']:.0%} at {c['genus_acc_fallback'] or 0:.0%} right, wrong overall {c['wrong_overall']:.0%}")
    (ROOT / "results" / f"accuracy_{name}.json").write_text(json.dumps(results, indent=1), encoding="utf8")


if __name__ == "__main__":
    main(*sys.argv[1:2])
