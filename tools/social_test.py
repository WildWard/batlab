"""Can we tell bats talking from bats navigating, on species the model has never heard?

Label: ChiroVox call type, distress/social vs search (sonar). Each recording is scored by a model trained
only on other species, so a score can't come from recognising the species. Three feature sets:
  shape     measured event shape (event_features.py)
  perch     Perch 2.0 on the loudest windows
  quiet     Perch 2.0 on the quietest windows, the control: if background alone separates the labels,
            the difference is the recording setup, not the bat
Talking and sonar recordings come mostly from different devices (D1000X is the only one with both), so a
score across all devices is mostly the device. The honest scope is one device, trained and tested there;
"exact" narrows it further to groups holding both labels from the same species, site and method.
p-values shuffle labels within each species.
Usage: python tools/social_test.py [name] [permutations]
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
SHAPE = ["dur_ms", "f_peak", "f_low", "rise", "range", "turns", "harm", "flat"]
LOG = {"dur_ms", "f_peak", "f_low"}
MIN_SNR = 30        # most detected events sit near the threshold and are noise, not calls
MIN_EVENTS = 5
DEVICE = "D1000X"


def shape_row(ev: list[dict], seconds: float) -> list[float]:
    ev = [e for e in ev if e["snr"] >= MIN_SNR]
    if len(ev) < MIN_EVENTS:
        return [np.nan] * (len(SHAPE) * 2 + 1)
    row = []
    for k in SHAPE:
        v = np.array([e[k] for e in ev], float)
        v = np.log(np.maximum(v, 0.5)) if k in LOG else v
        row += [np.median(v), np.percentile(v, 90)]
    return row + [np.log(len(ev) / max(seconds, 0.1))]


def group_scores(F, y, groups, C, folds=5, seed=0):
    """Out-of-fold scores; a species is never in both train and test."""
    keep = ~np.isnan(F).any(axis=1)
    idx = np.flatnonzero(keep)
    scores = np.full(len(y), np.nan)
    g = np.array(groups)[idx]
    uniq = {s: i for i, s in enumerate(np.random.default_rng(seed).permutation(np.unique(g)))}
    fold = np.array([uniq[s] % folds for s in g])
    for k in range(folds):
        tr, te = idx[fold != k], idx[fold == k]
        if len(set(y[tr])) < 2 or len(te) == 0:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C, class_weight="balanced"))
        m.fit(F[tr], y[tr])
        scores[te] = m.decision_function(F[te])
    return scores


def auc(y, s):
    ok = ~np.isnan(s)
    return float(roc_auc_score(y[ok], s[ok])) if len(set(y[ok])) == 2 else float("nan")


def shuffled(y, groups, rng):
    out = y.copy()
    g = np.array(groups)
    for s in np.unique(g):
        i = np.flatnonzero(g == s)
        out[i] = y[rng.permutation(i)]
    return out


def run(F, y, sp, C, perms, rng, sub):
    """AUROC (averaged over 5 fold assignments) on subset sub, and a within-species permutation p."""
    a = float(np.mean([auc(y[sub], group_scores(F, y, sp, C, seed=s)[sub]) for s in range(5)]))
    null = []
    for _ in range(perms):
        yp = shuffled(y, sp, rng)
        null.append(auc(yp[sub], group_scores(F, yp, sp, C)[sub]))
    p = (1 + sum(n >= a for n in null)) / (1 + len(null)) if null else None
    return {"auroc": round(a, 3), "p": None if p is None else round(p, 3)}


def scope(M, ev, P, rec, kind, perms, rng) -> dict:
    uids = [m["uid"] for m in M]
    y = np.array([m["group"] != "sonar" for m in M], int)
    sp = [m["taxon"] for m in M]
    key = lambda m: (m["taxon"], m["locality"], m["method"])
    exact = {key(m) for m in M if m["group"] != "sonar"} & {key(m) for m in M if m["group"] == "sonar"}
    sub = np.array([key(m) in exact for m in M])
    both = {s for s in sp if {m["group"] != "sonar" for m in M if m["taxon"] == s} == {True, False}}
    print(f"{len(M)} recordings: {y.sum()} talking, {(1 - y).sum()} sonar, {len(set(sp))} species, "
          f"{len(both)} species with both, exact-match subset {sub.sum()} ({y[sub].sum()} talking)", flush=True)

    def mean_of(k):
        return np.stack([P[(rec == u) & (kind == k)].mean(axis=0) if ((rec == u) & (kind == k)).any() else np.full(P.shape[1], np.nan) for u in uids])
    feats = {"shape": (np.array([shape_row(ev["events"][u], ev["seconds"][u]) for u in uids], float), 0.1),
             "perch": (mean_of("call"), 0.001), "quiet": (mean_of("quiet"), 0.001)}
    out = {"n": len(M), "talking": int(y.sum()), "species": len(set(sp)), "species_with_both": len(both),
           "exact_subset": int(sub.sum()), "exact_talking": int(y[sub].sum())}
    everything = np.ones(len(y), bool)
    for fname, (F, C) in feats.items():
        out[fname] = {"all": run(F, y, sp, C, perms, rng, everything)}
        if len(set(y[sub])) == 2:
            out[fname]["exact"] = run(F, y, sp, C, perms, rng, sub)
        print(fname, out[fname], flush=True)
    return out


def main(name: str = "SEA_context", perms: int = 200) -> None:
    w = ROOT / "data" / "windows"
    manifest = json.loads((ROOT / "data" / "dataset" / name / "manifest.json").read_text(encoding="utf8"))
    ev = json.loads((w / f"{name}_events.json").read_text(encoding="utf8"))
    d = np.load(w / f"{name}.npz")
    rec, kind = d["rec"], d["kind"]
    P = np.load(w / f"{name}_perch.npy")
    M = [m for m in manifest if m["uid"] in ev["events"] and m["group"] in ("distress", "social", "sonar")]
    rng = np.random.default_rng(1)
    out = {}
    print("-- one device,", DEVICE)
    out["one_device"] = scope([m for m in M if m["device"] == DEVICE], ev, P, rec, kind, perms, rng)
    print("-- all devices (confounded, for comparison, no permutations)")
    out["all_devices"] = scope(M, ev, P, rec, kind, 0, rng)
    (ROOT / "results" / f"social_test_{name}.json").write_text(json.dumps(out, indent=1), encoding="utf8")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "SEA_context", int(a[1]) if len(a) > 1 else 200)
