"""Controlled test: does training on more sites help on a new site?

For species recorded at 5 or more sites: repeatedly hold out one site per species as the test,
then train on recordings from k other sites (k = 1, 2, 3, 4) of the same species. Same species,
same test recordings, only the number of training sites changes. Reports mean accuracy with a
95% bootstrap confidence interval over repeats.
Usage: python tools/site_curve.py [name] [repeats]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent


def main(name: str = "SEA", repeats: int = 30, min_sites: int = 5) -> None:
    d = np.load(ROOT / "data" / "windows" / f"{name}.npz")
    P = np.load(ROOT / "data" / "windows" / f"{name}_perch.npy")
    man = {m["uid"]: m for m in json.loads((ROOT / "data" / "dataset" / name / "manifest.json").read_text(encoding="utf8"))}
    rec, sp = d["rec"], d["species"]
    site = np.array([man[u]["locality"] for u in rec])
    # recording-level vectors (mean of windows), as the app does per pass
    recs = sorted(set(rec))
    X = np.stack([P[rec == r].mean(axis=0) for r in recs])
    y = np.array([sp[rec == r][0] for r in recs])
    s = np.array([site[rec == r][0] for r in recs])
    sites_of = defaultdict(list)
    for i, (yy, ss) in enumerate(zip(y, s)):
        sites_of[yy].append((ss, i))
    keep = [k for k, v in sites_of.items() if len({ss for ss, _ in v}) >= min_sites]
    print(f"{len(keep)} species with >= {min_sites} sites: {', '.join(sorted(keep))}")
    rng = np.random.default_rng(0)
    ks = [1, 2, 3, 4]
    results = {k: [] for k in ks}
    for _ in range(repeats):
        test_idx, pools = [], {}
        for spc in keep:
            by_site = defaultdict(list)
            for ss, i in sites_of[spc]:
                by_site[ss].append(i)
            order = list(by_site)
            rng.shuffle(order)
            test_idx += by_site[order[0]]          # one unseen site per species
            pools[spc] = [by_site[o] for o in order[1:]]
        for k in ks:
            # equalise the number of training recordings per species across k, so only site variety changes
            train_idx = []
            for spc in keep:
                chosen = [i for site_recs in pools[spc][:k] for i in site_recs]
                train_idx += list(rng.choice(chosen, size=min(4, len(chosen)), replace=len(chosen) < 4))
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
            clf.fit(X[train_idx], y[train_idx])
            results[k].append(float((clf.predict(X[test_idx]) == y[test_idx]).mean()))
    print("training sites | accuracy on a new site (mean, 95% interval over repeats)")
    out = {}
    for k in ks:
        a = np.array(results[k])
        lo, hi = np.percentile(a, [2.5, 97.5])
        out[k] = {"mean": float(a.mean()), "lo": float(lo), "hi": float(hi)}
        print(f"  {k}              | {a.mean():.0%}  ({lo:.0%} to {hi:.0%})")
    # paired: same held-out sites and draws, 4 training sites vs 1
    diff = np.array(results[4]) - np.array(results[1])
    boot = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    better = float((diff > 0).mean())
    out["paired_4_vs_1"] = {"mean_gain": float(diff.mean()), "lo": float(lo), "hi": float(hi), "share_better": better}
    print(f"paired, 4 sites vs 1 on the same test: +{diff.mean():.0%} (95% CI {lo:+.0%} to {hi:+.0%}), better in {better:.0%} of repeats")
    (ROOT / "results" / f"site_curve_{name}.json").write_text(json.dumps(out, indent=1), encoding="utf8")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "SEA", int(a[1]) if len(a) > 1 else 30)
