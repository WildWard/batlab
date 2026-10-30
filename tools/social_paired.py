"""The fair version of social_test.py: compare talking and sonar inside matched groups.

Background alone (silence_control.py, gaps between calls) separates the labels at 0.87 across one device,
so that score is mostly setup. Here every recording is compared only with others of the same species,
site, method and device. Features are centred within each group, which removes what the group shares,
and a recording is scored by a model trained on other species only. All three feature sets use the same
recordings. The score is the share of talking-sonar pairs in a group where talking scores higher; 0.5 is
chance. p-values shuffle labels within each group. If gaps score as well as calls, the difference is setup.
Usage: python tools/social_paired.py [name] [permutations]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from social_test import DEVICE, shape_row

ROOT = Path(__file__).resolve().parent.parent


def centred(F, groups):
    out = F.copy()
    for g in set(groups):
        i = [k for k, x in enumerate(groups) if x == g]
        out[i] -= F[i].mean(axis=0)
    return out


def scores(F, y, sp, C):
    s = np.zeros(len(y))
    sp = np.array(sp)
    for held in set(sp):
        te, tr = sp == held, sp != held
        if len(set(y[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=2000, C=C, class_weight="balanced"))
        m.fit(F[tr], y[tr])
        s[te] = m.decision_function(F[te])
    return s


def paired(s, y, groups):
    wins, total = 0.0, 0
    for g in set(groups):
        i = [k for k, x in enumerate(groups) if x == g]
        for a in i:
            for b in i:
                if y[a] == 1 and y[b] == 0:
                    wins += (s[a] > s[b]) + 0.5 * (s[a] == s[b])
                    total += 1
    return wins / total


def shuffled(y, groups, rng):
    out = y.copy()
    for g in set(groups):
        i = np.array([k for k, x in enumerate(groups) if x == g])
        out[i] = y[rng.permutation(i)]
    return out


def main(name: str = "SEA_context", perms: int = 300) -> None:
    w = ROOT / "data" / "windows"
    manifest = json.loads((ROOT / "data" / "dataset" / name / "manifest.json").read_text(encoding="utf8"))
    raw = {}
    for f in (ROOT / "raw").glob("*.json"):
        d = json.loads(f.read_text(encoding="utf8"))
        raw.update({r["chirovox"]: r for r in d} if isinstance(d, list) else {})
    ev = json.loads((w / f"{name}_events.json").read_text(encoding="utf8"))
    d = np.load(w / f"{name}.npz")
    rec, kind, P = d["rec"], d["kind"], np.load(w / f"{name}_perch.npy")
    gap_rec, G = np.load(w / f"{name}_silent.npz")["rec"], np.load(w / f"{name}_silent_perch.npy")
    M = [m for m in manifest if m["device"] == DEVICE and m["group"] in ("distress", "social", "sonar") and m["uid"] in set(gap_rec)]
    key = lambda m: f'{m["taxon"]}|{m["locality"]}|{m["method"]}'
    has = defaultdict(set)
    for m in M:
        row = shape_row(ev["events"][m["uid"]], ev["seconds"][m["uid"]])
        if not np.isnan(row).any():
            has[key(m)].add(m["group"] != "sonar")
    M = [m for m in M if has[key(m)] == {True, False} and not np.isnan(shape_row(ev["events"][m["uid"]], ev["seconds"][m["uid"]])).any()]
    y = np.array([m["group"] != "sonar" for m in M], int)
    groups, sp = [key(m) for m in M], [m["taxon"] for m in M]
    same_day = sum(len({raw[m["uid"]]["date"] for m in M if key(m) == g}) == 1 for g in set(groups))
    print(f"{len(M)} recordings ({y.sum()} talking) in {len(set(groups))} groups, {len(set(sp))} species, {same_day} groups recorded on one date")
    print(Counter(m["taxon"] for m in M).most_common())
    feats = {
        "calls_shape": (np.array([shape_row(ev["events"][m["uid"]], ev["seconds"][m["uid"]]) for m in M], float), 0.1),
        "calls_perch": (np.stack([P[(rec == m["uid"]) & (kind == "call")].mean(axis=0) for m in M]), 0.001),
        "gaps_perch": (np.stack([G[gap_rec == m["uid"]].mean(axis=0) for m in M]), 0.001),
    }
    rng = np.random.default_rng(1)
    out = {"recordings": len(M), "talking": int(y.sum()), "groups": len(set(groups)), "species": len(set(sp)), "groups_one_date": same_day}
    for fname, (F, C) in feats.items():
        Fc = centred(F, groups)
        a = paired(scores(Fc, y, sp, C), y, groups)
        null = []
        for _ in range(perms):
            yp = shuffled(y, groups, rng)
            null.append(paired(scores(Fc, yp, sp, C), yp, groups))
        p = (1 + sum(n >= a for n in null)) / (1 + perms)
        out[fname] = {"paired": round(a, 3), "p": round(p, 3), "null_mean": round(float(np.mean(null)), 3)}
        print(fname, out[fname], flush=True)
    (ROOT / "results" / f"social_paired_{name}.json").write_text(json.dumps(out, indent=1), encoding="utf8")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "SEA_context", int(a[1]) if len(a) > 1 else 300)
