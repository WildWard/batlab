"""Test 13: does the recording method (hand release, hand held, enclosure, free flight) change what the classifier hears?

ChiroVox notes how each call was recorded. Researchers warn that a bat calls differently in the hand,
in a flight tent or flying free, so a library built from one method may not fit surveys made another way.
Three checks, all at unseen sites (5-fold by locality, as in site_eval.py):
  A. accuracy split by the test recording's method
  B. for each method, train without any recordings of that method, against a control that drops the same
     number of recordings of the same species from other methods; the gap is the cost of a method missing
  C. can the method itself be read from the sound, inside one device model (so the microphone can't explain it)
Usage: python tools/method_eval.py [set]
"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
METHODS = {"hand release": "hand release", "hand held": "hand held", "enclosure": "enclosure", "free fly": "free flight"}
SHUFFLES = 3


def raw_meta() -> dict:
    out = {}
    for f in glob.glob(str(ROOT / "raw" / "*.json")) + glob.glob(str(ROOT / "raw" / "europe" / "*.json")):
        d = json.loads(Path(f).read_text(encoding="utf8"))
        for r in d if isinstance(d, list) else []:
            out[r["chirovox"]] = r
    return out


def load(name: str):
    d = np.load(ROOT / "data" / "windows" / f"{name}.npz")
    perch = np.load(ROOT / "data" / "windows" / f"{name}_perch.npy")
    manifest = {m["uid"]: m for m in json.loads((ROOT / "data" / "dataset" / name / "manifest.json").read_text(encoding="utf8"))}
    raw = raw_meta()
    method = np.array([METHODS.get(raw[u]["method"].lower(), "other") for u in d["rec"]])
    site = np.array([f"{manifest[u]['country']}:{manifest[u]['locality']}" for u in d["rec"]])
    device = np.array([raw[u]["device_model"] for u in d["rec"]])
    return dict(F=perch, species=d["species"], rec=d["rec"], site=site, method=method, device=device)


def fit(F, y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=0.5, class_weight="balanced"))
    return clf.fit(F, y)


def score_recs(clf, F, y, rec) -> list[tuple[str, bool]]:
    """One answer per recording: mean probability over its windows. Skips species the model never saw."""
    proba, known, out = clf.predict_proba(F), set(clf.classes_), []
    for r in np.unique(rec):
        i = rec == r
        if y[i][0] in known:
            out.append((r, clf.classes_[proba[i].mean(axis=0).argmax()] == y[i][0]))
    return out


def check_a(d) -> dict:
    hits = {}
    for tr, te in GroupKFold(n_splits=5).split(d["F"], d["species"], d["site"]):
        clf = fit(d["F"][tr], d["species"][tr])
        for r, ok in score_recs(clf, d["F"][te], d["species"][te], d["rec"][te]):
            hits.setdefault(d["method"][d["rec"] == r][0], []).append(ok)
    return {m: (float(np.mean(v)), len(v)) for m, v in sorted(hits.items()) if m != "other"}


def check_b(d) -> dict:
    """Paired: the same test recordings, scored by a model that lacks their method vs one that lacks as many others."""
    res = {m: {"without": [], "control": []} for m in METHODS.values()}
    rng = np.random.default_rng(13)
    for shuffle in range(SHUFFLES):
        groups = rng.permutation(np.unique(d["site"]))   # new site-to-fold assignment each shuffle
        gid = {s: k for k, s in enumerate(groups)}
        g = np.array([gid[s] for s in d["site"]])
        for tr, te in GroupKFold(n_splits=5).split(d["F"], d["species"], g % 5):
            for m in res:
                te_m = te[d["method"][te] == m]
                if len(te_m) == 0:
                    continue
                same = tr[d["method"][tr] == m]
                others = tr[d["method"][tr] != m]
                if len(same) == 0:
                    continue
                # The control drops as many recordings of each species as the method removal does, taken from that
                # species' other methods. A plain random control drops far fewer of the test species (a species is
                # often recorded one way), which would blame the method for what is really less data per species.
                rec_sp = dict(zip(d["rec"][tr], d["species"][tr]))
                same_recs = np.unique(d["rec"][same])
                other_recs = np.unique(d["rec"][others])
                drop = set()
                for s, n in Counter(rec_sp[r] for r in same_recs).items():
                    pool = [r for r in other_recs if rec_sp[r] == s]
                    drop |= set(rng.choice(pool, size=min(n, len(pool)), replace=False)) if pool else set()
                tr_without = others
                tr_control = tr[~np.isin(d["rec"][tr], list(drop))]
                a = dict(score_recs(fit(d["F"][tr_without], d["species"][tr_without]), d["F"][te_m], d["species"][te_m], d["rec"][te_m]))
                b = dict(score_recs(fit(d["F"][tr_control], d["species"][tr_control]), d["F"][te_m], d["species"][te_m], d["rec"][te_m]))
                both = a.keys() & b.keys()   # only recordings both models could answer
                res[m]["without"] += [a[r] for r in both]
                res[m]["control"] += [b[r] for r in both]
    return {m: {k: float(np.mean(v)) for k, v in x.items()} | {"n": len(x["without"])} for m, x in res.items() if x["without"]}


def check_c(d) -> dict:
    out = {}
    for dev, n in Counter(d["device"]).most_common():
        i = (d["device"] == dev) & (d["method"] != "other")
        kinds = Counter(d["method"][i])
        if len([k for k, c in kinds.items() if c >= 20]) < 2 or len(np.unique(d["site"][i])) < 5:
            continue
        keep = i & np.isin(d["method"], [k for k, c in kinds.items() if c >= 20])
        F, y, rec, site = d["F"][keep], d["method"][keep], d["rec"][keep], d["site"][keep]
        hits = []
        for tr, te in GroupKFold(n_splits=5).split(F, y, site):
            if len(np.unique(y[tr])) < 2:
                continue
            hits += [ok for _, ok in score_recs(fit(F[tr], y[tr]), F[te], y[te], rec[te])]
        per_rec = Counter(y[np.unique(rec, return_index=True)[1]])
        out[dev] = dict(accuracy=float(np.mean(hits)), n=len(hits), majority=max(per_rec.values()) / sum(per_rec.values()),
                        methods=dict(per_rec), sites=int(len(np.unique(site))))
    return out


def main(name: str = "SEA_wide_clean") -> None:
    d = load(name)
    print(f"{len(np.unique(d['rec']))} recordings, {len(np.unique(d['site']))} sites, {len(np.unique(d['species']))} species")
    print("methods:", dict(Counter(d["method"][np.unique(d["rec"], return_index=True)[1]])))
    a = check_a(d)
    print("\nA. unseen-site species accuracy by the test recording's method")
    for m, (acc, n) in a.items():
        print(f"  {m:13s} {acc:5.0%}  ({n} recordings)")
    b = check_b(d)
    print(f"\nB. library missing the test method vs control ({SHUFFLES} shuffles)")
    for m, x in b.items():
        print(f"  {m:13s} without {x['without']:5.0%}  control {x['control']:5.0%}  gap {x['control'] - x['without']:+.0%}  ({x['n']} answers)")
    c = check_c(d)
    print("\nC. method read from the sound, one device model at a time, unseen sites")
    for dev, x in c.items():
        print(f"  {dev:24s} {x['accuracy']:5.0%} vs majority {x['majority']:4.0%}  ({x['n']} recs, {x['sites']} sites, {x['methods']})")
    out = ROOT / "results" / f"method_eval_{name}.json"
    out.write_text(json.dumps(dict(A=a, B=b, C=c), indent=1), encoding="utf8")


if __name__ == "__main__":
    main(*sys.argv[1:])
