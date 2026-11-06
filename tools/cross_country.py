"""Test 14: Bulgaria to Hungary. Does a classifier trained on one country's recordings work in another?

Hungary in ChiroVox is nearly all one recordist on a Pettersson D500X, mostly free-flying bats; Bulgaria is three
teams on Anabat Swift, M500 and AudioMoth. So a Bulgaria-trained model meets a new country, new sites, a new
detector and new hands at once: close to what a new user of a shared tool would face.
  1. train on Bulgaria, test on Hungary (species both have)
  2. Hungary alone, unseen sites (0.1 degree cells, 5-fold): the ceiling with home data
  3. Hungary unseen sites, with Bulgaria added to training: does foreign data help or hurt
Scores per recording (mean probability over its windows), Perch 2.0 + logistic regression as in earlier tests.
Usage: python tools/cross_country.py [set]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CF_GENERA = {"Rhinolophus"}   # constant-frequency callers; everything else here is FM or QCF


def load(name: str):
    d = np.load(ROOT / "data" / "windows" / f"{name}.npz")
    perch = np.load(ROOT / "data" / "windows" / f"{name}_perch.npy")
    man = {m["uid"]: m for m in json.loads((ROOT / "data" / "dataset" / name / "manifest.json").read_text(encoding="utf8"))}
    col = lambda k: np.array([man[u][k] for u in d["rec"]])
    return dict(F=perch, species=d["species"], genus=d["genus"], rec=d["rec"], country=col("country"),
                site=col("locality"), method=col("method"), device=col("device"))


def fit(F, y):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=0.5, class_weight="balanced")).fit(F, y)


def answers(clf, F, y, g, rec) -> list[dict]:
    proba, out = clf.predict_proba(F), []
    for r in np.unique(rec):
        i = rec == r
        p = proba[i].mean(axis=0)
        guess = clf.classes_[p.argmax()]
        out.append(dict(rec=r, truth=y[i][0], guess=guess, conf=float(p.max()),
                        species_ok=guess == y[i][0], genus_ok=guess.split()[0] == g[i][0]))
    return out


def summary(ans: list[dict], label: str) -> dict:
    if not ans:
        return {}
    s = dict(n=len(ans), species=float(np.mean([a["species_ok"] for a in ans])), genus=float(np.mean([a["genus_ok"] for a in ans])),
             species_n=len({a["truth"] for a in ans}))
    sure = [a for a in ans if a["conf"] >= 0.7]
    s["sure_share"], s["sure_right"] = len(sure) / len(ans), (float(np.mean([a["species_ok"] for a in sure])) if sure else None)
    print(f"  {label:40s} species {s['species']:4.0%}  genus {s['genus']:4.0%}  ({s['n']} recordings, {s['species_n']} species)"
          f"  70%+ sure: {s['sure_share']:.0%} of answers, {s['sure_right'] or 0:.0%} right", flush=True)
    return s


def split_by(ans, d, key_fn) -> dict:
    groups = defaultdict(list)
    for a in ans:
        groups[key_fn(a)].append(a)
    return groups


def main(name: str = "Europe") -> None:
    d = load(name)
    hu, bg = d["country"] == "Hungary", d["country"] == "Bulgaria"
    shared = set(d["species"][hu]) & set(d["species"][bg])
    first = np.unique(d["rec"], return_index=True)[1]
    print(f"{len(first)} recordings: Hungary {len(np.unique(d['rec'][hu]))}, Bulgaria {len(np.unique(d['rec'][bg]))}, "
          f"{len(shared)} shared species; Hungary sites {len(np.unique(d['site'][hu]))}, Bulgaria sites {len(np.unique(d['site'][bg]))}")
    rec_method = dict(zip(d["rec"][first], d["method"][first]))
    res = {}

    print("\n1. Train on Bulgaria, test on Hungary (shared species only)")
    tr, te = bg & np.isin(d["species"], list(shared)), hu & np.isin(d["species"], list(shared))
    a1 = answers(fit(d["F"][tr], d["species"][tr]), d["F"][te], d["species"][te], d["genus"][te], d["rec"][te])
    res["bg_to_hu"] = summary(a1, "all")
    for k, v in sorted(split_by(a1, d, lambda a: "CF (Rhinolophus)" if a["truth"].split()[0] in CF_GENERA else "FM / QCF").items()):
        res[f"bg_to_hu/{k}"] = summary(v, k)
    for k, v in sorted(split_by(a1, d, lambda a: rec_method[a["rec"]]).items()):
        if len(v) >= 10:
            res[f"bg_to_hu/{k}"] = summary(v, f"recorded {k}")

    print("\n2. Hungary only, unseen sites")
    print("\n3. Hungary unseen sites, Bulgaria added to training")
    a2, a3 = [], []
    idx_hu = np.where(hu)[0]
    for ftr, fte in GroupKFold(n_splits=5).split(idx_hu, groups=d["site"][idx_hu]):
        tr_hu, te_hu = idx_hu[ftr], idx_hu[fte]
        a2 += answers(fit(d["F"][tr_hu], d["species"][tr_hu]), d["F"][te_hu], d["species"][te_hu], d["genus"][te_hu], d["rec"][te_hu])
        tr_both = np.concatenate([tr_hu, np.where(bg)[0]])
        a3 += answers(fit(d["F"][tr_both], d["species"][tr_both]), d["F"][te_hu], d["species"][te_hu], d["genus"][te_hu], d["rec"][te_hu])
    known = set(d["species"][hu])   # score only species the Hungary-only model could have known in some fold
    for label, a, key in (("Hungary only", a2, "hu_only"), ("Hungary + Bulgaria", a3, "hu_plus_bg")):
        res[key] = summary([x for x in a if x["truth"] in known], label)
        res[key + "/shared"] = summary([x for x in a if x["truth"] in shared], label + ", shared species")

    per = defaultdict(lambda: [0, 0])
    for a in a1:
        per[a["truth"]][0] += a["species_ok"]
        per[a["truth"]][1] += 1
    res["bg_to_hu_per_species"] = {s: f"{ok}/{n}" for s, (ok, n) in sorted(per.items())}
    res["bg_to_hu_confusions"] = Counter(f"{a['truth']} -> {a['guess']}" for a in a1 if not a["species_ok"]).most_common(10)
    print("\nper species (Bulgaria -> Hungary):", res["bg_to_hu_per_species"])
    print("top confusions:", res["bg_to_hu_confusions"])
    (ROOT / "results" / f"cross_country_{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main(*sys.argv[1:])
