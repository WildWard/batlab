"""Test 16: which species can be trusted at a new site, and does "answer only when sure" help?

Two questions asked on the WILDLABS thread (Smilla Emneryd, 25 Nov 2026):
1. Per-species recall at recording sites the model never trained on.
2. A risk-coverage curve: if the tool only answers above a confidence level, how often is it right,
   and how many recordings does it still answer?

Two models, same recordings, same unseen-site splits (5 folds grouped by site, several shuffles):
- "perch":  the bare test model behind the published 63% (Perch windows, C=0.5, mean over windows)
- "listen": WildWard Listen's own classifier as rebuilt in Test 11 (Perch + measured call
            parameters x3, C=0.05), i.e. what a user of the app gets.
Only recordings whose species is in that fold's training data are scored (the rest is the open-set
problem, Tests 5 and 8). Nothing leaves batlab; the results hold species names and numbers only.
Usage: python tools/per_species.py [name] [repeats]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
LISTEN = ROOT.parent / "wildward-listen"
sys.path.insert(0, str(LISTEN))
from listen.calls import feature_vector  # noqa: E402

COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)


def load(name: str):
    d = np.load(ROOT / "data" / "windows" / f"{name}.npz")
    P = np.load(ROOT / "data" / "windows" / f"{name}_perch.npy")
    man = {m["uid"]: m for m in json.loads((ROOT / "data" / "dataset" / name / "manifest.json").read_text(encoding="utf8"))}
    calls = {r["uid"]: r for r in json.loads((ROOT / "data" / "windows" / f"{name}_calls.json").read_text(encoding="utf8"))}
    recs = np.array(sorted(set(d["rec"])))
    y = np.array([man[r]["taxon"] for r in recs])
    site = np.array([f"{man[r]['country']}/{man[r]['locality']}" for r in recs])
    emb = np.stack([P[d["rec"] == r].mean(axis=0) for r in recs])
    F = np.stack([feature_vector(calls.get(r) or {}) for r in recs])
    X_listen = np.hstack([emb, np.repeat(F, 3, axis=1)])      # exactly as Survey.suggest builds its features
    win_rec = np.searchsorted(recs, d["rec"])                  # window -> recording index
    return dict(recs=recs, y=y, site=site, X_listen=X_listen, P=P, win_rec=win_rec)


def splits(site: np.ndarray, repeats: int):
    """Unseen-site folds over recordings. Repeat 0 is the plain split, the others reshuffle the sites."""
    sites = np.unique(site)
    for rep in range(repeats):
        order = np.arange(len(sites)) if rep == 0 else np.random.default_rng(rep).permutation(len(sites))
        groups = np.array([dict(zip(sites, order))[s] for s in site])
        for tr, te in GroupKFold(n_splits=5).split(groups, groups, groups):
            yield rep, tr, te


def run(d, repeats: int):
    """One row per scored (recording, repeat): model, repeat, recording index, truth, guess, confidences."""
    rows = []
    y, P, win_rec = d["y"], d["P"], d["win_rec"]
    for rep, tr, te in splits(d["site"], repeats):
        tr_set = np.zeros(len(y), bool)
        tr_set[tr] = True
        # bare Perch model: trained and scored on windows, one answer per recording (mean over its windows)
        wtr = tr_set[win_rec]
        perch = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=0.5, class_weight="balanced"))
        perch.fit(P[wtr], y[win_rec[wtr]])
        listen = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                               LogisticRegression(max_iter=4000, C=0.05, class_weight="balanced"))
        listen.fit(d["X_listen"][tr], y[tr])
        for model, clf in (("perch", perch), ("listen", listen)):
            cls = clf.classes_
            genus_of = np.array([c.split()[0] for c in cls])
            genera = np.unique(genus_of)
            known = set(cls)
            for i in te:
                if y[i] not in known:
                    continue
                pr = (clf.predict_proba(P[win_rec == i]).mean(axis=0) if model == "perch"
                      else clf.predict_proba(d["X_listen"][i:i + 1])[0])
                gp = np.array([pr[genus_of == g].sum() for g in genera])
                rows.append((model, rep, int(i), y[i], cls[pr.argmax()], float(pr.max()),
                             genera[gp.argmax()], float(gp.max())))
        print(f"  repeat {rep}: {len(te)} test recordings in this fold", flush=True)
    return rows


def per_species(rows, model, sure: float):
    """sure = the confidence level above which the model answers half of all recordings."""
    by = defaultdict(lambda: dict(recs=set(), hits=[], genus_hits=[], wrong=Counter(), sure_hits=[]))
    for m, rep, i, truth, guess, conf, gguess, gconf in rows:
        if m != model:
            continue
        s = by[truth]
        s["recs"].add(i)
        s["hits"].append(guess == truth)
        s["genus_hits"].append(gguess == truth.split()[0])
        if conf >= sure:
            s["sure_hits"].append(guess == truth)
        if guess != truth:
            s["wrong"][guess] += 1
    out = []
    for sp, s in by.items():
        top = s["wrong"].most_common(1)
        out.append(dict(species=sp, recordings=len(s["recs"]), recall=float(np.mean(s["hits"])),
                        genus_recall=float(np.mean(s["genus_hits"])),
                        answered_when_sure=round(len(s["sure_hits"]) / len(s["hits"]), 2),
                        recall_when_sure=float(np.mean(s["sure_hits"])) if s["sure_hits"] else None,
                        mostly_mistaken_for=top[0][0] if top else "",
                        share_of_errors=round(top[0][1] / sum(s["wrong"].values()), 2) if top else 0.0))
    return sorted(out, key=lambda r: (r["recall"], r["species"]))


def risk_coverage(conf, ok):
    """Answer the most confident recordings first: accuracy among the answered share."""
    order = np.argsort(-conf)
    conf, ok = conf[order], ok[order]
    out = []
    for c in COVERAGES:
        n = max(1, int(round(c * len(ok))))
        out.append(dict(coverage=c, answered=n, accuracy=float(ok[:n].mean()), min_confidence=float(conf[n - 1])))
    return out


def curves(rows, model):
    r = [x for x in rows if x[0] == model]
    conf = np.array([x[5] for x in r]); ok = np.array([x[4] == x[3] for x in r], float)
    gconf = np.array([x[7] for x in r]); gok = np.array([x[6] == x[3].split()[0] for x in r], float)
    return dict(species=risk_coverage(conf, ok), genus=risk_coverage(gconf, gok),
                raw=dict(conf=conf, ok=ok, gconf=gconf, gok=gok))


def plot(all_curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, colour in (("listen", "#1f6f8b"), ("perch", "#c0652b")):
        raw = all_curves[model]["raw"]
        for conf, ok, style, level in ((raw["conf"], raw["ok"], "-", "species"), (raw["gconf"], raw["gok"], "--", "genus")):
            o = np.argsort(-conf)
            acc = np.cumsum(ok[o]) / np.arange(1, len(o) + 1)
            cov = np.arange(1, len(o) + 1) / len(o)
            keep = cov >= 0.05          # the first few answers are too noisy to draw
            ax.plot(100 * cov[keep], 100 * acc[keep], style, color=colour, label=f"{model}, {level}")
    ax.set_xlabel("share of recordings it answers (%), most confident first")
    ax.set_ylabel("right among the answered (%)")
    ax.set_title("Answer only when sure: unseen recording sites, 49 species")
    ax.set_xlim(100, 5); ax.set_ylim(50, 100); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=140)


def main(name: str = "SEA_full", repeats: int = 5) -> None:
    d = load(name)
    print(f"{len(d['recs'])} recordings, {len(np.unique(d['site']))} sites, {len(np.unique(d['y']))} species, {repeats} repeats")
    rows = run(d, repeats)
    out = {"set": name, "repeats": repeats,
           "note": "unseen recording sites, 5 folds grouped by site; recall is over all (recording, repeat) pairs"}
    all_curves = {}
    for model in ("perch", "listen"):
        r = [x for x in rows if x[0] == model]
        first = [x for x in r if x[1] == 0]
        overall = float(np.mean([x[4] == x[3] for x in r]))
        genus = float(np.mean([x[6] == x[3].split()[0] for x in r]))
        print(f"\n=== {model}: species {overall:.1%}, genus {genus:.1%} over {len(r)} answers "
              f"(plain split alone: {np.mean([x[4] == x[3] for x in first]):.1%} of {len(first)})")
        c = curves(rows, model)
        all_curves[model] = c
        sure = next(p["min_confidence"] for p in c["species"] if p["coverage"] == 0.5)
        table = per_species(rows, model, sure)
        print(f"  ('when sure' = shown confidence >= {sure:.2f}, the level that answers half of all recordings)")
        for t in table:
            ws = "   -" if t["recall_when_sure"] is None else f"{t['recall_when_sure']:4.0%}"
            print(f"  {t['species']:34s} n={t['recordings']:3d}  species {t['recall']:4.0%}  genus {t['genus_recall']:4.0%}"
                  f"  when sure: answers {t['answered_when_sure']:4.0%}, right {ws}"
                  f"  {('-> ' + t['mostly_mistaken_for']) if t['mostly_mistaken_for'] else ''}")
        for level in ("species", "genus"):
            print(f"  answer only when sure, {level}:")
            for p in c[level]:
                print(f"    answers {p['coverage']:4.0%} ({p['answered']:4d})  right {p['accuracy']:5.1%}  shown confidence >= {p['min_confidence']:.2f}")
        out[model] = dict(species_accuracy=overall, genus_accuracy=genus, answers=len(r), per_species=table,
                          risk_coverage=dict(species=c["species"], genus=c["genus"]))
    res = ROOT / "results"
    (res / f"per_species_{name}_rows.json").write_text(json.dumps(
        [dict(model=m, repeat=rep, rec=str(d["recs"][i]), truth=t, guess=g, conf=round(cf, 4), genus_guess=gg, genus_conf=round(gc, 4))
         for m, rep, i, t, g, cf, gg, gc in rows], ensure_ascii=False), encoding="utf8")
    (res / f"per_species_{name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf8")
    plot(all_curves, res / f"risk_coverage_{name}.png")
    print("\nwrote", res / f"per_species_{name}.json", "and the plot")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "SEA_full", int(a[1]) if len(a) > 1 else 5)
