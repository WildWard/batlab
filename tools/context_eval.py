"""Test 15: does it help to TELL the classifier how a recording was made?

The ChiroVox team's idea (Zsebők Sándor): the user enters the recording circumstances and the program
uses them. Test 13 showed that the method changes the calls. This test asks the next question: if the
classifier knows the method of every reference call and of the recording it has to name, does it name
the species more often?

Three simple ways to use the method, all on top of the same Perch 2.0 + logistic regression, all at
unseen sites (5-fold by locality, 3 site shuffles), one answer per recording:
  centre   subtract the average embedding of the recording's method (learned on the training fold), so
           what all calls of one method share (room echo, distance, gain) is removed before classifying
  weight   when naming a recording of method m, reference calls of the same method count 3 times
  prior    the method is an extra input column; a linear model can only turn this into "which species
           are usual for this method", so it measures the species mix, not the sound
Control for each: the same procedure with the method labels shuffled between recordings. A gain that the
shuffled labels also give is not about the method.

Usage: python tools/context_eval.py [set ...]      (default: Europe SEA_wide_clean)
"""
import json
import sys
import time
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from method_eval import ROOT, load

SHUFFLES = 3
WEIGHT = 3.0
MIN_WINDOWS = 20      # a method needs this many training windows to get its own centre
VARIANTS = ["base", "centre", "weight", "prior", "centre_shuffled", "weight_shuffled", "prior_shuffled"]


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=0.5, class_weight="balanced"))


def answers(clf, F, y, rec) -> dict:
    """recording -> right or wrong (mean probability over its windows). Species the model never saw are skipped."""
    proba, known, out = clf.predict_proba(F), set(clf.classes_), {}
    for r in np.unique(rec):
        i = rec == r
        if y[i][0] in known:
            out[r] = bool(clf.classes_[proba[i].mean(axis=0).argmax()] == y[i][0])
    return out


def centred(F, method, tr):
    """Subtract each method's training-fold mean. Methods too rare in training use the overall training mean."""
    overall = F[tr].mean(axis=0)
    out = F - overall
    for m in np.unique(method):
        i_tr = tr[method[tr] == m]
        if len(i_tr) >= MIN_WINDOWS:
            out[method == m] = F[method == m] - F[i_tr].mean(axis=0)
    return out


def one_fold(F, y, rec, method, tr, te) -> dict:
    res = {}
    # centre
    Fc = centred(F, method, tr)
    res["centre"] = answers(model().fit(Fc[tr], y[tr]), Fc[te], y[te], rec[te])
    # weight: one model per method present in the test fold
    out = {}
    for m in np.unique(method[te]):
        te_m = te[method[te] == m]
        w = np.where(method[tr] == m, WEIGHT, 1.0)
        clf = model().fit(F[tr], y[tr], logisticregression__sample_weight=w)
        out.update(answers(clf, F[te_m], y[te_m], rec[te_m]))
    res["weight"] = out
    # prior: one-hot method columns next to the embedding
    kinds = np.unique(method)
    H = (method[:, None] == kinds[None, :]).astype(F.dtype)
    Fp = np.hstack([F, H])
    res["prior"] = answers(model().fit(Fp[tr], y[tr]), Fp[te], y[te], rec[te])
    return res


def run(name: str) -> dict:
    d = load(name)
    F, y, rec, site, method = d["F"], d["species"], d["rec"], d["site"], d["method"]
    recs, first = np.unique(rec, return_index=True)
    rec_method = dict(zip(recs, method[first]))
    print(f"\n=== {name}: {len(recs)} recordings, {len(np.unique(site))} sites, {len(np.unique(y))} species")
    print("methods:", dict(Counter(rec_method.values())))
    rng = np.random.default_rng(15)
    hits = {v: {} for v in VARIANTS}          # variant -> recording -> list of right/wrong over shuffles
    t0 = time.time()
    for shuffle in range(SHUFFLES):
        gid = {s: k for k, s in enumerate(rng.permutation(np.unique(site)))}
        g = np.array([gid[s] for s in site]) % 5
        fake_of = dict(zip(recs, rng.permutation(np.array([rec_method[r] for r in recs]))))   # labels shuffled between recordings
        fake = np.array([fake_of[r] for r in rec])
        for tr, te in GroupKFold(n_splits=5).split(F, y, g):
            base = answers(model().fit(F[tr], y[tr]), F[te], y[te], rec[te])
            real = one_fold(F, y, rec, method, tr, te)
            ctrl = one_fold(F, y, rec, fake, tr, te)
            for v, a in [("base", base)] + list(real.items()) + [(k + "_shuffled", x) for k, x in ctrl.items()]:
                for r, ok in a.items():
                    hits[v].setdefault(r, []).append(ok)
        print(f"  shuffle {shuffle + 1}/{SHUFFLES} done, {time.time() - t0:.0f} s", flush=True)

    common = sorted(set.intersection(*[set(h) for h in hits.values()]))
    per = {v: np.array([np.mean(hits[v][r]) for r in common]) for v in VARIANTS}
    m_of = np.array([rec_method[r] for r in common])
    boot = np.random.default_rng(1).integers(0, len(common), size=(2000, len(common)))

    def gain(v, mask=None):
        diff = per[v] - per["base"]
        if mask is not None:
            diff = diff[mask]
            idx = np.random.default_rng(1).integers(0, len(diff), size=(2000, len(diff)))
        else:
            idx = boot
        b = diff[idx].mean(axis=1)
        return float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    out = {"set": name, "recordings": len(common), "methods": dict(Counter(m_of.tolist())), "variants": {}}
    print(f"\nspecies right at unseen sites, {len(common)} recordings ({SHUFFLES} shuffles)")
    print(f"  {'variant':18s} {'all':>6s}  gain vs base (95% CI)")
    for v in VARIANTS:
        acc = float(per[v].mean())
        g_all = gain(v) if v != "base" else (0.0, 0.0, 0.0)
        out["variants"][v] = {"all": acc, "gain": g_all, "by_method": {}}
        print(f"  {v:18s} {acc:6.1%}  {g_all[0]:+6.1%} ({g_all[1]:+.1%} to {g_all[2]:+.1%})")
    print("\nby the test recording's method: accuracy, and gain vs base")
    kinds = [m for m, n in Counter(m_of.tolist()).most_common() if n >= 10]
    print(f"  {'variant':18s} " + " ".join(f"{m + ' (' + str(int((m_of == m).sum())) + ')':>24s}" for m in kinds))
    for v in VARIANTS:
        cells = []
        for m in kinds:
            mask = m_of == m
            acc = float(per[v][mask].mean())
            g_m = gain(v, mask) if v != "base" else (0.0, 0.0, 0.0)
            out["variants"][v]["by_method"][m] = {"n": int(mask.sum()), "acc": acc, "gain": g_m}
            cells.append(f"{acc:5.0%} {g_m[0]:+5.1%} ({g_m[1]:+.0%},{g_m[2]:+.0%})".rjust(24))
        print(f"  {v:18s} " + " ".join(cells))
    (ROOT / "results" / f"context_eval_{name}.json").write_text(json.dumps(out, indent=1), encoding="utf8")
    return out


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["Europe", "SEA_wide_clean"]):
        run(s)
