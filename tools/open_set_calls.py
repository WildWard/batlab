"""Open set on measured call parameters: does a species we never trained on stand out by its calls?

Leave-one-species-out. For every known-species recording, the known-species statistics are
rebuilt without that recording's site (so "known" means a known species at a new site, the hard
case from Test 5). The hidden species' recordings are the unknowns.

Scores (higher = more unfamiliar):
  fchar     - distance in characteristic frequency to the nearest known species, in units of that
              species' spread (robust: median and MAD, floor 1.5 kHz or 2% of the frequency)
  calls     - same idea in 4 dimensions: characteristic and start frequency, duration, bandwidth
              (log scale), nearest known species by standardised distance
  perch     - cosine distance to the nearest known species' mean Perch embedding
Run separately for constant-frequency bats (horseshoe and leaf-nosed: Rhinolophus, Hipposideros,
Aselliscus) and for all species. Only species with 5+ recordings are hidden.
Usage: python tools/open_set_calls.py [name]
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
CF_GENERA = {"Rhinolophus", "Hipposideros", "Aselliscus", "Coelops"}


def robust(v):
    v = np.asarray(v, float)
    med = np.median(v, axis=0)
    mad = 1.4826 * np.median(np.abs(v - med), axis=0)
    return med, mad


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def evaluate(rows, P):
    taxa = np.array([r["taxon"] for r in rows])
    site = np.array([r["site"] for r in rows])
    fchar = np.array([r["f_char_khz"] for r in rows], float)
    M = np.column_stack([np.log(np.clip([r[k] for r in rows], 0.5, None)) for k in
                         ("f_char_khz", "f_start_khz", "duration_ms", "bandwidth_khz")])
    U = unit(P)
    counts = {t: (taxa == t).sum() for t in set(taxa)}
    hidden_set = sorted(t for t, n in counts.items() if n >= 5)
    out = {k: ([], []) for k in ("fchar", "calls", "perch")}   # (scores, is_unknown)

    def score(i, ref):
        """Unfamiliarity of recording i against reference recordings ref (bool mask)."""
        s = {"fchar": np.inf, "calls": np.inf, "perch": np.inf}
        for t in set(taxa[ref]):
            m = ref & (taxa == t)
            if m.sum() < 2:
                continue
            med, mad = robust(fchar[m])
            s["fchar"] = min(s["fchar"], abs(fchar[i] - med) / max(mad, 1.5, 0.02 * med))
            mv, madv = robust(M[m])
            s["calls"] = min(s["calls"], float(np.sqrt((((M[i] - mv) / np.maximum(madv, 0.05)) ** 2).mean())))
            c = unit(U[m].mean(axis=0, keepdims=True))[0]
            s["perch"] = min(s["perch"], 1 - float(U[i] @ c))
        return s

    for hidden in hidden_set:
        known = taxa != hidden
        for i in np.where(~known)[0]:                       # unknown species, full known library
            for k, v in score(i, known).items():
                out[k][0].append(v); out[k][1].append(1)
        for i in np.where(known & np.isin(taxa, [hidden], invert=True))[0]:
            if counts[taxa[i]] < 5:
                continue
            ref = known & (site != site[i])                  # known species, but a site never seen
            if (ref & (taxa == taxa[i])).sum() < 2:
                continue
            for k, v in score(i, ref).items():
                out[k][0].append(v); out[k][1].append(0)
    res = {}
    for k, (s, u) in out.items():
        s, u = np.array(s), np.array(u, bool)
        fin = np.isfinite(s)
        auc = roc_auc_score(u[fin], s[fin])
        thr = np.quantile(s[u & fin], 0.5)                   # flag half of the unknowns
        res[k] = {"auroc": round(float(auc), 3), "known_flagged_at_50pct": round(float((s[~u & fin] >= thr).mean()), 3),
                  "n_unknown": int(u.sum()), "n_known": int((~u).sum())}
    return res, len(hidden_set)


def main(name: str = "SEA_full") -> None:
    rows = json.loads((ROOT / "data" / "windows" / f"{name}_calls.json").read_text(encoding="utf8"))
    d = np.load(ROOT / "data" / "windows" / f"{name}.npz")
    P = np.load(ROOT / "data" / "windows" / f"{name}_perch.npy")
    emb = {r: P[d["rec"] == r].mean(axis=0) for r in set(d["rec"])}
    rows = [r for r in rows if r["uid"] in emb and r.get("n_calls", 0) >= 3]
    results = {}
    for label, keep in (("constant-frequency bats", lambda r: r["genus"] in CF_GENERA), ("all species", lambda r: True)):
        sub = [r for r in rows if keep(r)]
        res, n_hidden = evaluate(sub, np.stack([emb[r["uid"]] for r in sub]))
        results[label] = res
        print(f"{label}: {len(sub)} recordings, {n_hidden} species hidden in turn")
        for k, v in res.items():
            print(f"  {k:6s} AUROC {v['auroc']:.2f} | flag half the unknowns -> {v['known_flagged_at_50pct']:.0%} of known flagged"
                  f"  ({v['n_unknown']} unknown, {v['n_known']} known tests)")
    results["per_species_cf"] = per_species_cf([r for r in rows if r["genus"] in CF_GENERA])
    out = ROOT / "results" / f"open_set_calls_{name}.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf8")


def per_species_cf(rows):
    """Characteristic-frequency score, one hidden constant-frequency species at a time, against how
    far its median frequency sits from the nearest other species in the library."""
    taxa = np.array([r["taxon"] for r in rows])
    site = np.array([r["site"] for r in rows])
    f = np.array([r["f_char_khz"] for r in rows], float)
    counts = {t: (taxa == t).sum() for t in set(taxa)}
    stats = {}

    def score(i, ref):
        best = np.inf
        for t in set(taxa[ref]):
            m = ref & (taxa == t)
            if m.sum() >= 2:
                med, mad = robust(f[m])
                best = min(best, abs(f[i] - med) / max(mad, 1.5, 0.02 * med))
        return best

    known = np.array([score(i, site != site[i]) for i in range(len(rows))
                      if counts[taxa[i]] >= 5 and ((site != site[i]) & (taxa == taxa[i])).sum() >= 2])
    med = {t: np.median(f[taxa == t]) for t in counts}
    for h in sorted(t for t in counts if counts[t] >= 5):
        unk = np.array([score(i, taxa != h) for i in np.where(taxa == h)[0]])
        gap = min(abs(med[h] - med[t]) / med[h] for t in med if t != h)
        auc = roc_auc_score(np.r_[np.zeros(len(known)), np.ones(len(unk))], np.r_[known, unk])
        stats[h] = {"nearest_species_gap": round(float(gap), 4), "auroc": round(float(auc), 3)}
    gaps = np.array([v["nearest_species_gap"] for v in stats.values()])
    aucs = np.array([v["auroc"] for v in stats.values()])
    rng = np.random.default_rng(0)
    ci = np.percentile([rng.choice(aucs, len(aucs)).mean() for _ in range(5000)], [2.5, 97.5])
    print(f"per species (constant-frequency): mean AUROC {aucs.mean():.2f} (95% CI {ci[0]:.2f}-{ci[1]:.2f});"
          f" {(gaps >= .05).sum()} species >= 5% from any other: {aucs[gaps >= .05].mean():.2f};"
          f" {(gaps < .05).sum()} closer: {aucs[gaps < .05].mean():.2f}")
    return stats


if __name__ == "__main__":
    main(*sys.argv[1:2])
