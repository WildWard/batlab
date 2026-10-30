"""Do distress calls fall into call types that repeat across species?

Takes loud, long events (not sonar pulses) from every talking recording, describes each by its shape,
and clusters them. Three checks, since clustering always returns something:
  discreteness   silhouette against data with the same spread but no joint structure (each principal
                 component shuffled on its own)
  prediction strength   cluster two species-disjoint halves; do the same calls group together in the
                 half that wasn't used to fit the clusters?
  spread         how many genera and devices each type spans
Usage: python tools/social_types.py [name]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
MIN_SNR, MIN_MS = 30, 8     # sonar pulses are mostly shorter than this
PER_REC = 40                # no single recording dominates a type
KS = range(2, 11)


def row(e: dict) -> list[float]:
    return [np.log(e["dur_ms"]), np.log(max(e["f_peak"], 8)), np.log1p(e["range"]), np.log1p(e["turns"]), e["harm"], e["flat"], np.sign(e["rise"]) * np.log1p(abs(e["rise"]))]


def pc_shuffled(X, rng):
    p = PCA().fit(X)
    Z = p.transform(X)
    for j in range(Z.shape[1]):
        Z[:, j] = rng.permutation(Z[:, j])
    return p.inverse_transform(Z)


def sil(X, k, seed=0):
    return silhouette_score(X, KMeans(k, n_init=5, random_state=seed).fit_predict(X), sample_size=1500, random_state=seed)


def prediction_strength(X, sp, k, rng, repeats=20):
    names = np.unique(sp)
    out = []
    for _ in range(repeats):
        a = rng.permutation(names)[: len(names) // 2]
        A, B = np.isin(sp, a), ~np.isin(sp, a)
        fit, own = KMeans(k, n_init=5, random_state=0).fit(X[A]), KMeans(k, n_init=5, random_state=0).fit_predict(X[B])
        cross = fit.predict(X[B])
        worst = 1.0
        for c in range(k):   # of pairs in one of B's own clusters, the share the A clusters also keep together
            i = np.flatnonzero(own == c)
            if len(i) < 2:
                continue
            counts = np.bincount(cross[i], minlength=k)
            worst = min(worst, float(((counts * (counts - 1)).sum()) / (len(i) * (len(i) - 1))))
        out.append(worst)
    return float(np.mean(out))


def main(name: str = "SEA_context") -> None:
    w = ROOT / "data" / "windows"
    manifest = json.loads((ROOT / "data" / "dataset" / name / "manifest.json").read_text(encoding="utf8"))
    ev = json.loads((w / f"{name}_events.json").read_text(encoding="utf8"))["events"]
    rng = np.random.default_rng(3)
    rows, meta = [], []
    for m in manifest:
        if m["group"] not in ("distress", "social"):
            continue
        cand = [e for e in ev[m["uid"]] if e["snr"] >= MIN_SNR and e["dur_ms"] >= MIN_MS]
        cand = sorted(cand, key=lambda e: -e["snr"])[:PER_REC]
        for e in cand:
            rows.append(row(e))
            meta.append({"uid": m["uid"], "taxon": m["taxon"], "genus": m["genus"], "device": m["device"], "file": m["file"], "rate": m.get("samplerate_khz", ""), "t": e["t"], "dur_ms": e["dur_ms"]})
    X = StandardScaler().fit_transform(np.array(rows))
    sp = np.array([x["taxon"] for x in meta])
    print(f"{len(X)} events, {len(set(x['uid'] for x in meta))} recordings, {len(set(sp))} species, {len({x['genus'] for x in meta})} genera")
    table = []
    for k in KS:
        real = sil(X, k)
        null = [sil(pc_shuffled(X, rng), k, seed=s) for s in range(10)]
        ps = prediction_strength(X, sp, k, rng)
        table.append({"k": k, "silhouette": round(float(real), 3), "null_mean": round(float(np.mean(null)), 3), "null_max": round(float(np.max(null)), 3), "prediction_strength": round(ps, 3)})
        print(table[-1], flush=True)
    best = max(table, key=lambda r: (r["prediction_strength"] >= 0.6, r["k"] if r["prediction_strength"] >= 0.6 else 0, r["silhouette"]))
    k = best["k"]
    labels = KMeans(k, n_init=20, random_state=0).fit_predict(X)
    raw = np.array(rows)
    types = []
    for c in range(k):
        i = labels == c
        g, d = Counter(x["genus"] for x, q in zip(meta, i) if q), Counter(x["device"] for x, q in zip(meta, i) if q)
        types.append({"type": c, "events": int(i.sum()), "species": len({x["taxon"] for x, q in zip(meta, i) if q}), "genera": len(g), "devices": dict(d),
                      "dur_ms": round(float(np.exp(np.median(raw[i, 0]))), 1), "f_peak_khz": round(float(np.exp(np.median(raw[i, 1]))), 1),
                      "range_khz": round(float(np.expm1(np.median(raw[i, 2]))), 1), "turns": round(float(np.expm1(np.median(raw[i, 3]))), 1),
                      "harm": round(float(np.median(raw[i, 4])), 2), "flat": round(float(np.median(raw[i, 5])), 2)})
        print(types[-1], flush=True)
    (ROOT / "results" / f"social_types_{name}.json").write_text(json.dumps({"events": len(X), "scan": table, "chosen_k": k, "types": types}, indent=1), encoding="utf8")
    (w / f"{name}_types.json").write_text(json.dumps([m | {"type": int(c)} for m, c in zip(meta, labels)]), encoding="utf8")


if __name__ == "__main__":
    main(*sys.argv[1:2])
