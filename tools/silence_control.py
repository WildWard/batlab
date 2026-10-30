"""A stricter version of the quiet-window control in social_test.py.

The quietest windows of these 2 s recordings still hold calls (median peak 61 dB above the noise floor in
talking recordings, 42 in sonar), so a good score on them says nothing about the room. No 0.4 s stretch
is truly silent either. So this splices together only the gaps between calls, with everything loud and
13 ms around it removed, loops that to one Perch window, and scores it.
  python tools/silence_control.py cut    writes data/windows/<name>_silent.npz, then run embed_perch.py <name>_silent
  python tools/silence_control.py score  prints the scores, writes results/silence_control.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import ndimage, signal

from prepare_windows import SR, WIN, load
from social_test import DEVICE, auc, group_scores

ROOT = Path(__file__).resolve().parent.parent
NPERSEG = 1024
HOT_DB = 20         # noise alone peaks around 16 dB in this band
PAD_FRAMES = 5      # 13 ms either side of anything loud
MIN_GAP = 0.1       # seconds of calm audio a recording must have


def gap_clip(x: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Everything between the calls, spliced and looped to one Perch window, or None if under MIN_GAP."""
    f, _, S = signal.stft(x[: 120 * SR], fs=SR, nperseg=NPERSEG, noverlap=0, boundary=None, padded=False)
    band = (f >= 8_000) & (f <= 185_000)
    db = 20 * np.log10(np.abs(S[band]) + 1e-9)
    ridge = (db - np.percentile(db, 20, axis=1, keepdims=True)).max(axis=0)
    calm = ~ndimage.binary_dilation(ridge >= HOT_DB, iterations=PAD_FRAMES)   # padding drops echoes
    seconds = calm.sum() * NPERSEG / SR
    if seconds < MIN_GAP:
        return None
    edges = np.flatnonzero(np.diff(np.r_[0, calm.astype(int), 0]))
    fade = np.linspace(0, 1, 64, dtype=np.float32)
    parts = []
    for a, b in zip(edges[::2], edges[1::2]):
        part = x[a * NPERSEG: b * NPERSEG].copy()
        part[:64] *= fade
        part[-64:] *= fade[::-1]
        parts.append(part)
    joined = np.concatenate(parts)
    w = np.resize(joined, WIN)
    return w / (np.abs(w).max() + 1e-9), float(seconds)


def recordings(name: str) -> list[dict]:
    manifest = json.loads((ROOT / "data" / "dataset" / name / "manifest.json").read_text(encoding="utf8"))
    return [m for m in manifest if m["device"] == DEVICE and m["group"] in ("distress", "social", "sonar")]


def cut(name: str) -> None:
    X, rec, peak = [], [], []
    for k, m in enumerate(recordings(name)):
        x = load(ROOT / "data" / "dataset" / name / m["file"], m.get("samplerate_khz", ""), highpass=8_000)
        clip = gap_clip(x)
        if clip:
            X.append(clip[0]); rec.append(m["uid"]); peak.append(clip[1])
        if k % 50 == 0:
            print(f"{k}", flush=True)
    np.savez_compressed(ROOT / "data" / "windows" / f"{name}_silent.npz", X=np.stack(X), rec=np.array(rec), peak=np.array(peak))
    print(f"{len(X)} windows from {len(set(rec))} recordings")


def score(name: str) -> None:
    w = ROOT / "data" / "windows"
    d = np.load(w / f"{name}_silent.npz")
    rec, peak = d["rec"], d["peak"]
    P = np.load(w / f"{name}_silent_perch.npy")
    M = [m for m in recordings(name) if m["uid"] in set(rec)]
    y = np.array([m["group"] != "sonar" for m in M], int)
    sp = [m["taxon"] for m in M]
    key = lambda m: (m["taxon"], m["locality"], m["method"])
    exact = {key(m) for m in M if y[M.index(m)]} & {key(m) for m in M if not y[M.index(m)]}
    sub = np.array([key(m) in exact for m in M])
    secs = np.array([sf.info(ROOT / "data" / "dataset" / name / m["file"]).duration for m in M])
    out = {"recordings": len(M), "talking": int(y.sum()),
           "seconds_median": {"talking": round(float(np.median(secs[y == 1])), 1), "sonar": round(float(np.median(secs[y == 0])), 1)},
           "duration_only_auroc": round(float(auc(y, np.log(secs))), 3)}
    F = np.stack([P[rec == m["uid"]].mean(axis=0) for m in M])
    scores = [group_scores(F, y, sp, 0.001, seed=s) for s in range(5)]
    out["gaps"] = {"auroc": round(float(np.mean([auc(y, s) for s in scores])), 3),
                   "exact_auroc": round(float(np.mean([auc(y[sub], s[sub]) for s in scores])), 3),
                   "exact_recordings": int(sub.sum()), "exact_talking": int(y[sub].sum())}
    print(json.dumps(out, indent=1))
    (ROOT / "results" / "silence_control.json").write_text(json.dumps(out, indent=1), encoding="utf8")


if __name__ == "__main__":
    {"cut": cut, "score": score}[sys.argv[1]](sys.argv[2] if len(sys.argv) > 2 else "SEA_context")
