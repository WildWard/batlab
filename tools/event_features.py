"""Find every sound event in a recording and measure it, without assuming it is sonar.

Listen's calls.py drops rising calls and anything over 120 ms, because those are rarely echolocation.
Bat social calls are exactly those sounds, so this has its own detector: any burst above the noise floor
from 8 kHz up, 1-800 ms long. Each event gets shape measurements (rise, reversals, harmonicity) on top
of the usual ones. Also cuts Perch windows: the loudest parts (calls) and the quietest (background),
the second being the control for "the classifier only hears the room".
Output: data/windows/<name>_events.json, data/windows/<name>.npz
Usage: python tools/event_features.py [name] [workers]
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import signal

from prepare_windows import SR, WIN, call_windows, load

ROOT = Path(__file__).resolve().parent.parent
NPERSEG, HOP = 1024, 128        # 2.7 ms window, 0.33 ms steps at 384 kHz
BAND = (8_000, 185_000)
SNR_DB = 15                     # above each frequency's own noise floor
GAP_FRAMES = 3                  # a 1 ms dropout doesn't split an event
MIN_MS, MAX_MS = 1.0, 800.0
BLOCK = 10 * SR                 # analyse 10 s at a time to keep memory flat
MAX_SECONDS = 120
MAX_EVENTS = 1500               # the loudest per recording
FRAME_MS = HOP / SR * 1000


def measure(db: np.ndarray, f_khz: np.ndarray, a: int, b: int, t0: float, t: np.ndarray) -> dict | None:
    """Measurements for frames a..b of a spectrogram in dB above the noise floor."""
    seg = np.maximum(db[:, a:b], 0)
    ridge_db, ridge = seg.max(axis=0), f_khz[seg.argmax(axis=0)]
    peak = ridge_db.max()
    keep = ridge_db > peak - 20
    if keep.sum() < 3:
        return None
    r = signal.medfilt(ridge[keep], 5) if keep.sum() >= 5 else ridge[keep]
    spec = seg.mean(axis=1)
    above = f_khz[spec > spec.max() - 20]
    # harmonicity: how strongly the event's spectrum repeats at a fixed spacing (2-60 kHz)
    df = f_khz[1] - f_khz[0]
    lo, hi = max(1, int(2 / df)), int(60 / df)
    s = spec - spec.mean()
    ac = np.correlate(s, s, "full")[len(s) - 1:]
    harm = float(ac[lo:hi].max() / (ac[0] + 1e-9))
    # reversals: the pitch changes direction by at least 3 kHz (a sonar sweep has none)
    turns, last, direction = 0, r[0], 0
    for v in r[1:]:
        if abs(v - last) >= 3:
            d = 1 if v > last else -1
            turns += direction != 0 and d != direction
            direction, last = d, v
    power = 10 ** (seg / 10)
    pm = power.mean(axis=1)
    return {"t": round(t0 + float(t[a]), 5), "dur_ms": round((b - a) * FRAME_MS, 2), "snr": round(float(peak), 1),
            "f_peak": round(float(ridge[ridge_db.argmax()]), 1), "f_start": round(float(r[0]), 1), "f_end": round(float(r[-1]), 1),
            "f_low": round(float(above.min()), 1), "f_high": round(float(above.max()), 1), "rise": round(float(r[-1] - r[0]), 1),
            "range": round(float(r.max() - r.min()), 1), "turns": int(turns), "harm": round(harm, 3),
            "flat": round(float(np.exp(np.log(pm + 1e-12).mean()) / (pm.mean() + 1e-12)), 4)}


def events(x: np.ndarray) -> list[dict]:
    out = []
    for start in range(0, min(len(x), MAX_SECONDS * SR), BLOCK):
        block = x[start: start + BLOCK]
        if len(block) < NPERSEG * 4:
            continue
        f, t, S = signal.stft(block, fs=SR, nperseg=NPERSEG, noverlap=NPERSEG - HOP, boundary=None, padded=False)
        band = (f >= BAND[0]) & (f <= BAND[1])
        db = 20 * np.log10(np.abs(S[band]) + 1e-9)
        db = (db - np.percentile(db, 20, axis=1, keepdims=True)).astype(np.float32)   # 20th: CF bats call most of the time
        hot = db.max(axis=0) > SNR_DB
        edges = np.flatnonzero(np.diff(np.r_[0, hot.astype(int), 0]))
        spans = list(zip(edges[::2], edges[1::2]))
        merged = []
        for a, b in spans:
            if merged and a - merged[-1][1] <= GAP_FRAMES:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        for a, b in merged:
            if MIN_MS <= (b - a) * FRAME_MS <= MAX_MS:
                m = measure(db, f[band] / 1000, a, b, start / SR, t)
                if m:
                    out.append(m)
    out.sort(key=lambda e: -e["snr"])
    return sorted(out[:MAX_EVENTS], key=lambda e: e["t"])


def quiet_windows(x: np.ndarray, k: int) -> list[np.ndarray]:
    if len(x) < WIN:
        return []
    hop = 4096
    frames = np.lib.stride_tricks.sliding_window_view(x, 8192)[::hop]
    energy = np.log10((frames ** 2).mean(axis=1) + 1e-12)
    per_win = WIN // hop
    score = np.convolve(energy, np.ones(per_win), mode="valid")
    taken, out = np.zeros(len(score), bool), []
    for i in np.argsort(score):
        if len(out) >= k:
            break
        if taken[max(0, i - per_win): i + per_win].any():
            continue
        w = x[i * hop: i * hop + WIN]
        if len(w) == WIN:
            taken[i] = True
            out.append(w / (np.abs(w).max() + 1e-9))
    return out


def one(job) -> tuple:
    path, khz, uid = job
    try:
        x = load(Path(path), khz, highpass=8_000)
        return uid, events(x), call_windows(x, 5), quiet_windows(x, 3), len(x) / SR
    except Exception as e:   # unreadable or odd files are skipped, not fatal
        return uid, None, [], [], str(e)


def main(name: str = "SEA_context", workers: int = 6) -> None:
    src = ROOT / "data" / "dataset" / name
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf8"))
    jobs = [(str(src / m["file"]), m.get("samplerate_khz", ""), m["uid"]) for m in manifest]
    ev, X, rec, kind, secs = {}, [], [], [], {}
    with ProcessPoolExecutor(workers) as pool:
        for k, (uid, e, calls, quiet, s) in enumerate(pool.map(one, jobs, chunksize=2)):
            if e is None:
                print("skip", uid, s, flush=True)
                continue
            ev[uid], secs[uid] = e, s
            for w in calls:
                X.append(w); rec.append(uid); kind.append("call")
            for w in quiet:
                X.append(w); rec.append(uid); kind.append("quiet")
            if k % 25 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    out = ROOT / "data" / "windows"
    (out / f"{name}_events.json").write_text(json.dumps({"events": ev, "seconds": secs}), encoding="utf8")
    np.savez_compressed(out / f"{name}.npz", X=np.stack(X), rec=np.array(rec), kind=np.array(kind))
    print(f"{sum(map(len, ev.values()))} events, {len(X)} windows, {len(ev)} recordings")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "SEA_context", int(a[1]) if len(a) > 1 else 6)
