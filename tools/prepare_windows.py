"""Cut call-rich windows out of each recording, ready for Perch and the baseline.

Ultrasound is made audible by time expansion: audio at 384 kHz played at 32 kHz
is slowed 12x, so 12-192 kHz bat calls land at 1-16 kHz, inside Perch's range.
One Perch input (5 s at 32 kHz) therefore covers 5/12 s of real time.
Usage: python tools/prepare_windows.py [country] [windows_per_recording]
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
SR = 384_000
EXPANSION = 12
WIN = 160_000  # 5 s at 32 kHz after expansion = 0.417 s real time


def load(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x[:, 0]
    if sr != SR:
        g = np.gcd(sr, SR)
        x = signal.resample_poly(x, SR // g, sr // g).astype(np.float32)
    sos = signal.butter(6, 12_000, "highpass", fs=SR, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def call_windows(x: np.ndarray, k: int) -> list[np.ndarray]:
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    hop = 4096
    frames = np.lib.stride_tricks.sliding_window_view(x, 8192)[::hop]
    energy = np.log10((frames ** 2).mean(axis=1) + 1e-12)
    # sum energy over each candidate window, then greedily take the loudest non-overlapping ones
    per_win = WIN // hop
    score = np.convolve(energy - energy.min(), np.ones(per_win), mode="valid")
    taken, out = np.zeros(len(score), bool), []
    for i in np.argsort(score)[::-1]:
        if len(out) >= k:
            break
        if taken[max(0, i - per_win): i + per_win].any():
            continue
        start = i * hop
        w = x[start: start + WIN]
        if len(w) < WIN:
            continue
        taken[i] = True
        out.append(w / (np.abs(w).max() + 1e-9))
    return out


def main(country: str = "Malaysia", k: int = 6) -> None:
    src = ROOT / "data" / "dataset" / country
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf8"))
    X, species, genus, rec = [], [], [], []
    for m in manifest:
        try:
            wins = call_windows(load(src / m["file"]), k)
        except Exception as e:  # unreadable or odd files are skipped, not fatal
            print("skip", m["file"], e)
            continue
        for w in wins:
            X.append(w)
            species.append(m["taxon"])
            genus.append(m["genus"])
            rec.append(m["uid"])
    out = ROOT / "data" / "windows"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{country}.npz", X=np.stack(X), species=np.array(species),
                        genus=np.array(genus), rec=np.array(rec))
    print(f"{len(X)} windows from {len(set(rec))} recordings, {len(set(species))} species")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "Malaysia", int(a[1]) if len(a) > 1 else 6)
