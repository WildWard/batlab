"""Make "field-like" copies of every call window: fainter, farther away and noisier, the way a
passive detector hears a bat that a library recording caught up close.

Per window: a distance effect (high frequencies fade faster in air, roughly 1-3 dB per 10 kHz,
plus overall attenuation), then background noise at a random 6-20 dB signal-to-noise ratio.
Saves <name>_aug.npz (same order as <name>.npz) for embed_perch.py.
Usage: python tools/augment_windows.py [name] [seed]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SR = 384_000


def field_like(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(w)
    f_khz = np.fft.rfftfreq(len(w), 1 / SR) / 1000
    tilt_db = rng.uniform(1.0, 3.0) * f_khz / 10           # extra loss at high frequency
    spec *= 10 ** (-tilt_db / 20)
    x = np.fft.irfft(spec, n=len(w))
    x *= 10 ** (-rng.uniform(0, 12) / 20)                   # farther away overall
    snr_db = rng.uniform(6, 20)
    noise = rng.standard_normal(len(x))
    noise = np.cumsum(noise) * 0.02 + noise                 # a little low-frequency rumble, like wind and insects
    noise *= np.sqrt(np.mean(x ** 2) / (np.mean(noise ** 2) + 1e-12)) * 10 ** (-snr_db / 20)
    y = x + noise
    return (y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def main(name: str = "SEA_full", seed: int = 0) -> None:
    d = np.load(ROOT / "data" / "windows" / f"{name}.npz")
    rng = np.random.default_rng(seed)
    X = np.stack([field_like(w, rng) for w in d["X"]])
    np.savez_compressed(ROOT / "data" / "windows" / f"{name}_aug.npz", X=X, species=d["species"], genus=d["genus"], rec=d["rec"])
    print(f"{len(X)} field-like windows -> {name}_aug.npz")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "SEA_full", int(a[1]) if len(a) > 1 else 0)
