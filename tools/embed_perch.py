"""Embed time-expanded call windows with Perch 2.0 (CPU build from Kaggle).

Input: data/windows/<country>.npz from prepare_windows.py (384 kHz audio, 160000 samples).
The same 160000 samples are fed as if they were 32 kHz audio, i.e. slowed down 12x.
Output: data/windows/<country>_perch.npy, one 1536-d embedding per window.
Usage: python tools/embed_perch.py [country]
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path.home() / ".cache/kagglehub/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1"


def main(country: str = "Malaysia", batch: int = 16) -> None:
    X = np.load(ROOT / "data" / "windows" / f"{country}.npz")["X"].astype(np.float32)
    model = tf.saved_model.load(str(MODEL)).signatures["serving_default"]
    out = []
    for i in range(0, len(X), batch):
        out.append(model(inputs=tf.constant(X[i:i + batch]))["embedding"].numpy())
        print(f"{min(i + batch, len(X))}/{len(X)}", flush=True)
    np.save(ROOT / "data" / "windows" / f"{country}_perch.npy", np.concatenate(out))
    print("saved", np.concatenate(out).shape)


if __name__ == "__main__":
    main(*sys.argv[1:2])
