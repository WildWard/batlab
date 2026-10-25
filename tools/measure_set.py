"""Measure every recording's calls the way Listen does (start/end/characteristic frequency,
duration, bandwidth, slope, shape) and save one row per recording.

Uses WildWard Listen's call measurement code; expects the listen repo next to this one.
Usage: python tools/measure_set.py [name]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "wildward-listen"))
from listen.calls import measure_calls, summarise, FEATURES  # noqa: E402
from prepare_windows import load  # noqa: E402


def main(name: str = "SEA_full") -> None:
    src = ROOT / "data" / "dataset" / name
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf8"))
    rows = []
    for k, m in enumerate(manifest):
        try:
            x = load(src / m["file"], m.get("samplerate_khz", ""))
            s = summarise(measure_calls(x))
        except Exception as e:  # unreadable files are skipped, not fatal
            print("skip", m["file"], e)
            continue
        if s:
            rows.append({"uid": m["uid"], "taxon": m["taxon"], "genus": m["genus"], "site": f"{m['country']}/{m['locality']}"} | s)
        if k % 50 == 0:
            print(f"{k}/{len(manifest)}", flush=True)
    out = ROOT / "data" / "windows" / f"{name}_calls.json"
    out.write_text(json.dumps(rows, indent=0), encoding="utf8")
    print(f"{len(rows)} of {len(manifest)} recordings measured -> {out.name}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
