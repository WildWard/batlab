"""Drop uncertain identifications and merge spelling variants in a prepared window set.

"cf." (compare with), "aff." (related to) and "sp" labels are not confirmed species, so they
can't be used as species labels; "Rhinolophus pearsoni" and "pearsonii" are the same bat.
Writes <name>_clean.npz and <name>_clean_perch.npy next to the originals.
Usage: python tools/clean_set.py [name]
"""
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAME = {"Rhinolophus pearsoni": "Rhinolophus pearsonii"}


def uncertain(taxon: str) -> bool:
    return bool(re.search(r"\b(cf|aff)\.|\bsp\.?$", taxon)) or len(taxon.split()) < 2


def main(name: str = "SEA_all") -> None:
    w = ROOT / "data" / "windows"
    d = np.load(w / f"{name}.npz")
    P = np.load(w / f"{name}_perch.npy")
    keep = np.array([not uncertain(t) for t in d["species"]])
    species = np.array([SAME.get(t, t) for t in d["species"][keep]])
    np.savez_compressed(w / f"{name}_clean.npz", X=d["X"][keep], species=species, genus=d["genus"][keep], rec=d["rec"][keep])
    np.save(w / f"{name}_clean_perch.npy", P[keep])
    # matching manifest (no audio) so site-based tests can find each recording's locality
    import json
    src = ROOT / "data" / "dataset" / name
    man = [m | {"taxon": SAME.get(m["taxon"], m["taxon"])} for m in json.loads((src / "manifest.json").read_text(encoding="utf8"))
           if not uncertain(m["taxon"])]
    (ROOT / "data" / "dataset" / f"{name}_clean").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "dataset" / f"{name}_clean" / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf8")
    print(f"kept {keep.sum()} of {len(keep)} windows, {len(set(d['rec'][keep]))} recordings, {len(set(species))} species")


if __name__ == "__main__":
    main(*sys.argv[1:2])
