"""Download a balanced multi-species ChiroVox set for local classifier experiments.

Same licence rules as fetch_chirovox.py: local, non-commercial use only, never redistributed.
Usage: python tools/fetch_dataset.py [country] [per_species] [min_records]
"""
import json
import random
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

from fetch_chirovox import BASE, UA, get, get_json

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 25 * 1024 * 1024


def content_length(url: str) -> int:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def main(country: str = "Malaysia", per_species: int = 12, min_records: int = 14) -> None:
    out = ROOT / "data" / "dataset" / country
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8")) if manifest_path.exists() else []
    have = {m["uid"] for m in manifest}

    records = [r for r in get_json(f"getselectedlist=Country&value={country}")
               if r["recording_type"] == "Real time" and r["samplerate"] in ("384", "500")]
    counts = Counter(r["taxonname"] for r in records)
    species = [s for s, n in counts.most_common() if n >= min_records and " " in s]
    print(f"{len(species)} species with >= {min_records} recordings")

    random.seed(11)
    for sp in species:
        pool = [r for r in records if r["taxonname"] == sp]
        random.shuffle(pool)
        got = sum(1 for m in manifest if m["taxon"] == sp)
        for rec in pool:
            if got >= per_species:
                break
            uid = rec["chirovox"]
            if uid in have:
                continue
            rand = get_json(f"getrandtext=UID&value={uid}")[0]["randtext"]
            meta = get_json(f"getrecord=UID_randtext&value={uid}_{rand}")
            meta = meta[0] if isinstance(meta, list) else meta
            if "restricted" in str(meta.get("access", "")).lower():
                continue
            url = f"{BASE}/getphoto?ref={uid}_{rand}.WAV"
            if content_length(url) > MAX_BYTES:
                continue
            wav = get(url)
            if len(wav) < 1000 or not wav.startswith(b"RIFF") or len(wav) > MAX_BYTES:
                continue
            name = f"{uid}_{sp.replace(' ', '_')}.wav"
            (out / name).write_bytes(wav)
            manifest.append({"uid": uid, "taxon": sp, "genus": rec["genus"],
                             "samplerate_khz": rec["samplerate"], "access": meta.get("access"),
                             "recorded_by": rec["recorded_by"], "locality": rec["locality"],
                             "file": name})
            have.add(uid)
            got += 1
            manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf8")
            time.sleep(1)
        print(f"{sp}: {got}")
    print(f"total {len(manifest)} recordings in {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0] if args else "Malaysia",
         int(args[1]) if len(args) > 1 else 12,
         int(args[2]) if len(args) > 2 else 14)
