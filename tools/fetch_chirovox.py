"""Download a small, polite sample of ChiroVox recordings for local model testing.

Licence note: most ChiroVox calls are CC BY-NC-ND 4.0. We only use them locally,
non-commercially, to evaluate existing models; we never redistribute the audio.
Cite: Görföl et al. 2022, ChiroVox: a public library of bat calls, PeerJ 10:e12445.
"""
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://www.chirovox.org"
UA = {"User-Agent": "Mozilla/5.0 (batlab research sample; contact via chirovox)"}
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "chirovox"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def get_json(query: str):
    return json.loads(get(f"{BASE}/ajax?m=chirovox&{query}"))


def main(country: str = "Malaysia", per_genus: int = 2) -> None:
    genera = ["Hipposideros", "Rhinolophus", "Kerivoula", "Myotis", "Miniopterus", "Murina"]
    records = get_json(f"getselectedlist=Country&value={country}")
    random.seed(7)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for genus in genera:
        pool = [r for r in records if r["genus"] == genus and r["recording_type"] == "Real time"
                and r["samplerate"] in ("384", "500")]
        random.shuffle(pool)
        got = 0
        for rec in pool:
            if got >= per_genus:
                break
            uid = rec["chirovox"]
            rand = get_json(f"getrandtext=UID&value={uid}")[0]["randtext"]
            meta = get_json(f"getrecord=UID_randtext&value={uid}_{rand}")
            meta = meta[0] if isinstance(meta, list) else meta
            if "restricted" in str(meta.get("access", "")).lower():
                continue
            wav = get(f"{BASE}/getphoto?ref={uid}_{rand}.WAV")
            if len(wav) < 1000 or not wav.startswith(b"RIFF"):
                continue
            path = OUT / f"{uid}_{rec['taxonname'].replace(' ', '_')}.wav"
            path.write_bytes(wav)
            manifest.append({"uid": uid, "taxon": rec["taxonname"], "genus": genus,
                             "samplerate_khz": rec["samplerate"], "access": meta.get("access"),
                             "recorded_by": rec["recorded_by"], "file": path.name})
            got += 1
            print(f"{uid} {rec['taxonname']} {len(wav)//1024} KB {meta.get('access')}")
            time.sleep(1)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf8")
    print(f"saved {len(manifest)} recordings to {OUT}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
