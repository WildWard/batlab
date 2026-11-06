"""Download Hungarian and Bulgarian ChiroVox recordings for the cross-country test (Test 14).

Hungary is almost all one recordist (Estók Péter) and one detector (Pettersson D500X); Bulgaria is three
teams on three detector types. Locality is often empty, so the site is a 0.1 degree grid cell (about 10 km).
Up to PER_SPECIES recordings per species and country, spread round-robin over sites. Metadata lists come
from raw/europe/ (saved 2026-11-06). Same licence rules as fetch_chirovox.py: local, non-commercial, never shared.
Usage: python tools/fetch_europe.py [per_species] [budget_mb]
"""
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from fetch_chirovox import BASE, get, get_json
from fetch_dataset import content_length

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 25 * 1024 * 1024


def cell(r: dict) -> str:
    try:
        return f"{float(r['latitude']):.1f},{float(r['longitude']):.1f}"
    except ValueError:
        return r["locality"] or "unknown"


def spread(pool: list[dict]) -> list[dict]:
    """Round-robin over sites, so a capped sample covers as many places as possible."""
    by_site = defaultdict(list)
    for r in pool:
        by_site[cell(r)].append(r)
    queues = [random.sample(v, len(v)) for v in by_site.values()]
    random.shuffle(queues)
    out = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop())
    return out


def main(per_species: int = 20, budget_mb: float = 1500.0) -> None:
    random.seed(14)
    out = ROOT / "data" / "dataset" / "Europe"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8")) if manifest_path.exists() else []
    have = {m["uid"] for m in manifest}
    used = sum((out / m["file"]).stat().st_size for m in manifest if (out / m["file"]).exists())
    for country in ("Hungary", "Bulgaria"):
        rows = json.loads((ROOT / "raw" / "europe" / f"{country}.json").read_text(encoding="utf8"))
        rows = [r | {"taxonname": " ".join(r["taxonname"].split())} for r in rows
                if r["recording_type"] == "Real time" and r["taxon_rank"] != "genus" and "/" not in r["taxonname"]]
        species = sorted({r["taxonname"] for r in rows})
        for sp in species:
            got = sum(1 for m in manifest if m["taxon"] == sp and m["country"] == country)
            for rec in spread([r for r in rows if r["taxonname"] == sp]):
                if got >= per_species or used > budget_mb * 1024 * 1024:
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
                used += len(wav)
                manifest.append({"uid": uid, "taxon": sp, "genus": rec["genus"], "country": country,
                                 "samplerate_khz": rec["samplerate"] or meta.get("samplerate") or "",
                                 "te": meta.get("te"), "access": meta.get("access"), "recorded_by": rec["recorded_by"],
                                 "device": rec["device_model"].strip(), "method": rec["method"].lower(),
                                 "locality": cell(rec), "file": name})
                have.add(uid)
                got += 1
                manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf8")
                time.sleep(1)
            print(f"{country} {sp}: {got}  ({used / 1e6:.0f} MB)", flush=True)
    print(f"total {len(manifest)} recordings, {used / 1e6:.0f} MB in {out}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 20, float(a[1]) if len(a) > 1 else 1500.0)
