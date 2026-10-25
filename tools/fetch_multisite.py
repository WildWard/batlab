"""Build a multi-country, multi-site ChiroVox set: species recorded at several localities,
sampled round-robin across localities so no single site dominates a species.

Same licence rules as fetch_chirovox.py: local, non-commercial use only, never redistributed.
Usage: python tools/fetch_multisite.py [name] [per_species] [min_sites] [country ...]
"""
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from fetch_chirovox import BASE, get, get_json
from fetch_dataset import MAX_BYTES, content_length

ROOT = Path(__file__).resolve().parent.parent


MAX_FILE = 8 * 1024 * 1024   # smaller files save mobile data; most single passes fit easily


def folder_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.glob("*.wav")) / 1e6


def main(name="SEA", per_species=15, min_sites=3, countries=("Malaysia", "Vietnam", "Indonesia"), budget_mb=1200.0) -> None:
    out = ROOT / "data" / "dataset" / name
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8")) if manifest_path.exists() else []
    have = {m["uid"] for m in manifest}

    records = []
    for c in countries:
        records += [r for r in get_json(f"getselectedlist=Country&value={c}")
                    if r["recording_type"] == "Real time" and r["samplerate"] in ("384", "500") and " " in r["taxonname"]
                    and "/" not in r["taxonname"]]  # "A b/c" = uncertain ID, not usable as a label
        time.sleep(1)
    site_of = lambda r: f"{r['country']}/{r['locality']}"
    sites = defaultdict(set)
    for r in records:
        sites[r["taxonname"]].add(site_of(r))
    species = sorted(s for s, ss in sites.items() if len(ss) >= min_sites)
    print(f"{len(species)} species recorded at >= {min_sites} sites")

    random.seed(13)
    for sp in species:
        by_site = defaultdict(list)
        for r in records:
            if r["taxonname"] == sp:
                by_site[site_of(r)].append(r)
        for pool in by_site.values():
            random.shuffle(pool)
        order = sorted(by_site, key=lambda s: random.random())
        got = sum(1 for m in manifest if m["taxon"] == sp)
        while got < per_species and any(by_site.values()):
            for s in order:  # one recording per site per round
                if got >= per_species or not by_site[s]:
                    continue
                rec = by_site[s].pop()
                uid = rec["chirovox"]
                if uid in have:
                    continue
                fname = f"{uid}_{sp.replace(' ', '_')}.wav"
                local = next((p for p in (ROOT / "data" / "dataset" / d / fname for d in ("Malaysia", "SEA", "SEA_full"))
                              if p.exists() and p.parent != out), ROOT / "data" / "dataset" / "Malaysia" / fname)
                if local.exists() and local.parent != out:  # already downloaded for an earlier set, reuse it
                    (out / fname).write_bytes(local.read_bytes())
                    manifest.append({"uid": uid, "taxon": sp, "genus": rec["genus"], "country": rec["country"],
                                     "samplerate_khz": rec["samplerate"], "access": "CC BY-NC-ND 4.0",
                                     "recorded_by": rec["recorded_by"], "locality": s, "file": fname})
                    have.add(uid)
                    got += 1
                    continue
                try:
                    rand = get_json(f"getrandtext=UID&value={uid}")[0]["randtext"]
                    meta = get_json(f"getrecord=UID_randtext&value={uid}_{rand}")
                    meta = meta[0] if isinstance(meta, list) else meta
                    if "restricted" in str(meta.get("access", "")).lower():
                        continue
                    url = f"{BASE}/getphoto?ref={uid}_{rand}.WAV"
                    if content_length(url) > MAX_FILE:
                        continue
                    if folder_mb(out) > budget_mb:
                        print(f"data budget of {budget_mb:.0f} MB reached, stopping")
                        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf8")
                        return
                    wav = get(url)
                except Exception as e:  # one broken record shouldn't stop the run
                    print("skip", uid, e)
                    continue
                if len(wav) < 1000 or not wav.startswith(b"RIFF") or len(wav) > MAX_FILE:
                    continue
                (out / fname).write_bytes(wav)
                manifest.append({"uid": uid, "taxon": sp, "genus": rec["genus"], "country": rec["country"],
                                 "samplerate_khz": rec["samplerate"], "access": meta.get("access"),
                                 "recorded_by": rec["recorded_by"], "locality": s, "file": fname})
                have.add(uid)
                got += 1
                manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf8")
                time.sleep(1)
        print(f"{sp}: {got} recordings from {len({m['locality'] for m in manifest if m['taxon'] == sp})} sites", flush=True)
    print(f"total {len(manifest)} recordings")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "SEA", int(a[1]) if len(a) > 1 else 15, int(a[2]) if len(a) > 2 else 3,
         tuple(a[3:]) if len(a) > 3 else ("Malaysia", "Vietnam", "Indonesia"),
         float(__import__("os").environ.get("BUDGET_MB", 1200)))
