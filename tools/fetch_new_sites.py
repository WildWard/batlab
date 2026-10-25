"""Grow a set with one recording from every site it doesn't cover yet, per species.

Test 4 showed site variety is what improves accuracy at new sites, so this spends the data budget on
new localities rather than more recordings from known ones. Starts from an existing set (files are
hard-linked, not copied) and writes a new set. Same licence rules as fetch_chirovox.py.
Usage: python tools/fetch_new_sites.py [base] [name] [budget_mb] [country ...]
"""
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from fetch_chirovox import BASE, get, get_json
from fetch_dataset import content_length

ROOT = Path(__file__).resolve().parent.parent
MAX_FILE = 50 * 1024 * 1024   # uncovered sites are often long recordings (10-40 MB)
UNCERTAIN = re.compile(r"\b(cf|aff)\.|\bsp\.?$")


def main(base="SEA_all", name="SEA_sites", budget_mb=1300.0, countries=("Malaysia", "Vietnam", "Indonesia", "Cambodia")):
    src, out = ROOT / "data" / "dataset" / base, ROOT / "data" / "dataset" / name
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    else:   # start from the base set, minus uncertain labels
        manifest = [m for m in json.loads((src / "manifest.json").read_text(encoding="utf8")) if not UNCERTAIN.search(m["taxon"])]
        for m in manifest:
            if not (out / m["file"]).exists():
                try:
                    os.link(src / m["file"], out / m["file"])
                except OSError:
                    (out / m["file"]).write_bytes((src / m["file"]).read_bytes())
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf8")
    have = {m["uid"] for m in manifest}
    covered = defaultdict(set)
    for m in manifest:
        covered[m["taxon"]].add(m["locality"].split("/")[-1] if "/" in m["locality"] else m["locality"])

    records = []
    for c in countries:
        records += [r for r in get_json(f"getselectedlist=Country&value={c}")
                    if r["recording_type"] == "Real time" and r["samplerate"] in ("384", "500")
                    and " " in r["taxonname"] and "/" not in r["taxonname"] and not UNCERTAIN.search(r["taxonname"])]
        time.sleep(1)
    by_sp_site = defaultdict(list)
    for r in records:
        if r["chirovox"] not in have and r["locality"] not in covered[r["taxonname"]]:
            by_sp_site[(r["taxonname"], r["locality"])].append(r)
    todo = list(by_sp_site.items())
    random.seed(7)
    random.shuffle(todo)   # no alphabetical bias if the budget runs out
    print(f"{len(todo)} new species-site pairs to fetch")
    downloaded_mb, added = 0.0, 0
    for (sp, loc), pool in todo:
        random.shuffle(pool)
        for rec in pool:   # first usable recording from this site
            uid = rec["chirovox"]
            try:
                rand = get_json(f"getrandtext=UID&value={uid}")[0]["randtext"]
                meta = get_json(f"getrecord=UID_randtext&value={uid}_{rand}")
                meta = meta[0] if isinstance(meta, list) else meta
                if "restricted" in str(meta.get("access", "")).lower():
                    continue
                url = f"{BASE}/getphoto?ref={uid}_{rand}.WAV"
                size = content_length(url)
                if size > MAX_FILE:
                    continue
                if downloaded_mb + size / 1e6 > budget_mb:
                    print(f"data budget of {budget_mb:.0f} MB reached")
                    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf8")
                    print(f"added {added} recordings, {downloaded_mb:.0f} MB")
                    return
                wav = get(url)
            except Exception as e:   # one broken record shouldn't stop the run
                print("skip", uid, e)
                continue
            if len(wav) < 1000 or not wav.startswith(b"RIFF"):
                continue
            fname = f"{uid}_{sp.replace(' ', '_')}.wav"
            (out / fname).write_bytes(wav)
            downloaded_mb += len(wav) / 1e6
            manifest.append({"uid": uid, "taxon": sp, "genus": rec["genus"], "country": rec["country"],
                             "samplerate_khz": rec["samplerate"], "access": meta.get("access"),
                             "recorded_by": rec["recorded_by"], "locality": f"{rec['country']}/{loc}", "file": fname})   # same form as fetch_multisite
            have.add(uid)
            added += 1
            manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf8")
            time.sleep(1)
            break
        if added and added % 25 == 0:
            print(f"{added} added, {downloaded_mb:.0f} MB", flush=True)
    print(f"added {added} recordings, {downloaded_mb:.0f} MB")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "SEA_all", a[1] if len(a) > 1 else "SEA_sites", float(a[2]) if len(a) > 2 else 1300.0,
         tuple(a[3:]) if len(a) > 3 else ("Malaysia", "Vietnam", "Indonesia", "Cambodia"))
