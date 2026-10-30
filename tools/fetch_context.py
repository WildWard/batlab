"""Build a set for the social-call test: every Distress/Social recording in ChiroVox plus sonar controls.

ChiroVox tags some recordings with a call type. Distress and Social are the bat talking to other bats,
Search is sonar. Differences between the two can come from the species, the site or the recording setup
rather than from the call, so controls are matched to each distress recording as tightly as the library
allows: same species, site and method first, then same genus and device. Resting recordings are kept as
a third group. Metadata comes from the raw lists already in raw/. Same licence rules as fetch_chirovox.py.
Usage: python tools/fetch_context.py [name] [budget_mb]
"""
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from fetch_chirovox import BASE, get, get_json
from fetch_dataset import content_length

ROOT = Path(__file__).resolve().parent.parent
MAX_FILE = 50 * 1024 * 1024
LOCAL_SETS = ["SEA_wide", "SEA_sites", "SEA_all", "SEA_full", "Malaysia"]
GROUP = {"Distress": "distress", "Social": "social", "Search": "sonar", "Searching": "sonar", "Flying": "sonar", "Resting": "resting"}
PER_DISTRESS = 3   # sonar controls per species-site-method group


def raw_records() -> list[dict]:
    rows = []
    for f in (ROOT / "raw").glob("*.json"):
        d = json.loads(f.read_text(encoding="utf8"))
        rows += d if isinstance(d, list) else []
    return [r for r in rows if r["recording_type"] == "Real time" and r["call_type"] in GROUP]


def pick(rows: list[dict]) -> list[tuple[dict, str]]:
    random.seed(7)
    talk = [r for r in rows if GROUP[r["call_type"]] in ("distress", "social")]
    sonar = [r for r in rows if GROUP[r["call_type"]] == "sonar"]
    resting = [r for r in rows if GROUP[r["call_type"]] == "resting"]
    by_exact, by_genus = defaultdict(list), defaultdict(list)
    for r in sonar:
        by_exact[(r["taxonname"], r["locality"], r["method"].lower())].append(r)
        by_genus[(r["genus"], r["device_model"])].append(r)
    chosen, seen = [(r, "exact") for r in talk], {r["chirovox"] for r in talk}
    for key in {(r["taxonname"], r["locality"], r["method"].lower()) for r in talk}:
        pool = by_exact.get(key, [])
        for r in random.sample(pool, min(PER_DISTRESS, len(pool))):
            if r["chirovox"] not in seen:
                chosen.append((r, "exact"))
                seen.add(r["chirovox"])
    for t in talk:   # no exact match: closest we can do is the same genus on the same device
        if (t["taxonname"], t["locality"], t["method"].lower()) in by_exact:
            continue
        pool = [r for r in by_genus.get((t["genus"], t["device_model"]), []) if r["chirovox"] not in seen]
        for r in random.sample(pool, min(2, len(pool))):
            chosen.append((r, "genus_device"))
            seen.add(r["chirovox"])
    chosen += [(r, "resting") for r in resting if r["chirovox"] not in seen]
    return chosen


def main(name: str = "SEA_context", budget_mb: float = 2500.0) -> None:
    out = ROOT / "data" / "dataset" / name
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8")) if manifest_path.exists() else []
    have = {m["uid"] for m in manifest}
    local = {}
    for s in LOCAL_SETS:
        p = ROOT / "data" / "dataset" / s / "manifest.json"
        if p.exists():
            for m in json.loads(p.read_text(encoding="utf8")):
                local.setdefault(m["uid"], ROOT / "data" / "dataset" / s / m["file"])
    rows = raw_records()
    todo = pick(rows)
    wanted = {r["chirovox"] for r, _ in todo}
    # labelled sonar we already hold costs nothing and gives the classifier more species and devices
    todo += [(r, "local_sonar") for r in rows if GROUP[r["call_type"]] == "sonar" and r["chirovox"] in local and r["chirovox"] not in wanted]
    print(f"{len(todo)} recordings wanted, {sum(r['chirovox'] in local for r, _ in todo)} already local", flush=True)
    downloaded_mb, skipped = 0.0, 0
    for k, (r, match) in enumerate(todo):
        uid = r["chirovox"]
        if uid in have:
            continue
        fname = f"{uid}_{r['taxonname'].replace(' ', '_').replace('/', '-')}.wav"
        try:
            if uid in local:
                try:
                    os.link(local[uid], out / fname)
                except OSError:
                    (out / fname).write_bytes(local[uid].read_bytes())
            else:
                rand = get_json(f"getrandtext=UID&value={uid}")[0]["randtext"]
                meta = get_json(f"getrecord=UID_randtext&value={uid}_{rand}")
                meta = meta[0] if isinstance(meta, list) else meta
                if "restricted" in str(meta.get("access", "")).lower():
                    skipped += 1
                    continue
                url = f"{BASE}/getphoto?ref={uid}_{rand}.WAV"
                size = content_length(url)
                if size > MAX_FILE:
                    skipped += 1
                    continue
                if downloaded_mb + size / 1e6 > budget_mb:
                    print(f"data budget of {budget_mb:.0f} MB reached")
                    break
                wav = get(url)
                if len(wav) < 1000 or not wav.startswith(b"RIFF"):
                    skipped += 1
                    continue
                (out / fname).write_bytes(wav)
                downloaded_mb += len(wav) / 1e6
                time.sleep(1)
        except Exception as e:   # one broken record shouldn't stop the run
            print("skip", uid, e, flush=True)
            skipped += 1
            continue
        manifest.append({"uid": uid, "taxon": r["taxonname"], "genus": r["genus"], "country": r["country"],
                         "samplerate_khz": r["samplerate"], "recorded_by": r["recorded_by"],
                         "locality": f"{r['country']}/{r['locality']}", "file": fname,
                         "group": GROUP[r["call_type"]], "call_type": r["call_type"], "match": match,
                         "method": r["method"].lower(), "device": r["device_model"]})
        have.add(uid)
        if k % 20 == 0:
            manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf8")
            print(f"{k}/{len(todo)} done, {downloaded_mb:.0f} MB, {skipped} skipped", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf8")
    print(f"set has {len(manifest)} recordings, downloaded {downloaded_mb:.0f} MB, {skipped} skipped")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "SEA_context", float(a[1]) if len(a) > 1 else 2500.0)
