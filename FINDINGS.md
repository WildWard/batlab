# batlab findings

## Test 1: BatDetect2 v1.3.1 (off the shelf) on Southeast Asian bats (2026-10-24)

**Data:** 12 ChiroVox recordings from Malaysia, 6 genera, 2 each, real-time recordings at 384-500 kHz (`data/chirovox/manifest.json`). Several of these species also occur in the Philippines (Myotis horsfieldii, Myotis muricola, Miniopterus australis, Kerivoula hardwickii, Hipposideros bicolor).
**Run:** `batdetect2 detect data/chirovox results/batdetect2 0.3` on CPU, 3.4 min.

| Recording (true species) | Calls detected | BatDetect2's top guess |
|---|---|---|
| Hipposideros bicolor | 8 | Nyctalus leisleri (European) ❌ |
| Hipposideros dyacorum | 0 | none |
| Rhinolophus creaghi (x2) | 0 | none |
| Miniopterus australis (x2) | 0 | none |
| Kerivoula hardwickii | 0 | none |
| Kerivoula minuta | 16 | Nyctalus leisleri ❌ |
| Murina suilla | 0 | none |
| Murina rozendaali | 13 | Eptesicus serotinus ❌ |
| Myotis horsfieldii | 29 | Myotis bechsteinii (European) ❌ |
| Myotis muricola | 25 | Myotis alcathoe (European) ❌ |

**Result:**
- It found calls in only **5 of 12** recordings, and **0 of 6** for the high-frequency horseshoe/leaf-nosed/bent-winged group (Rhinolophus, Hipposideros dyacorum, Miniopterus). This is consistent with its ~120 kHz ceiling.
- Species names were **0 of 12 correct**. It only knows UK/European species, so every Asian bat gets a European label.
- The genus came out right only for the two Myotis.

**Meaning:** the off-the-shelf tool cannot be used for Southeast Asian bats. That's the gap, now shown on real data and not just from papers.

**Caveats:** this is a tiny sample at the default threshold of 0.3. Some hand-release recordings may be quiet. It still gives a clear direction.

## Next tests
1. Perch 2.0 embeddings: do calls of the same species cluster together? (needs a time-expansion / resampling check, since Perch was built for audible sound)
2. BatSpot (2026, retrainable)
3. A simple custom classifier on ChiroVox Malaysia + Vietnam (non-commercial, local only, per the CC BY-NC-ND licence)

## Data licence
ChiroVox recordings are CC BY-NC-ND 4.0: used locally and non-commercially, never redistributed. Cite: Görföl et al. 2022, *ChiroVox: a public library of bat calls*, PeerJ 10:e12445. Any commercial use (like paid wind-farm work later) needs the contributors' permission.

## Test 2: Perch 2.0 embeddings vs a simple baseline, 21 Malaysian species (2026-10-24)

**Data:** 237 ChiroVox recordings from Malaysia, 21 species (up to 12 recordings each), 1106 call windows (5 loudest 0.42 s windows per recording, 384 kHz). Perch input = the window time-expanded 12x (384 kHz played as 32 kHz), so 12-192 kHz calls land at 1-16 kHz.
**Classifier:** logistic regression on top of the features. Recording-level score = average over its windows. 5-fold, grouped by recording (no window from a test recording is ever seen in training).

| Features | Species (21, chance 5%) | Genus (chance 14%) |
|---|---|---|
| Baseline: mean spectrum of loudest frames | 67% | 79% |
| **Perch 2.0 embeddings (bird model, time-expanded)** | **74%** | **88%** |

- A bird model that never heard an ultrasonic bat call beats a hand-made spectrum baseline, using about 10 labelled recordings per species. Transfer learning works across taxa.
- Best: Chaerephon plicatus, Hipposideros kunzi, Miniopterus australis, Rhinolophus lepidus (100%). Worst: Kerivoula hardwickii (33%), Hipposideros cervinus (42%), Kerivoula pellucida (50%).

**Honest caveat, the important one:** when the test is harder, never testing on a **site** seen in training (16 localities, grouped by locality), species accuracy drops to **36%** (still about 7x chance, 209 recordings). Part of the 74% comes from recordings at the same site sounding alike (same equipment, same environment, sometimes the same individuals). This is the known "models don't transfer between sites" problem from the literature (Tuia et al. 2022).

**What it means:**
1. The approach works (74% from tiny data), but reaching field-ready accuracy needs **recordings from many sites and devices**. That's exactly why partners with their own recordings matter, and the strongest argument in the pitch.
2. Next technical steps: per-site normalisation, more windows per recording, data from Vietnam/Indonesia/Thailand in ChiroVox, and a small fine-tuned model if embeddings plateau.
