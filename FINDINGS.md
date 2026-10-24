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

## Correction: time-expanded files (2026-10-24, later the same day)

**Bug:** 157 of the 237 Malaysian recordings are stored by ChiroVox time-expanded (a 384 kHz recording slowed 16x and saved at 24 kHz; the metadata gives the real rate). Test 2 resampled them as if they were real-time 24 kHz audio, so after the 12 kHz high-pass most of their calls were gone. Checked on a Hipposideros cervinus file: stored peak 7.2 kHz x 16 = 114.9 kHz, its known call frequency.

**Fix:** `prepare_windows.load` uses the recorded sample rate from the manifest when the file rate is below 192 kHz.

**Corrected results** (same split as Test 2):

| Features | Species | Genus | Species, unseen sites | Genus, unseen sites |
|---|---|---|---|---|
| Baseline spectrum | 73% | 81% | 45% | 59% |
| **Perch 2.0, time-expanded 12x** | **81%** | **92%** | **53%** | **84%** |

Site normalisation (subtracting each site's mean embedding) made unseen-site accuracy worse (37% to 26-30% on the buggy run), because sites differ in species mix, so the site mean carries species information. The fix for unseen sites is more sites per species, not normalisation.

## Test 3: many sites, three countries (2026-10-24, night)

**Data:** 319 ChiroVox recordings, 34 species recorded at 3 or more sites each, **66 sites** in Vietnam (166), Malaysia (142) and Indonesia (11). Sampled round-robin across sites (`tools/fetch_multisite.py`, files over 8 MB skipped, 1.2 GB cap). Same pipeline as Test 2 (time-expanded files read correctly).

| Features | Species (34, chance 3%) | Genus | Species, unseen sites | Genus, unseen sites |
|---|---|---|---|---|
| Baseline spectrum | 63% | 68% | 47% | 53% |
| **Perch 2.0, time-expanded 12x** | **78%** | **91%** | **62%** | **86%** |

95% confidence intervals (Wilson): species 73-82%, genus 87-94%, unseen-site species 56-67%, unseen-site genus 82-89%.

The rise from 53% (Test 2) to 62% on unseen sites is suggestive but **not a fair comparison**: different species sets and datasets. The controlled test below is the real evidence.

## Test 4: controlled, does site variety help? (`tools/site_curve.py`)

18 species with recordings from 5 or more sites. Each repeat holds out one site per species as the test, then trains on recordings from k other sites, always using the **same number of training recordings (4 per species)**, so only site variety changes. 40 repeats.

| Training sites | Accuracy on the held-out site (mean, 95% range over repeats) |
|---|---|
| 1 | 49% (29-67%) |
| 2 | 58% (42-78%) |
| 3 | 62% (46-79%) |
| 4 | 65% (51-77%) |

Paired on the same test, 4 sites vs 1: **+16 points (95% bootstrap CI +13 to +20), better in 90% of repeats.** Caveat: repeats share test sites, so the interval is somewhat optimistic; 18 species only.

## Limits of all tests so far
- **Coverage:** 34 of 390+ Southeast Asian species. Species the model never heard get forced into the nearest known one; open-set detection is not done yet.
- **Recordings:** mostly hand-release or close-range library calls, cleaner than passive survey recordings.
- **Sample size:** about ten recordings per species; per-species accuracy is very uncertain (roughly plus or minus 25 points).
- **Next:** test on real passive survey nights with expert labels, from partners.


## Test 5: can the model say "I don't know this species"? (`tools/open_set.py`) Negative result

Leave-one-species-out on the 34-species multi-site set: train on every species but one, then score how unfamiliar each test recording looks. Known test recordings come from held-out sites.

| Unfamiliarity score | AUROC (1.0 perfect, 0.5 chance) |
|---|---|
| 1 minus top classifier probability | 0.46 |
| Distance to nearest species centroid (Perch embedding) | 0.54 |
| Mean distance to 3 nearest training recordings | 0.52 |
| Distance from each known species' measured call-frequency range | 0.51 |

**None works.** A species the model never heard looks no more unusual than a known species recorded at a new site: site and device differences are as large as many species differences in this data. With 33 known species, their call-frequency ranges also cover almost the whole band, so frequency alone can't flag newcomers either.

**What Listen does instead:** it says plainly that it only suggests species it knows, and it has a "Not in my library" button. Passes labelled that way become an "Unknown" class, so similar-sounding passes get routed there instead of forced into a wrong species (in a test, labelling one 42-pass group sent 22 more passes to Unknown). Proper open-set detection remains an open problem; more species and real survey data should help.
