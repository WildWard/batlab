# batlab

Open experiments in recognising **Southeast Asian bat species from their echolocation calls**, by [WildWard](https://wildward.github.io).

Most free bat-call AI was trained on European and North American bats. This repo tests how those tools do on Asian bats, and tries a simple alternative: reuse Google's **Perch 2.0** (a bioacoustics model trained mostly on birds) by time-expanding ultrasonic calls into its hearing range, then train a small classifier on top.

## Results so far

| Test | Data | Result |
|---|---|---|
| BatDetect2 v1.3.1, off the shelf | 12 Malaysian recordings, 6 genera | 0 of 12 species correct, calls found in 5 of 12 recordings |
| Spectrum baseline + logistic regression | 237 recordings, 21 Malaysian species | 73% species / 81% genus |
| **Perch 2.0 embeddings + logistic regression** | same | **81% species / 92% genus** |
| Perch 2.0, tested only on unseen recording sites | same, grouped by locality | 53% species / 84% genus |
| **Perch 2.0, many sites** | 319 recordings, 34 species, 66 sites in Vietnam, Malaysia, Indonesia | **78% species (95% CI 73-82) / 91% genus (87-94); 62% (56-67) / 86% (82-89) on unseen sites** |
| **Perch 2.0, 49 species** | 595 recordings, 85 sites, each species at 3+ sites | **83% species (80-86) / 90% genus (87-92); 63% (59-67) / 82% (79-85) on unseen sites** |
| **Perch 2.0, 91 species** | 855 recordings, every confirmed ChiroVox species in Vietnam, Malaysia, Indonesia | **78% species (75-81) / 90% genus (87-91)** |
| Controlled, repeated on 23 species with 5+ sites | 30 repeats | 1 site 49%, 4 sites 66%; 4 vs 1: +18 points (95% CI +14 to +22) |
| Controlled: more training sites, same amount of data | 18 species with 5+ sites, one held-out site each, 40 repeats | 1 site 49%, 2 sites 58%, 3 sites 62%, 4 sites 65%; 4 vs 1 on the same test: +16 points (95% CI +13 to +20) |

Scores are per recording (window predictions averaged), with cross-validation grouped by recording so no recording is used for both training and testing. Chance is 5% for species and 14% for genus.

> **Correction, 24 Oct 2026:** 157 of the 237 recordings are stored in ChiroVox time-expanded (slowed 16x, saved at 24 kHz). The first version read them as real-time audio, which filtered most of their calls away. `prepare_windows.py` now uses the library's recorded sample rate. Every score went up; earlier numbers were 74% / 88% / 36%.

**The honest headline:** the method learns Southeast Asian species from about ten examples each. Accuracy drops on sites it has never heard, and a controlled test shows that training on more sites, with the same amount of data, closes much of that gap. Scope matters: 91 of 390+ Southeast Asian species, mostly clean close-range library recordings, about ten per species, so these numbers are early evidence, not field accuracy. Field use needs survey recordings from many more sites and devices, and a way to flag species the model has never heard. Details and caveats are in [FINDINGS.md](FINDINGS.md).

## How it works

1. `tools/fetch_dataset.py` downloads a small balanced sample of recordings from the [ChiroVox](https://www.chirovox.org) public bat call library (politely rate-limited, licence noted per file).
2. `tools/prepare_windows.py` resamples to 384 kHz, high-passes at 12 kHz, and cuts the loudest call windows. Each window is 160,000 samples, so playing it at 32 kHz slows it down 12x and moves 12-192 kHz calls to 1-16 kHz.
3. `tools/embed_perch.py` turns each window into a 1536-number Perch 2.0 embedding.
4. `tools/evaluate.py` compares a hand-made spectrum baseline with Perch embeddings, at species and genus level.

`tools/fetch_chirovox.py` grabs the tiny 12-recording sample used for the BatDetect2 test.

## Run it

Python 3.12, CPU only.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
cd tools
python fetch_dataset.py Malaysia 12 14
python prepare_windows.py Malaysia 5
python embed_perch.py Malaysia
python evaluate.py Malaysia
```

Perch 2.0 (CPU build) is downloaded from Kaggle with `kagglehub` on first use.

## Data and licences

- **No audio is included in this repo.** ChiroVox recordings are mostly CC BY-NC-ND 4.0. They were used locally for non-commercial research and are not redistributed. If you use them, cite: Görföl T. et al. (2022) *ChiroVox: a public library of bat calls.* PeerJ 10:e12445, and respect each recording's licence. Commercial use needs the rights holders' permission.
- Perch 2.0: Google DeepMind, see its model card on Kaggle for its licence.
- BatDetect2: Mac Aodha et al., [github.com/macaodha/batdetect2](https://github.com/macaodha/batdetect2).
- Code in this repo: MIT licence.

## Want to help?

If you hold identified bat recordings from Southeast Asia, or you're tired of identifying calls by hand, we'd like to hear from you: **wildward.org@gmail.com**. The goal is a free, offline tool that researchers can train on their own species, and recordings from many sites are what it needs most.
