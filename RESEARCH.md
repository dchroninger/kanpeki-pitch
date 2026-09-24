# Pitch accent grader — research log

Living log of experiments on the grading pipeline (spike/). Newest sections at the bottom.

## Setup

- **Pipeline:** OpenJTalk expected accent → hiragana wav2vec2 CTC aligner (mora timing) → SwiftF0 pitch → learned per-mora "accent nucleus" classifier (HistGradientBoosting) → phrase posterior → verdict.
- **Verdict thresholds:** *balanced* = pass if P(expected) ≥ 0.8, flag if ≤ 0.2; *strict* = 0.9 / 0.1. Between = "not sure" (abstain).
- **Metrics** (per accent phrase):
  - **FA** (false alarm): native said it correctly, grader flagged it. Worst failure for learners.
  - **MISS**: grade against a deliberately wrong expected accent; grader passed it.
  - **caught**: same test, grader flagged it (good).
  - **abstain**: "not sure", no verdict.
- **Data:** JSUT basic5000 (1 female speaker, hand-labelled accents; 4000 train / 1000 test). JVS parallel100 (100 speakers, 49 M / 51 F, same sentences; labels = OpenJTalk prediction, phrases failed by >50% of speakers dropped as label errors).
- **Fixed test set (never trained on):** 30 JVS speakers (15 M / 15 F) × 10 JVS sentences, plus JSUT's last 1000 utterances.

## Findings so far (2026-09-24)

1. Rule-based grader (2-level template fit) passed synthetic TTS tests (~1–2% FA) but **failed on native speech: 44% FA.** Causes: *peak delay* (pitch peak lands in the mora after the nucleus) and *syllable-level fall* (drop after ン/ー/ッ, not the nucleus mora).
2. Learned classifier (JSUT-300) → ~10% FA on native speech; **generalises across speakers** (35 unseen JVS speakers similar to JSUT held-out). Males worse.
3. A first "unseen speaker" test was inflated (2–3% FA) because test speakers read the same sentences as training speakers. **Always hold out sentences as well as speakers.**
4. Manual review of 53 consensus-dropped JVS phrases: ~15 OpenJTalk reading/accent/segmentation errors (e.g. 明王 → アキラオー), ~16 compound phrasing differences, ~9 names not in any dictionary, **~5 true grader blind spots — all atamadaka** (accent on 1st mora; worst when that mora is devoiced, e.g. 来ていた, or long, e.g. ベース), ~7 unsure.

## Caching throughput (JVS, aligner + SwiftF0 per recording)

Mac: 10 cores (8P + 2E), 32 GB. Numbers include model load per worker.

| Config (workers × threads, device) | 60 recs rec/s | 200 recs rec/s | × realtime (200) |
|---|---|---|---|
| 1 × 2 CPU | 1.77 | – | – |
| 1 × 8 CPU | 2.14 | – | – |
| 4 × 2 CPU | 2.06 | – | – |
| 5 × 2 CPU | 2.09 | – | – |
| 6 × 1 CPU | 1.92 | – | – |
| 3 × 3 CPU | 2.38 | 3.13 | 24.9 |
| 1 × 2 MPS | 3.29 | – | – |
| 2 × 1 MPS | 4.38 | 6.82 | 54.2 |
| **3 × 1 MPS** | – | **7.21** | **57.3** |
| 4 × 1 MPS | – | 6.88 | 54.6 |
| 3 × 2 MPS | – | 6.75 | 53.6 |

- **Use MPS (Apple GPU), 3 workers × 1 thread.** ~5× the original single-process CPU run (~1.5 rec/s).
- GPU saturates at 2–3 workers; more workers or threads add nothing.
- CPU multi-process barely helps: the per-recording Python/NumPy work is small, and the model is the bottleneck; CPU threads contend.
- MPS output verified identical to CPU (20 recordings, 860/860 mora boundaries, 0 ms difference).
- Model load per worker (~5–8 s) dominates tiny runs — benchmark with ≥200 recordings.

## Data scaling: adding JVS sentences in batches of 1000 recordings

Each batch = 10 new sentences × 100 speakers (test speakers' recordings of new sentences are never used). Feature set v1.

Columns: FA M / F · MISS · caught · abstain on the fixed JVS test; **atamadaka** FA / abstain (phrases with accent on mora 1); JSUT held-out FA / MISS / abstain. All %.

| Stage | Train sentences (JVS) | Thresholds | JVS FA M / F | MISS | caught | abstain | atamadaka FA / abst | JSUT FA / MISS / abst |
|---|---|---|---|---|---|---|---|---|
| baseline (0 batches) | 20 | balanced | 7.6 / 8.8 | 1.8 | 85.9 | 17.4 | 8.6 / 10.9 | 7.2 / 0.9 / 20.0 |
| baseline (0 batches) | 20 | strict | 3.2 / 5.1 | 0.9 | 79.6 | 29.6 | 5.4 / 21.6 | 4.2 / 0.3 / 32.8 |
| +1 batch | 30 | balanced | 6.5 / 8.0 | 1.5 | 85.6 | 18.2 | 8.6 / 11.6 | 7.1 / 0.9 / 20.6 |
| +1 batch | 30 | strict | 3.4 / 4.8 | 0.9 | 79.5 | 29.3 | 6.0 / 22.2 | 4.3 / 0.3 / 32.9 |
| +2 batch | 40 | balanced | 6.1 / 7.5 | 1.9 | 85.6 | 18.0 | 8.4 / 12.0 | 7.3 / 0.9 / 20.9 |
| +2 batch | 40 | strict | 3.7 / 5.2 | 0.9 | 80.0 | 28.6 | 6.3 / 21.5 | 4.4 / 0.3 / 33.3 |
| +3 batch | 50 | balanced | 6.3 / 8.3 | 2.1 | 86.2 | 16.7 | 8.7 / 11.2 | 7.5 / 1.0 / 21.5 |
| +3 batch | 50 | strict | 2.9 / 4.7 | 0.9 | 80.2 | 28.1 | 4.9 / 21.6 | 4.3 / 0.4 / 34.4 |
| +4 batch | 60 | balanced | 6.8 / 7.9 | 2.1 | 86.4 | 16.5 | 9.8 / 10.9 | 7.8 / 0.9 / 21.4 |
| +4 batch | 60 | strict | 3.3 / 4.4 | 1.0 | 80.3 | 27.9 | 5.9 / 21.5 | 4.5 / 0.3 / 34.0 |
| +5 batch | 70 | balanced | 5.9 / 7.5 | 1.8 | 85.9 | 17.8 | 8.6 / 11.6 | 7.7 / 0.9 / 21.7 |
| +5 batch | 70 | strict | 2.6 / 4.1 | 0.9 | 79.6 | 28.8 | 4.9 / 21.5 | 4.4 / 0.4 / 34.6 |
| +6 batch | 80 | balanced | 5.9 / 7.6 | 1.8 | 86.2 | 17.5 | 8.6 / 11.1 | 8.2 / 1.0 / 22.1 |
| +6 batch | 80 | strict | 2.4 / 4.0 | 0.9 | 79.7 | 29.3 | 4.1 / 21.8 | 4.6 / 0.4 / 34.8 |
| +7 batch | 90 | balanced | 5.4 / 6.8 | 1.7 | 86.7 | 17.1 | 7.8 / 11.5 | 8.1 / 1.0 / 21.9 |
| +7 batch | 90 | strict | 2.5 / 3.8 | 0.8 | 79.6 | 29.2 | 4.3 / 21.9 | 4.7 / 0.4 / 34.8 |

### Batch throughput (3 × 1 MPS)

| Batch | Recordings | Wall (s) | rec/s | × realtime |
|---|---|---|---|---|
| 1 | 1000 | 111.7 | 8.95 | 60.5 |
| 2 | 999 | 129.1 | 7.74 | 62.0 |
| 3 | 1000 | 120.9 | 8.27 | 61.9 |
| 4 | 1000 | 122.2 | 8.18 | 63.1 |
| 5 | 1000 | 132.2 | 7.57 | 64.1 |
| 6 | 1000 | 139.3 | 7.18 | 61.4 |
| 7 | 999 | 139.7 | 7.15 | 65.6 |

7000 recordings in ~15 min of caching (vs ~75 min at the original ~1.5 rec/s). rec/s drifts with sentence length; × realtime is flat (~60–65×), so audio duration is the real cost driver. Retrain+eval between batches: ~1 min each.

### Data scaling conclusions

- **More distinct sentences steadily reduce false alarms**: strict JVS FA M/F 3.2 / 5.1 → 2.5 / 3.8, balanced 7.6 / 8.8 → 5.4 / 6.8 (20 → 90 training sentences). Roughly −25% relative; the curve has **not** flattened at 90.
- **Abstain does not move with data (~29% strict).** "Not sure" is a model/feature limitation, not a data-quantity one.
- **JSUT held-out FA creeps up** (4.2 → 4.7 strict) as JVS (OpenJTalk auto-labels) dominates training — the model absorbs some OpenJTalk label bias. Clean (hand) labels matter; keep an eye on this when adding auto-labelled data.
- Adding *speakers* on the *same* sentences mostly inflates scores (see finding 3); adding *sentences* is what generalises.

## Failure breakdown (tree v2, baseline data, strict; JSUT held-out + JVS test, 8801 phrases)

| Group | n | FA % | abstain % |
|---|---|---|---|
| all | 8801 | 4.0 | 39.7 |
| heiban (0) | 2591 | 0.7 | 21.1 |
| atamadaka (1) | 2443 | 3.6 | 25.5 |
| **nakadaka (2..n-1)** | 3617 | **6.8** | **61.5** |
| odaka (n) | 150 | 3.3 | 65.3 |
| phrase has devoiced mora | 240 | 7.9 | 58.8 |
| 1–2 morae | 792 | 1.5 | 29.7 |
| 8+ morae | 1186 | 4.0 | 50.6 |
| utterance-medial | 6232 | 4.6 | 41.5 |

**Correction to finding 4:** the manual review (5 examples) suggested atamadaka was the blind spot. The full breakdown shows the real weak spot is **nakadaka** — the grader can't decide between "drop after mora k" and "k+1" (peak delay), so it abstains on ~62% of them. Small manual samples mislead; always follow up with a breakdown.

## Feature & model experiments (fixed test protocol)

All at full data (JSUT-4000 + JVS 70 train speakers × 90 sentences) unless noted. JVS-dict = cleaner JVS subset (single noun + particles, OpenJTalk label agrees with Kanjium dictionary; 510 phrases).

| Model | Thresholds | JVS FA M / F | JVS MISS | JVS abstain | JVS-dict FA / MISS / abst | JSUT FA | JSUT MISS | JSUT abstain |
|---|---|---|---|---|---|---|---|---|
| tree v1 | 0.9/0.1 | 2.5 / 3.8 | 0.8 | 29.2 | – | 4.7 | 0.4 | 34.8 |
| tree v3 | 0.9/0.1 | 3.0 / 4.0 | 0.7 | 26.6 | 3.1 / 0.6 / 23.1 | 4.3 | 0.3 | 31.3 |
| tree v3 | 0.95/0.05 | 1.2 / 2.4 | 0.3 | 38.8 | 1.2 / 0.2 / 35.9 | 2.5 | 0.1 | 43.9 |
| seq (BiGRU), no aug | 0.9/0.1 | 6.9 / 6.8 | 2.3 | 13.1 | – | 3.0 | 0.7 | 15.0 |
| seq, pitch-range aug | 0.9/0.1 | 8.6 / 7.1 | 2.3 | 15.3 | – | 3.6 | 0.8 | 15.6 |
| **avg(tree v3, seq no-aug)** | **0.9/0.1** | **3.0 / 3.6** | 0.9 | **24.3** | **2.2 / 1.6 / 20.0** | **2.4** | 0.3 | **27.2** |
| avg(tree v3, seq no-aug) | 0.95/0.05 | 1.1 / 2.0 | 0.2 | 34.2 | 0.8 / 0.4 / 29.5 | 1.4 | 0.1 | 38.2 |

- **Features:** v2 (peak position, steepest-fall slope/position, utterance-relative pitch, interpolation for unvoiced morae) and v3 (+ 6 raw contour samples per mora for j−1..j+2) give small, consistent gains, mostly in abstain rate and JSUT FA. Differences < ~1.5 pts on the JVS test (~2k phrases) are within noise; JSUT held-out (~6.9k phrases) is the more sensitive benchmark.
- **Sequence model (BiGRU over per-mora contour samples):** far fewer abstains and much better on nakadaka (JSUT nakadaka abstain ~62% → ~24%), but **overconfident on unseen speakers** — more FA *and* MISS on JVS, including the dictionary-clean subset, so it is speaker overfitting (JSUT's single voice is ~45% of training phrases), not label noise. Pitch-range augmentation did not fix it.
- **Ensemble (average of posteriors) is the best trade-off:** keeps the tree's calibration on new speakers and gets the sequence model's coverage. Geometric mean was worse than arithmetic (inherits seq overconfidence).
- **Current pick for the demo:** avg(tree v3 full, seq no-aug full), strict 0.9/0.1.

## Demo sanity check (native JSUT recordings through the live demo server)

15 random demo sentences, 74 phrases: **55 ok, 1 wrong, 18 unsure.** (Optimistic — some of these sentences are in JSUT training.)

**Known issue:** VOICEVOX audio with *enforced* per-mora pitch (used for the "play correct pitch" reference) is unnaturally flat-stepped; the ensemble often scores it as wrong (seq model sees it as out-of-distribution). It sounds right to humans, but for the app: keep VOICEVOX's natural contour and only correct moras where its predicted pitch contradicts the dictionary pattern.

## Open items before iOS

- [ ] Real learner speech — the only data type we have zero of. Demo sessions are the first source.
- [ ] More distinct sentences (scaling curve not flat at 90).
- [ ] Speaker-robust sequence model (more speakers per sentence diversity, or speaker-normalised inputs) — would cut abstain further.
- [ ] Natural-contour TTS reference (see known issue).
- [ ] Product rules from review: skip names / out-of-dictionary words; accept alternative phrasing for long compounds; confirm kana reading in free-speech mode.
- [ ] Licensing: UTokyo commercial permission for models trained on JSUT/JVS.

## TTS reference: natural-contour fix (2026-09-24)

`tts.natural_fix`: keep VOICEVOX's predicted mora pitches; per phrase, blend toward flat dictionary levels only as far as needed (w ∈ {0, .25, .5, .75, 1}) until no L→H step rises < 1 st, no H→L step falls < 1 st, and no same-level step falls > 1.5 st.

Graded by the ensemble (20 demo sentences × 2 voices, 182 phrases): raw VOICEVOX ok 46.7 / **wrong 9.3** / unsure 44.0 · flat-enforced 43.4 / **9.3** / 47.3 · **natural-fix 42.9 / 4.9 / 52.2**. Blend weights: 32% untouched, 17% light, 42% at 0.75, 9% full. The rules may be stricter than real Tokyo speech (weak initial rise phrase-medially) — candidate for loosening. TTS audio of any kind is ~50% "unsure" for the grader: don't use TTS to evaluate the grader.
