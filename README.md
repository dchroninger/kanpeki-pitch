# kanpeki-pitch

Analyze and train Japanese **pitch accent**. Say a word or sentence, and it tells you whether your pitch
matches the dictionary accent, phrase by phrase, and where it didn't. It runs on-device: no cloud speech API.

> **Status: research spike + working demo.** This is a research prototype built to see whether on-device
> grading is feasible before porting it to iOS. It is not a product. See [Limitations](#limitations).

## How it works

```
text  ─► jpa.accent  OpenJTalk: accent phrases, expected high/low pattern per mora
audio ─► jpa.align   hiragana wav2vec2 CTC: what was said + when each mora happened
      ─► jpa.pitch   SwiftF0 → pitch in semitones relative to the speaker's median
      ─► jpa.grade   per-mora classifiers (tree + sequence model, averaged) → best-fit accent
                     type per phrase → verdict + confidence
jpa.tts              VOICEVOX with per-mora pitch shaped to match the dictionary, for reference audio
```

A verdict is *ok*, *wrong*, or *not sure*. The grader abstains rather than guessing: a false alarm on
correct speech is the worst failure for a learner.

## The demo

A small FastAPI app (`demo/`) with:

- **Word mode**: JLPT vocabulary (N5–N1), optionally with a particle and follow-up so heiban vs. odaka is
  audible; listen, shadow, record, get feedback per mora.
- **Minimal-pair mode**: words that share a reading but differ in accent (箸 / 橋 / 端).
- **Sentence mode**: sentences graded phrase by phrase, scored by the hardest word.
- Pitch guide overlay, English glosses, JLPT level picker, reference voices.

Run it: `uv run python demo/server.py`, then open <http://127.0.0.1:8765>.

## Results (honest version)

Measured per accent phrase on speakers **and sentences never seen in training** (30 JVS speakers × 10
sentences, plus JSUT's last 1,000 utterances). Current pick is an ensemble of a gradient-boosted tree model
and a BiGRU sequence model, strict thresholds (pass ≥ 0.9, flag ≤ 0.1):

| | false alarm | missed error | "not sure" |
|---|---|---|---|
| JVS (multi-speaker) | 3.7 % (M) / 4.4 % (F) | 1.0 % | 24 % |
| JSUT (single speaker) | 2.3 % | 0.3 % | 24 % |

"Missed error" is measured by grading native speech against a deliberately wrong accent.
The full experiment log, including what didn't work, is in [`RESEARCH.md`](RESEARCH.md).

## Limitations

- **No real learner speech yet**, the one data type this most needs. Numbers above are on native speakers.
- Known blind spot: atamadaka (accent on the first mora), worst when that mora is devoiced or long.
- TTS audio grades as "not sure" about half the time; don't use synthetic speech to evaluate the grader.
- Labels for the multi-speaker corpus come from OpenJTalk's own predictions (phrases most speakers
  "failed" were dropped as likely label errors), so some label noise remains.
- Trained models are derived from research corpora with usage restrictions; see [NOTICE.md](NOTICE.md).

## Setup

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and macOS on Apple silicon (VOICEVOX core wheel).

```bash
uv sync
```

Then fetch the third-party pieces this repo deliberately does **not** include (they have their own
licenses; see [NOTICE.md](NOTICE.md)):

| Path | What | Where from |
|---|---|---|
| `models/` | hiragana CTC aligner weights | `sakasegawa/japanese-wav2vec2-large-hiragana-ctc` on Hugging Face |
| `vendor/hiragana-asr/` | aligner code | <https://github.com/nyosegawa/hiragana-asr> |
| `vendor/voicevox/` | VOICEVOX core + voices | `download-osx-arm64` from the VOICEVOX project (interactive terms) |
| `vendor/jlpt-word-list/` | JLPT word lists | <https://github.com/elzup/jlpt-word-list> |
| `vendor/kanjium/accents.txt` | accent dictionary used for word mode | Kanjium; **personal/non-commercial use only** |
| `vendor/jmdict/` | English glosses | JMdict (EDRDG) |
| `vendor/jsut/`, `vendor/jvs/` | evaluation / training corpora | University of Tokyo (Takamichi lab); only needed to reproduce experiments |

Quick check: `uv run python try_one.py 箸が好きです`. Batch evaluation: `uv run python eval_synth.py`.

## Your recordings stay yours

Recordings made in the demo are written to `demo/recordings/` and `demo/private/`, which are gitignored,
along with all audio file types. Nothing is uploaded anywhere.

## Layout

```
jpa/        the grading library (accent, align, pitch, grade, tts)
demo/       FastAPI demo app + practice items
RESEARCH.md experiment log
*.py        data caching, training, and evaluation scripts
```

License: not yet chosen. All rights reserved until a license is added.
