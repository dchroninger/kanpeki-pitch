# Pitch accent grading spike

Proves the on-device grading pipeline on the Mac before porting to iOS.

```
text ─► jpa.accent  (OpenJTalk: accent phrases, expected H/L per mora)
audio ─► jpa.align  (kana CTC model: what was said + mora timing)
      ─► jpa.pitch  (SwiftF0 → semitones re: speaker median)
      ─► jpa.grade  (mid-mora pitch → best-fit accent type per phrase → verdict + confidence)
jpa.tts: VOICEVOX with per-mora pitch overridden so references match the dictionary exactly
```

Setup: `uv sync`, then fetch models:
- aligner: `sakasegawa/japanese-wav2vec2-large-hiragana-ctc` → `models/` (+ clone `nyosegawa/hiragana-asr` into `vendor/`)
- VOICEVOX: `vendor/voicevox/download-osx-arm64 -o vendor/voicevox/vv` (interactive terms)

Run: `uv run python try_one.py 箸が好きです` · batch: `uv run python eval_synth.py`
