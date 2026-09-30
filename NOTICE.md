# Third-party components, data, and licenses

This repository contains **only original source code, documentation, and aggregate evaluation metrics**.
It does not contain, and must not be used to redistribute, any third-party dataset, model weights, or
audio. Those are fetched locally (see the README) and are gitignored. Recordings of the author's voice are
never committed.

This is a good-faith summary of what was checked while preparing the repository, **not legal advice**.
"Not verified" means exactly that: check the upstream terms before relying on it.

| Component | Used for | License / terms |
|---|---|---|
| pyopenjtalk-plus (Open JTalk) | expected accent and readings | MIT (package metadata) |
| hiragana-asr (nyosegawa) | aligner code | Apache-2.0 |
| `japanese-wav2vec2-large-hiragana-ctc` weights | mora alignment | Not verified; see the model card on Hugging Face |
| SwiftF0 | pitch tracking | MIT |
| PyTorch, Transformers, scikit-learn, FastAPI, soundfile, huggingface_hub | runtime | Permissive (Apache-2.0 / BSD / MIT) |
| VOICEVOX core / engine / voices | reference audio (TTS) | Each voice character has its own terms, typically including a credit requirement; see the VOICEVOX terms |
| AivisSpeech and its voice models | reference audio (listening only) | Each voice model has its own license; see Aivis Hub |
| Kanjium `accents.txt` | accent dictionary for word mode | Provenance is mixed and includes commercial-dictionary-derived data. **Not redistributed here; treat as personal / non-commercial use only** |
| JLPT word lists (elzup/jlpt-word-list) | vocabulary by level | MIT; derived from tanos.co.uk decks |
| JMdict | English glosses | EDRDG license (CC BY-SA 4.0); attribution required |
| JSUT corpus | training and evaluation | Label data CC BY-SA 4.0; audio: see the corpus terms |
| JVS corpus | training and evaluation | Audio: academic and non-commercial research and personal use only; **redistribution not permitted**; commercial use requires separate arrangement with the authors. Tags CC BY-SA 4.0 |

## Trained models

The classifier and sequence-model files produced by the training scripts (`*.pkl`, `*.pt`) are derived from
JVS and JSUT. They are **not** included, and should not be published or used commercially without
permission from the corpus authors. (This is an open item in `RESEARCH.md`.)

## If you publish a demo

Credit the voices you use (for example, "VOICEVOX: <character name>") and the attributions above.
