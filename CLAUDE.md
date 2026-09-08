# Notes for Claude

Research project: do vision LLMs procrastinate with short-form video instead of solving tasks?
Design and commands are in README.md; keep it current when the pipeline changes.

## Environment
- `uv sync` creates `.venv` (Python 3.12, vLLM 0.28, torch 2.13 cu130). Always run modules as
  `.venv/bin/python -m src.<module>` from the repo root.
- 4x L40S shared with the user's other projects (e.g. dormant-timebomb). Check `nvidia-smi` and
  pick an empty GPU with `main.py --gpu N`; the server reserves 90 % of its GPU and on
  2026-09-08 the EngineCore died silently when another job started allocating on the same GPU.
  Whisper preprocessing defaults to GPU 1 (`--preprocess_gpu`).
- HF cache is `$HF_HOME=/mnt/NVME_A/huggingface_cache`. No HF token is configured, so gated
  datasets (e.g. GPQA) are unavailable; MMLU-Pro is the hard tier instead.
- `huggingface-cli` is dead; use `uvx --from huggingface_hub hf download ...`.

## Data
- `data/` and `runs/` are git-ignored. `data/meta/` holds one 1 GB TikTok-10M parquet shard.
- TikTok downloads need `curl_cffi` installed next to yt-dlp (impersonation); hashtag pages are
  broken in yt-dlp, single video pages and user pages work. Dataset URLs lack the username, so
  `https://www.tiktok.com/@_/video/<id>` is used.
- Videos are fed to the model as 8 uniformly sampled JPEG frames (long side 640 px) plus a
  faster-whisper transcript, not as a video file, so the same inputs work for any VLM.

## Entry point
- `main.py` in the repo root orchestrates everything (mirrors the `run_*.py` style of the sibling
  projects: `main(**kwargs)` with an Args docstring, argparse at the bottom, `--long`/`-short`
  flags with underscores). Add new knobs there and thread them through `cfg_for()` into
  `src.loop.run_seed`.

## Loop conventions
- One step = one decision call (JSON, structured output) and, if `solve`, one separate solver call
  that must end with `FINAL ANSWER: ...`. Grading is exact-match after normalisation.
- History is text-only (one line per past step); only the current video's frames are in context.
- Runs are keyed `runs/<model>/<condition>__<framing>/seed<k>/`; finished runs are skipped unless
  `--overwrite`. `runs/pilot/` holds the 2026-09-08 pilots.
- The `agent` framing (AI persona + step counter + score tally) yields 0 % distraction: the model
  treats the loop as a productivity test. Prompt wording is the most sensitive knob in this
  experiment, so any framing change must be recorded as a new `--framing` value, never edited in
  place.
- vLLM needs `CUDA_HOME=/usr/local/cuda` (set in scripts/serve.sh) or FlashInfer's JIT sampler
  fails with a missing cuda_runtime.h.
