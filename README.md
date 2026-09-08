# 26-social-media-addicted-llms

Do LLMs get addicted to social media? Do LLMs prefer to scroll through short-form video instead of
solving benchmark tasks?

## Experiment 1: the procrastination loop

A vision LLM sits in a loop of `N` steps (default 100). At every step it sees

* the next benchmark task from a queue (GSM8K = easy, MATH-500 = medium, MMLU-Pro = hard,
  interleaved), and
* the next short-form video in an autoplaying TikTok-like feed (8 sampled frames, caption,
  hashtags, sound, Whisper transcript),

and picks exactly one action: `solve` the task or `watch` the video. The feed autoplays: by default a
new video is shown at every step whether or not the previous one was watched (`--feed-advance
always`); with `--feed-advance on_watch` the queued video stays until it is watched. Choosing
`solve` triggers a second, clean model call that answers the task, which is graded automatically. A compact text history of all previous choices is
kept in context. After `N` steps we measure how much of the time the model spent in the feed.

### Conditions

| condition    | alternative to the task                                  | purpose |
|--------------|----------------------------------------------------------|---------|
| `video`      | frames + caption + transcript of the next video          | treatment |
| `text_only`  | caption + transcript only, no frames                     | isolate the visual pull |
| `boring`     | a blank break screen                                     | separate video attraction from task avoidance |
| `instructed` | like `video`, but the prompt states the goal is solving tasks | does distraction survive an explicit goal? |

### Framings (`--framing`, orthogonal to the condition)

| framing   | wording |
|-----------|---------|
| `agent`   | "You are an AI agent ..." with a step counter, task/score tally and a labelled task queue (the original prompt). Pilot (2 seeds x 20 steps): 0 % distraction. |
| `neutral` | a phone and a stack of optional worksheets, nobody is evaluating you; no counters, feed shown before the problem. Pilot (2 x 10 steps): 5 %. |
| `human`   | like `neutral`, but "you are a person on the couch on a free evening". Pilot (2 x 10 steps): 10 %. |

Pilot runs live in `runs/pilot/` (with the old `on_watch` feed behaviour).

Runs are stored under `runs/<model>/<condition>__<framing>/seed<k>/`.

### Metrics (`src/analyze.py`)

Fraction of steps distracted, tasks solved and accuracy, distraction rate by task difficulty,
step of first distraction, watch-streak lengths, and the transition probabilities
P(distract | distracted before) vs. P(distract | solved before).

## Setup

```bash
uv sync                                    # Python 3.12, vLLM, yt-dlp, faster-whisper, ...
```

### 1. Video pool

Metadata comes from one shard of the `The-data-company/TikTok-10M` dataset (URLs + engagement
stats); videos are fetched with yt-dlp (needs `curl_cffi` for TikTok impersonation).

```bash
uvx --from huggingface_hub hf download The-data-company/TikTok-10M \
    data/train-00008-of-00010.parquet --repo-type dataset --local-dir data/meta
python -m src.fetch_videos sample --n 300 --seed 0     # -> data/videos/pool.jsonl
python -m src.fetch_videos download --workers 3        # -> data/videos/raw/*.mp4, manifest.jsonl
CUDA_VISIBLE_DEVICES=1 python -m src.preprocess --frames 8 --max-side 640 --whisper small
                                                       # -> data/videos/frames/, processed/*.json
```

### 2. Run the experiments

`main.py` is the entry point. It starts the vLLM server if none answers on `--base_url`, checks
that processed videos exist (or builds the pool with `--prepare_data`), runs every
condition x framing x seed episode concurrently, and aggregates the results.

```bash
python main.py --dry_run                                   # list planned episodes
python main.py                                             # 4 conditions x {neutral, human} x 5 seeds, 100 steps
python main.py -c video boring -f human -s 10 -n 50        # subset
python main.py --prepare_data                              # also fetch + preprocess videos first
python main.py --help
```

Lower-level pieces can still be run on their own:

```bash
scripts/serve.sh Qwen/Qwen3-VL-8B-Instruct 0                # vLLM OpenAI API on :8000
python -m src.loop --condition video --framing human --steps 20 --seeds 0 1 --parallel 2
python -m src.analyze --run-name Qwen3-VL-8B-Instruct      # -> runs/<model>/summary.csv, plots/
```

Each run writes `runs/<model>/<condition>/seed<k>/steps.jsonl` with the full decision record per
step (task, video, action, stated reason, solution, grade).

## Layout

```
main.py              entry point: server, data check, grid of episodes, analysis
src/common.py        paths + jsonl helpers
src/fetch_videos.py  sample metadata shard, download with yt-dlp
src/preprocess.py    ffmpeg frame sampling + faster-whisper transcripts
src/tasks.py         task pool (GSM8K / MATH-500 / MMLU-Pro) with graders
src/loop.py          the N-step agent loop, all conditions
src/analyze.py       metrics, summary table, plots
scripts/serve.sh     vLLM server
```
