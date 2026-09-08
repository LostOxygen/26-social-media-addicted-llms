"""main hook to start the procrastination-loop experiments"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from src.analyze import analyze
from src.colors import TColors
from src.common import PROCESSED_DIR, ROOT, RUNS
from src.loop import CONDITIONS, FRAMINGS, run_seed, videos_for

PYTHON = sys.executable
SERVE_SCRIPT = ROOT / "scripts" / "serve.sh"
LOG_DIR = ROOT / "logs"
SERVER_TIMEOUT_S = 30 * 60


def server_ready(base_url: str) -> bool:
    """True if an OpenAI-compatible server answers on base_url"""
    try:
        return requests.get(f"{base_url}/models", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def ensure_server(model: str, gpu: int, base_url: str, serve: bool) -> subprocess.Popen | None:
    """start scripts/serve.sh for the model unless a server already answers.

    Returns the server process if this call started one, else None.
    """
    if server_ready(base_url):
        served = [m["id"] for m in requests.get(f"{base_url}/models", timeout=5).json()["data"]]
        if model not in served:
            print(f"{TColors.WARNING}server on {base_url} serves {served}, "
                  f"not {model}{TColors.ENDC}")
        else:
            print(f"{TColors.OKGREEN}using running server on {base_url}{TColors.ENDC}")
        return None
    if not serve:
        raise SystemExit(f"{TColors.FAIL}no server on {base_url} and --no_serve set{TColors.ENDC}")
    LOG_DIR.mkdir(exist_ok=True)
    log = (LOG_DIR / "vllm.log").open("a")
    print(f"{TColors.OKBLUE}starting vLLM for {model} on GPU {gpu} (log: logs/vllm.log)"
          f"{TColors.ENDC}")
    proc = subprocess.Popen([str(SERVE_SCRIPT), model, str(gpu)], stdout=log, stderr=log,
                            cwd=ROOT)
    t0 = time.time()
    while not server_ready(base_url):
        if proc.poll() is not None:
            raise SystemExit(f"{TColors.FAIL}vLLM exited with code {proc.returncode}, "
                             f"see logs/vllm.log{TColors.ENDC}")
        if time.time() - t0 > SERVER_TIMEOUT_S:
            proc.terminate()
            raise SystemExit(f"{TColors.FAIL}vLLM not ready after {SERVER_TIMEOUT_S}s{TColors.ENDC}")
        time.sleep(5)
    print(f"{TColors.OKGREEN}server ready after {time.time() - t0:.0f}s{TColors.ENDC}")
    return proc


def ensure_data(prepare: bool, n_videos: int, preprocess_gpu: int) -> None:
    """make sure processed videos exist; optionally run the fetch/preprocess pipeline"""
    n = len(list(PROCESSED_DIR.glob("*.json")))
    if n:
        print(f"{TColors.OKGREEN}{n} processed videos in {PROCESSED_DIR}{TColors.ENDC}")
        return
    if not prepare:
        raise SystemExit(
            f"{TColors.FAIL}no processed videos in {PROCESSED_DIR}. Re-run with --prepare_data "
            f"or follow README section 'Video pool'{TColors.ENDC}")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(preprocess_gpu)}
    stages = [
        [PYTHON, "-m", "src.fetch_videos", "sample", "--n", str(n_videos)],
        [PYTHON, "-m", "src.fetch_videos", "download"],
        [PYTHON, "-m", "src.preprocess"],
    ]
    for cmd in stages:
        print(f"{TColors.OKBLUE}$ {' '.join(cmd[2:])}{TColors.ENDC}")
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main(
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    conditions: list[str] | None = None,
    framings: list[str] | None = None,
    steps: int = 100,
    seeds: int = 5,
    parallel: int = 20,
    gpu: int = 0,
    base_url: str = "http://localhost:8000/v1",
    no_serve: bool = False,
    stop_server: bool = False,
    prepare_data: bool = False,
    n_videos: int = 300,
    preprocess_gpu: int = 1,
    feed_advance: str = "always",
    temperature: float = 0.7,
    top_p: float = 0.8,
    solve_tokens: int = 4096,
    tasks_per_source: int = 40,
    history: int = 100,
    no_feedback: bool = False,
    run_name: str = "",
    overwrite: bool = False,
    skip_analysis: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Main function to run the procrastination-loop grid: every condition x framing x seed is one
    episode of `steps` decisions; episodes run concurrently against one vLLM server.

    Args:
        model (str): HF id of the vision model to serve and query
        conditions (list[str]): subset of video, text_only, boring, instructed (default: all)
        framings (list[str]): subset of agent, neutral, human (default: neutral, human)
        steps (int): decisions per episode
        seeds (int): number of seeds per condition x framing (0 .. seeds-1)
        parallel (int): concurrently running episodes
        gpu (int): GPU for the vLLM server (only used when this script starts one)
        base_url (str): OpenAI-compatible endpoint of the server
        no_serve (bool): never start a server, fail if none answers on base_url
        stop_server (bool): terminate the server at the end if this script started it
        prepare_data (bool): fetch and preprocess the video pool if data/videos/processed is empty
        n_videos (int): candidates to sample when preparing data
        preprocess_gpu (int): GPU for Whisper when preparing data
        feed_advance (str): always (new video every step) or on_watch (video stays until watched)
        temperature (float): sampling temperature for decision and solver calls
        top_p (float): nucleus sampling for decision and solver calls
        solve_tokens (int): max tokens for the solver call
        tasks_per_source (int): tasks drawn per benchmark (pool = 3 x this)
        history (int): number of past-step lines shown to the agent
        no_feedback (bool): hide correct/incorrect feedback from the agent
        run_name (str): directory under runs/ (default: model short name)
        overwrite (bool): redo episodes that already have a summary.json
        skip_analysis (bool): do not aggregate/plot at the end
        dry_run (bool): only list the planned episodes
    """
    conditions = conditions or list(CONDITIONS)
    framings = framings or ["neutral", "human"]
    run_name = run_name or model.split("/")[-1]
    seed_list = list(range(seeds))
    start = datetime.datetime.now()

    print(f"{TColors.HEADER}{TColors.BOLD}Procrastination loop{TColors.ENDC} "
          f"{start:%Y-%m-%d %H:%M}")
    print(f"  model={model}  steps={steps}  seeds={seed_list}  parallel={parallel}")
    print(f"  conditions={conditions}  framings={framings}  feed_advance={feed_advance}")
    print(f"  runs -> {RUNS / run_name}")
    episodes = [(f, c, s) for f in framings for c in conditions for s in seed_list]
    if dry_run:
        for f, c, s in episodes:
            print(f"  {c}__{f}/seed{s}")
        print(f"{len(episodes)} episodes planned")
        return

    needs_videos = any(c != "boring" for c in conditions)
    if needs_videos:
        ensure_data(prepare_data, n_videos, preprocess_gpu)
    server = ensure_server(model, gpu, base_url, serve=not no_serve)

    def cfg_for(condition: str, framing: str) -> argparse.Namespace:
        return argparse.Namespace(
            condition=condition, framing=framing, steps=steps, seeds=seed_list,
            parallel=parallel, model=model, base_url=base_url, run_name=run_name,
            temperature=temperature, top_p=top_p, solve_tokens=solve_tokens, history=history,
            tasks_per_source=tasks_per_source, feedback=not no_feedback, overwrite=overwrite,
            feed_advance=feed_advance,
        )

    videos = {c: videos_for(c) for c in conditions}
    results: list[dict] = []
    failures = 0
    try:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = {ex.submit(run_seed, cfg_for(c, f), s, videos[c]): (f, c, s)
                    for f, c, s in episodes}
            for fut in as_completed(futs):
                f, c, s = futs[fut]
                try:
                    r = fut.result()
                    results.append(r)
                    print(f"{TColors.OKGREEN}done{TColors.ENDC} {c}__{f}/seed{s}: "
                          f"distracted {r['frac_distracted']:.2f}, solved {r['n_solved']} "
                          f"({r['n_correct']} correct), {r['seconds']}s  "
                          f"[{len(results) + failures}/{len(episodes)}]")
                except Exception as e:  # noqa: BLE001
                    failures += 1
                    print(f"{TColors.FAIL}failed{TColors.ENDC} {c}__{f}/seed{s}: {e}")
    finally:
        if server is not None and stop_server:
            print(f"{TColors.OKBLUE}stopping vLLM server{TColors.ENDC}")
            server.terminate()

    (RUNS / run_name).mkdir(parents=True, exist_ok=True)
    (RUNS / run_name / "grid_results.json").write_text(json.dumps(results, indent=1))
    if not skip_analysis and results:
        print(f"\n{TColors.HEADER}Summary per condition{TColors.ENDC}")
        analyze(run_name)
    elapsed = datetime.datetime.now() - start
    color = TColors.OKGREEN if not failures else TColors.WARNING
    print(f"\n{color}{len(results)} episodes finished, {failures} failed, "
          f"{str(elapsed).split('.')[0]} elapsed{TColors.ENDC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Social-media-addicted LLMs: procrastination loop")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HF id of the vision model to serve and query",
    )
    parser.add_argument(
        "--conditions",
        "-c",
        type=str,
        nargs="+",
        choices=CONDITIONS,
        default=None,
        help="conditions to run (default: all four)",
    )
    parser.add_argument(
        "--framings",
        "-f",
        type=str,
        nargs="+",
        choices=FRAMINGS,
        default=None,
        help="system-prompt framings to run (default: neutral human)",
    )
    parser.add_argument(
        "--steps",
        "-n",
        type=int,
        default=100,
        help="decisions per episode (default: 100)",
    )
    parser.add_argument(
        "--seeds",
        "-s",
        type=int,
        default=5,
        help="number of seeds per condition x framing (default: 5)",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=20,
        help="concurrently running episodes; vLLM batches them (default: 20)",
    )
    parser.add_argument(
        "--gpu",
        "-g",
        type=int,
        default=0,
        help="GPU for the vLLM server when this script starts it (default: 0)",
    )
    parser.add_argument(
        "--base_url",
        "-u",
        type=str,
        default="http://localhost:8000/v1",
        help="OpenAI-compatible endpoint (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--no_serve",
        "-ns",
        action="store_true",
        help="do not start a server; fail if none answers on --base_url",
    )
    parser.add_argument(
        "--stop_server",
        "-ss",
        action="store_true",
        help="terminate the server at the end if this script started it",
    )
    parser.add_argument(
        "--prepare_data",
        "-pd",
        action="store_true",
        help="sample, download and preprocess the video pool if data/videos/processed is empty",
    )
    parser.add_argument(
        "--n_videos",
        "-nv",
        type=int,
        default=300,
        help="candidate videos to sample with --prepare_data (default: 300)",
    )
    parser.add_argument(
        "--preprocess_gpu",
        "-pg",
        type=int,
        default=1,
        help="GPU for Whisper with --prepare_data (default: 1)",
    )
    parser.add_argument(
        "--feed_advance",
        "-fa",
        type=str,
        choices=("always", "on_watch"),
        default="always",
        help="always: a new video every step; on_watch: the video stays until watched",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.7,
        help="sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top_p",
        "-tp",
        type=float,
        default=0.8,
        help="nucleus sampling (default: 0.8)",
    )
    parser.add_argument(
        "--solve_tokens",
        "-st",
        type=int,
        default=4096,
        help="max tokens for the solver call (default: 4096)",
    )
    parser.add_argument(
        "--tasks_per_source",
        "-ts",
        type=int,
        default=40,
        help="tasks drawn per benchmark, pool = 3 x this (default: 40)",
    )
    parser.add_argument(
        "--history",
        "-h_",
        type=int,
        default=100,
        help="past-step lines shown to the agent (default: 100)",
    )
    parser.add_argument(
        "--no_feedback",
        "-nf",
        action="store_true",
        help="hide correct/incorrect feedback from the agent",
    )
    parser.add_argument(
        "--run_name",
        "-r",
        type=str,
        default="",
        help="directory under runs/ (default: model short name)",
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="redo episodes that already have a summary.json",
    )
    parser.add_argument(
        "--skip_analysis",
        "-sa",
        action="store_true",
        help="do not aggregate and plot at the end",
    )
    parser.add_argument(
        "--dry_run",
        "-d",
        action="store_true",
        help="only list the planned episodes",
    )
    args = parser.parse_args()
    main(**vars(args))
