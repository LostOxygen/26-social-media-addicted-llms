"""Aggregate runs into per-run metrics, a summary table, and plots.

Usage:
    python -m src.analyze --run-name Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import RUNS, read_jsonl  # noqa: E402


def streaks(actions: list[bool]) -> list[int]:
    out, cur = [], 0
    for a in actions:
        if a:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def run_metrics(steps_path: Path) -> dict | None:
    steps = list(read_jsonl(steps_path))
    if not steps:
        return None
    distracted = [s["action"] != "solve" for s in steps]
    n = len(steps)
    solved = [s for s in steps if s["action"] == "solve"]
    st = streaks(distracted)
    # P(distract at t+1 | distract at t) and P(distract at t+1 | solve at t)
    stay = [(distracted[i], distracted[i + 1]) for i in range(n - 1)]
    p_dd = _cond(stay, True)
    p_sd = _cond(stay, False)
    first = next((s["step"] for s in steps if s["action"] != "solve"), None)
    by_diff = {}
    for d in ("easy", "medium", "hard"):
        sub = [s for s in steps if s["task_difficulty"] == d]
        if sub:
            by_diff[f"distract_rate_{d}"] = sum(s["action"] != "solve" for s in sub) / len(sub)
    cond = steps[0]["condition"] + "__" + steps[0].get("framing", "agent")
    return {
        "condition": cond, "seed": steps[0]["seed"], "n_steps": n,
        "frac_distracted": sum(distracted) / n,
        "n_solved": len(solved),
        "accuracy": (sum(s.get("correct", False) for s in solved) / len(solved)) if solved else None,
        "first_distraction_step": first,
        "max_streak": max(st) if st else 0,
        "mean_streak": (sum(st) / len(st)) if st else 0,
        "p_distract_after_distract": p_dd,
        "p_distract_after_solve": p_sd,
        "parse_errors": sum(s.get("parse_error", False) for s in steps),
        **by_diff,
    }


def _cond(pairs: list[tuple[bool, bool]], prev: bool) -> float | None:
    sub = [b for a, b in pairs if a == prev]
    return (sum(sub) / len(sub)) if sub else None


def analyze(run_name: str) -> pd.DataFrame | None:
    """Aggregate all runs under runs/<run_name>; returns the per-condition summary table."""
    base = RUNS / run_name
    rows, curves = [], {}
    for steps_path in sorted(base.glob("*/seed*/steps.jsonl")):
        m = run_metrics(steps_path)
        if m:
            rows.append(m)
            curves.setdefault(m["condition"], []).append(
                [s["action"] != "solve" for s in read_jsonl(steps_path)])
    if not rows:
        print(f"no runs under {base}")
        return None
    df = pd.DataFrame(rows)
    df.to_csv(base / "runs.csv", index=False)
    agg = df.groupby("condition").agg(
        seeds=("seed", "count"), frac_distracted=("frac_distracted", "mean"),
        frac_sd=("frac_distracted", "std"), n_solved=("n_solved", "mean"),
        accuracy=("accuracy", "mean"), max_streak=("max_streak", "mean"),
        p_dd=("p_distract_after_distract", "mean"), p_sd=("p_distract_after_solve", "mean"),
        first_distraction=("first_distraction_step", "mean"),
    ).round(3)
    agg.to_csv(base / "summary.csv")
    pd.set_option("display.width", 200)
    print(agg.to_string())

    plots = base / "plots"
    plots.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for cond, runs in curves.items():
        n = min(len(r) for r in runs)
        mat = pd.DataFrame([r[:n] for r in runs])
        ax.plot(range(1, n + 1), mat.mean(0).rolling(5, min_periods=1).mean(), label=cond)
    ax.set_xlabel("step")
    ax.set_ylabel("P(distracted), 5-step rolling mean")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "distraction_over_time.png", dpi=150)

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * df["condition"].nunique()), 4))
    df.boxplot(column="frac_distracted", by="condition", ax=ax, rot=30)
    ax.set_ylabel("fraction of steps distracted")
    ax.set_title("")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(plots / "frac_distracted_by_condition.png", dpi=150)
    print(f"wrote {base/'summary.csv'} and plots to {plots}")
    return agg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    analyze(p.parse_args().run_name)


if __name__ == "__main__":
    main()
