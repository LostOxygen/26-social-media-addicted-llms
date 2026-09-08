"""Benchmark task pool: single-turn tasks with automatic grading.

Sources: GSM8K (easy), MATH-500 (medium), MMLU-Pro (hard, 10-way multiple choice).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable

from datasets import load_dataset

LETTERS = "ABCDEFGHIJ"


@dataclass
class Task:
    id: str
    source: str
    difficulty: str
    prompt: str
    answer: str
    grade: Callable[[str], bool] = field(repr=False)
    meta: dict = field(default_factory=dict)

    def short(self, n: int = 160) -> str:
        text = " ".join(self.prompt.split())
        return text if len(text) <= n else text[: n - 1] + "…"


def _strip_wrappers(s: str) -> str:
    s = s.strip()
    for _ in range(3):
        s = s.strip().strip("*").strip("$").strip().rstrip(".").strip()
    m = re.fullmatch(r"\\boxed\{(.*)\}", s)
    if m:
        s = m.group(1)
    return s.strip()


def _norm_math(s: str) -> str:
    s = _strip_wrappers(s)
    s = re.sub(r"\\(left|right|,|!|;)", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"\\frac{\1}{\2}", s)  # \frac 59 -> \frac{5}{9}
    s = s.replace(" ", "").replace("^{\\circ}", "").replace("^\\circ", "")
    s = re.sub(r"\\boxed\{(.*)\}", r"\1", s)
    s = s.replace("\\%", "%").rstrip("%")
    s = s.rstrip(".")
    return s


def _norm_number(s: str) -> str | None:
    s = _strip_wrappers(s).replace(",", "").replace("$", "").strip()
    s = re.sub(r"\\boxed\{(.*)\}", r"\1", s)
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    v = float(m.group())
    return str(int(v)) if v == int(v) else f"{v:g}"


def _gsm8k_task(i: int, row: dict) -> Task:
    gold = row["answer"].split("####")[-1].strip()
    gold_n = _norm_number(gold)
    return Task(
        id=f"gsm8k-{i}", source="gsm8k", difficulty="easy",
        prompt=row["question"].strip() + "\nGive the final numeric answer.",
        answer=gold,
        grade=lambda pred, g=gold_n: _norm_number(pred) == g,
    )


def _math500_task(i: int, row: dict) -> Task:
    gold = row["answer"]
    gold_n = _norm_math(gold)
    return Task(
        id=f"math500-{i}", source="math500", difficulty="medium",
        prompt=row["problem"].strip() + "\nGive the final answer in simplest form (LaTeX ok).",
        answer=gold,
        grade=lambda pred, g=gold_n: _norm_math(pred) == g,
        meta={"level": row["level"], "subject": row["subject"]},
    )


def _mmlupro_task(i: int, row: dict) -> Task:
    opts = "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(row["options"]))
    gold = row["answer"].strip().upper()

    def grade(pred: str, g: str = gold) -> bool:
        m = re.search(r"\b([A-J])\b", pred.strip().upper())
        return bool(m) and m.group(1) == g

    return Task(
        id=f"mmlupro-{row['question_id']}", source="mmlu_pro", difficulty="hard",
        prompt=row["question"].strip() + "\n" + opts + "\nAnswer with the letter only.",
        answer=gold, grade=grade, meta={"category": row["category"]},
    )


def build_pool(n_per_source: int, seed: int, sources: tuple[str, ...] = ("gsm8k", "math500",
                                                                          "mmlu_pro")) -> list[Task]:
    """Interleaved list of tasks, deterministic for a seed."""
    rng = random.Random(seed)
    per_source: dict[str, list[Task]] = {}
    if "gsm8k" in sources:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        idx = rng.sample(range(len(ds)), n_per_source)
        per_source["gsm8k"] = [_gsm8k_task(i, ds[i]) for i in idx]
    if "math500" in sources:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        idx = rng.sample(range(len(ds)), n_per_source)
        per_source["math500"] = [_math500_task(i, ds[i]) for i in idx]
    if "mmlu_pro" in sources:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        idx = rng.sample(range(len(ds)), n_per_source)
        per_source["mmlu_pro"] = [_mmlupro_task(i, ds[i]) for i in idx]
    pool: list[Task] = []
    lists = list(per_source.values())
    for k in range(n_per_source):
        block = [lst[k] for lst in lists]
        rng.shuffle(block)
        pool.extend(block)
    return pool


FINAL_RE = re.compile(r"FINAL ANSWER", re.IGNORECASE)


def extract_final(text: str) -> str:
    """Answer after the last 'FINAL ANSWER' marker, tolerant to markdown/LaTeX decoration.

    Handles '**FINAL ANSWER:** 42', 'FINAL ANSWER: $$\n42\n$$', '\\boxed{42}**', etc.
    Returns '' when no marker is present.
    """
    hits = list(FINAL_RE.finditer(text))
    if not hits:
        return ""
    rest = text[hits[-1].end():]
    rest = re.sub(r"^[\s:：*]+", "", rest)
    lines = [l.strip() for l in rest.splitlines()]
    # skip lines that are only decoration ('$$', '**', '', '```')
    for line in lines:
        core = re.sub(r"[\s$*`]", "", line)
        if core:
            return _strip_wrappers(line)
    return ""
