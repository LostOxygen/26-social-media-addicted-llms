"""Sample a pool of TikTok posts from the TikTok-10M metadata shard and download them.

Usage:
    python -m src.fetch_videos sample --n 300 --seed 0
    python -m src.fetch_videos download --workers 3
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from .common import MANIFEST_FILE, META_DIR, POOL_FILE, RAW_DIR, read_jsonl, write_jsonl

KEEP_COLS = [
    "id", "desc", "challenges", "duration", "create_time", "play_count", "digg_count",
    "comment_count", "share_count", "collect_count", "music_title", "music_author_name",
    "user_verified", "url",
]


def _load_meta(shard: Path) -> pd.DataFrame:
    table = pq.read_table(shard, columns=KEEP_COLS + ["is_ad"])
    df = table.to_pandas()
    df["challenges"] = df["challenges"].apply(
        lambda s: json.loads(s) if isinstance(s, str) else []
    )
    return df


def cmd_sample(args: argparse.Namespace) -> None:
    shards = sorted(META_DIR.glob("data/*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shard under {META_DIR}/data")
    df = pd.concat([_load_meta(s) for s in shards], ignore_index=True)
    n_all = len(df)
    min_ts = int(datetime(args.min_year, 1, 1, tzinfo=timezone.utc).timestamp())
    df = df[
        (df["is_ad"] == "f")
        & (df["duration"].between(args.min_dur, args.max_dur))
        & (df["create_time"] >= min_ts)
        & (df["play_count"] >= args.min_plays)
        & (df["desc"].fillna("").str.len() > 0)
    ]
    print(f"{n_all} rows -> {len(df)} after filters")
    # Weighted towards popular posts (log plays) so the feed resembles a real For-You page.
    weights = (df["play_count"].clip(lower=1)).apply(lambda x: 1 + __import__("math").log10(x))
    sample = df.sample(n=min(args.n, len(df)), weights=weights, random_state=args.seed)
    rows = []
    for r in sample.itertuples(index=False):
        d = r._asdict()
        d["id"] = str(d["id"])
        d["page_url"] = f"https://www.tiktok.com/@_/video/{d['id']}"
        for k in ("duration", "create_time", "play_count", "digg_count", "comment_count",
                  "share_count", "collect_count"):
            d[k] = int(d[k]) if pd.notna(d[k]) else None
        d.pop("is_ad", None)
        rows.append(d)
    write_jsonl(POOL_FILE, rows)
    print(f"wrote {len(rows)} candidates to {POOL_FILE}")
    print(sample["duration"].describe())


def _download_one(row: dict, out_dir: Path) -> dict:
    import yt_dlp  # noqa: WPS433  (heavy import kept local)

    vid = row["id"]
    target = out_dir / f"{vid}.mp4"
    rec = {"id": vid, "page_url": row["page_url"], "path": str(target)}
    if target.exists() and target.stat().st_size > 0:
        rec["status"] = "exists"
        return rec
    opts = {
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "format": "bv*[height<=720][ext=mp4]+ba/b[height<=720]/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "retries": 2,
        "socket_timeout": 30,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(row["page_url"], download=True)
        rec["status"] = "ok"
        rec["yt_title"] = info.get("title")
        rec["yt_duration"] = info.get("duration")
        rec["uploader"] = info.get("uploader")
        rec["view_count"] = info.get("view_count")
        rec["like_count"] = info.get("like_count")
    except Exception as e:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = str(e).splitlines()[0][:300]
    return rec


def cmd_download(args: argparse.Namespace) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pool = list(read_jsonl(POOL_FILE))
    done = {r["id"] for r in read_jsonl(MANIFEST_FILE) if r.get("status") in ("ok", "exists")}
    todo = [r for r in pool if r["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(pool)} in pool, {len(done)} already fetched, {len(todo)} to try")
    by_id = {r["id"]: r for r in pool}
    stats = {"ok": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex, MANIFEST_FILE.open("a") as mf:
        futs = []
        for r in todo:
            futs.append(ex.submit(_download_one, r, RAW_DIR))
            time.sleep(args.delay)  # stagger requests
        for fut in tqdm(as_completed(futs), total=len(futs)):
            rec = fut.result()
            rec["meta"] = by_id[rec["id"]]
            stats["ok" if rec["status"] in ("ok", "exists") else "error"] += 1
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mf.flush()
    print(stats)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--min-dur", type=int, default=5)
    s.add_argument("--max-dur", type=int, default=90)
    s.add_argument("--min-year", type=int, default=2024)
    s.add_argument("--min-plays", type=int, default=2000)
    s.set_defaults(fn=cmd_sample)
    d = sub.add_parser("download")
    d.add_argument("--workers", type=int, default=3)
    d.add_argument("--delay", type=float, default=0.5)
    d.add_argument("--limit", type=int, default=0)
    d.set_defaults(fn=cmd_download)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    random.seed(0)
    main()
