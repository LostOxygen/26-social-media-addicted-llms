"""Turn downloaded videos into model-ready inputs: sampled frames + Whisper transcript.

Usage:
    python -m src.preprocess --frames 8 --max-side 640 --whisper small
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from tqdm import tqdm

from .common import FRAMES_DIR, MANIFEST_FILE, PROCESSED_DIR, read_jsonl


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def extract_frames(path: Path, out_dir: Path, n_frames: int, max_side: int) -> list[str]:
    """Uniformly sample n_frames over the clip, longest side scaled to max_side."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = probe_duration(path)
    times = [(i + 0.5) * dur / n_frames for i in range(n_frames)]
    scale = f"scale='if(gt(iw,ih),{max_side},-2)':'if(gt(iw,ih),-2,{max_side})'"
    paths = []
    for i, t in enumerate(times):
        fp = out_dir / f"f{i:02d}.jpg"
        if not fp.exists():
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(path),
                 "-frames:v", "1", "-vf", scale, "-q:v", "3", str(fp)],
                check=True,
            )
        paths.append(str(fp))
    return paths


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--max-side", type=int, default=640)
    p.add_argument("--whisper", default="small", help="faster-whisper model size, or 'none'")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    recs = [r for r in read_jsonl(MANIFEST_FILE) if r.get("status") in ("ok", "exists")]
    seen = set()
    recs = [r for r in recs if not (r["id"] in seen or seen.add(r["id"]))]
    if args.limit:
        recs = recs[: args.limit]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    whisper = None
    if args.whisper != "none":
        from faster_whisper import WhisperModel
        whisper = WhisperModel(args.whisper, device=args.device, compute_type="float16")

    n_ok = 0
    for r in tqdm(recs):
        out = PROCESSED_DIR / f"{r['id']}.json"
        if out.exists():
            n_ok += 1
            continue
        path = Path(r["path"])
        if not path.exists():
            continue
        try:
            frames = extract_frames(path, FRAMES_DIR / r["id"], args.frames, args.max_side)
            dur = probe_duration(path)
            transcript, lang = "", None
            if whisper is not None:
                segments, info = whisper.transcribe(str(path), beam_size=3, vad_filter=True)
                transcript = " ".join(s.text.strip() for s in segments).strip()
                lang = info.language
            meta = r["meta"]
            doc = {
                "id": r["id"],
                "path": str(path),
                "duration": dur,
                "frames": frames,
                "transcript": transcript,
                "language": lang,
                "caption": meta.get("desc", ""),
                "hashtags": meta.get("challenges", []),
                "music": meta.get("music_title"),
                "play_count": meta.get("play_count"),
                "digg_count": meta.get("digg_count"),
                "uploader": r.get("uploader"),
            }
            out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[{r['id']}] failed: {e}")
    print(f"processed {n_ok}/{len(recs)} videos -> {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
