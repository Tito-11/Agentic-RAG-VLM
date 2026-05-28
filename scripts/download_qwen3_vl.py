#!/usr/bin/env python3
"""Download Qwen3-VL to a local directory with explicit logging."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--local-dir",
        default="/home/admin1/ct/Agentic-VLA/weights/Qwen3-VL-8B-Instruct",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from huggingface_hub import snapshot_download

    print(f"DOWNLOAD_START model={args.model_id} local_dir={args.local_dir}", flush=True)
    path = snapshot_download(
        repo_id=args.model_id,
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"DOWNLOAD_DONE path={path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
