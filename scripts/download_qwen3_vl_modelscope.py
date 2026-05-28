#!/usr/bin/env python3
"""Download Qwen3-VL from ModelScope to a local directory."""

from __future__ import annotations

import argparse
from pathlib import Path


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
    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    from modelscope.hub.snapshot_download import snapshot_download

    print(f"MODEL_SCOPE_DOWNLOAD_START model={args.model_id} local_dir={local_dir}", flush=True)
    model_dir = snapshot_download(
        model_id=args.model_id,
        cache_dir=str(local_dir.parent),
        local_dir=str(local_dir),
        local_files_only=False,
    )
    print(f"MODEL_SCOPE_DOWNLOAD_DONE path={model_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
