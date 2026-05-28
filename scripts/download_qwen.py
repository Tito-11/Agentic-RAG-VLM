import argparse
import logging
import os
import pathlib

from huggingface_hub import snapshot_download


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("QwenDownloader")

DEFAULT_REPO_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_LOCAL_DIR = pathlib.Path(__file__).resolve().parents[1] / "weights" / "Qwen3-VL-8B-Instruct"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the real Qwen3-VL checkpoint from Hugging Face.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo id.")
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL_DIR), help="Destination directory.")
    parser.add_argument("--revision", default=None, help="Optional revision/commit.")
    return parser.parse_args()


def download_qwen(repo_id: str, local_dir: str, revision: str | None = None) -> str:
    local_path = pathlib.Path(local_dir).expanduser().resolve()
    local_path.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s", repo_id, local_path)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_path),
        revision=revision,
        max_workers=1,
    )
    logger.info("Download complete: %s", local_path)
    return str(local_path)


if __name__ == "__main__":
    args = _parse_args()
    try:
        download_qwen(args.repo_id, args.local_dir, args.revision)
    except Exception as exc:
        logger.error("Failed to download Qwen model: %s", exc)
        raise SystemExit(1)
