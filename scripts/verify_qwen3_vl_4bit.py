#!/usr/bin/env python3
"""Verify that Qwen3-VL can be loaded locally with 4-bit quantization."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/home/admin1/ct/Agentic-VLA/weights/Qwen3-VL-8B-Instruct",
        help="Local model directory to verify.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    print("VERIFY_OK", model.__class__.__name__, processor.__class__.__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
