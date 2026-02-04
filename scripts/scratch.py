import argparse
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.path import resolve_model_snapshot

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%b %d, %Y %I:%M:%S%p",
    force=True,
)


def main(args: argparse.Namespace):
    snapshot = resolve_model_snapshot(args.model_name)
    logging.info(f"{args.model_name=}, {snapshot=}")

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        device_map="auto",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    devices = sorted(
        set(model.hf_device_map.values()),
        key=str,
    )

    devices_str = ", ".join(
        f"cuda:{d}" if isinstance(d, int) else str(d) for d in devices
    )

    # Check model device
    logging.info(f"Model is on device: {devices_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
    )
    args = parser.parse_args()
    main(args)
