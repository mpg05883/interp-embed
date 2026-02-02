import argparse
import logging
from argparse import Namespace
from pathlib import Path

import pandas as pd

from examples.functions import diff_features
from src.interp_embed import Dataset
from src.interp_embed.sae import LocalSAE
from src.utils.path import resolve_results_dirpath

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%b %d, %Y %I:%M:%S%p",
    force=True,
)


def main(args: Namespace):
    logging.info(f"Loading SAE: {args.sae_id=}, Release: {args.release=}")

    # 1. Load a SAE supported through the SAELens package
    sae = LocalSAE(
        sae_id=args.sae_id,
        release=args.release,
    )

    # 2. Prepare your data as a DataFrame
    df1 = pd.DataFrame(
        {
            "text": ["Good morning!", "Hello there!", "Good afternoon."],
            "date": ["2022-01-10", "2021-08-23", "2023-03-14"],  # Metadata column
        }
    )

    # 2. Prepare your data as a DataFrame
    df2 = pd.DataFrame(
        {
            "text": ["See you later!", "Goodbye!", "Goodbye."],
            "date": ["2022-01-10", "2021-08-23", "2023-03-14"],  # Metadata column
        }
    )

    # 3. Create dataset - computes and saves feature activations
    dataset1 = Dataset(
        data=df1,
        sae=sae,
    )

    dataset2 = Dataset(
        data=df2,
        sae=sae,
    )

    freq = diff_features(dataset1, dataset2)
    print(freq.head())

    out_path = resolve_results_dirpath() / "demo" / f"{sae.name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    freq.to_csv(out_path, index=False)
    logging.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        type=str,
        default="gpt2-small-res-jb",
    )
    parser.add_argument(
        "--sae_id",
        type=str,
        default="blocks.8.hook_resid_pre",
    )
    args = parser.parse_args()
    main(args)
