import argparse
import logging
from argparse import Namespace

import pandas as pd
from notebooks.functions import diff_features

from src.interp_embed import Dataset
from src.interp_embed.sae import GoodfireSAE, LocalSAE
from src.utils.path import resolve_results_dirpath

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%b %d, %Y %I:%M:%S%p",
    force=True,
)


def main(args: Namespace):
    logging.info("".join([f"{k}={v}\n" for k, v in vars(args).items()]))

    # 1. Load a Goodfire SAE or SAE supported through the SAELens package
    goodfire = args.sae_type == "goodfire"

    kwargs = (
        {
            "variant_name": args.variant_name,
            "device": {"model": "auto", "sae": "cuda:0"},
        }
        if goodfire
        else {
            "release": args.release,
            "sae_id": args.sae_id,
        }
    )

    SAEClass = GoodfireSAE if goodfire else LocalSAE

    sae = SAEClass(**kwargs)

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
    parser.add_argument(
        "--variant_name",
        type=str,
        default="Llama-3.3-70B-Instruct-SAE-l50",
    )
    parser.add_argument(
        "--sae_type",
        type=str,
        choices=["local", "goodfire"],
        default="goodfire",
    )
    args = parser.parse_args()
    main(args)
