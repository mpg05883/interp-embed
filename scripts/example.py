import argparse

import pandas as pd
from functions import diff_features
from interp_embed.saes import GoodfireSAE

from interp_embed import Dataset


def main(args: argparse.Namespace):
    # 1. Load a Goodfire SAE or SAE supported through the SAELens package
    sae = GoodfireSAE(
        variant_name=args.variant_name,
        device="cuda:0",  # optional
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant_name",
        type=str,
        default="Llama-3.1-8B-Instruct-SAE-l19",
        # or "Llama-3.3-70B-Instruct-SAE-l50" for higher quality features
    )
    args = parser.parse_args()
    main(args)
