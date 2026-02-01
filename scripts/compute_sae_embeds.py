import argparse
import logging
from argparse import Namespace

from datasets import load_dataset

from src.interp_embed.dataset_analysis import Dataset
from src.interp_embed.sae.local_sae import LocalSAE
from src.interp_embed.utils.helpers import safe_load_pkl
from src.utils.path import build_embeddings_path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%b %d, %Y %I:%M:%S%p",
    force=True,
)


def main(args: Namespace):
    df = load_dataset(args.dataset, "main", split=args.split).to_pandas()
    sae = LocalSAE(sae_id=args.sae_id, release=args.release)

    output_path = build_embeddings_path(
        dataset=args.dataset,
        split=args.split,
        field=args.field,
        model=sae.name,
    )

    if output_path.exists():
        params = safe_load_pkl(output_path)
        
        # Count the number of successfully completed samples
        num_rows = sum(1 for row in params["rows"] if row is not None)

        if num_rows == len(df):
            logging.info(
                f"All {num_rows} embeddings for dataset: {args.dataset}, split: "
                f"{args.split}, field: {args.field} have already been computed "
                f"with model: {sae.name} and saved to {output_path}\n Ending now..."
            )
            return

    logging.info(
        f"Computing embeddings for dataset: {args.dataset}, split: {args.split}, "
        f"field: {args.field} with SAE: {sae.name}"
    )
    dataset = Dataset(
        data=df,
        sae=sae,
        field=args.field,
        save_path=output_path,
    )
    dataset.save_to_file(save_path=output_path)
    logging.info(f"Saved embeddings for {len(df)} texts to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--field", type=str, default="answer")
    parser.add_argument("--sae_id", type=str, default="blocks.8.hook_resid_pre")
    parser.add_argument("--release", type=str, default="gpt2-small-res-jb")
    args = parser.parse_args()
    main(args)
