import argparse
import json
import logging
from argparse import Namespace

from datasets import load_dataset

from src.interp_embed import Dataset
from src.interp_embed.paper.clustering.algorithms import compute_clusters
from src.interp_embed.sae import GoodfireSAE, LocalSAE
from src.utils.path import build_experiment_results_filepath, build_dataset_filepath

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%b %d, %Y %I:%M:%S%p",
    force=True,
)


def main(args: Namespace):
    logging.info("".join([f"{k}={v}\n" for k, v in vars(args).items()]))

    goodfire = args.sae_type == "goodfire"

    kwargs = (
        {
            "variant_name": args.variant_name,
            "device": {"model": "auto", "sae": "cpu"},
        }
        if goodfire
        else {
            "release": args.release,
            "sae_id": args.sae_id,
        }
    )

    SAEClass = GoodfireSAE if goodfire else LocalSAE

    sae = SAEClass(**kwargs)
    df = load_dataset(args.dataset, "main", split=args.split).to_pandas()
    
    dataset_path = build_dataset_filepath(
        dataset=args.dataset,
        split=args.split,
        field=args.field,
        model=sae.name,
    )
    
    if dataset_path.exists():
        logging.info(f"Loading existing dataset from {dataset_path}")
        dataset = Dataset.load_from_file(dataset_path)
    else:
        dataset = Dataset(data=df, sae=sae, field="answer", save_path=dataset_path)
        dataset.save_to_file(dataset_path)
        logging.info(f"Saved embeddings for {len(df)} texts to {dataset_path}")

    clusters = compute_clusters(dataset, args.n_clusters)

    dataset_path = build_experiment_results_filepath(
        experiment="clustering",
        dataset=args.dataset,
        split=args.split,
        field="answer",
        model=sae.name,
        extension="json",
    )

    with open(dataset_path, "w") as f:
        json.dump(clusters, f)

    logging.info(f"Clusters saved to {dataset_path}")


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
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="answer",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
    )
    args = parser.parse_args()
    main(args)
