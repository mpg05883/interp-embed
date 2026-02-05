import argparse
import logging
import pickle
from argparse import Namespace

from datasets import load_dataset

from src.utils.openai import compute_embeddings, load_openai_client
from src.utils.path import build_dataset_filepath

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%b %d, %Y %I:%M:%S%p",
    force=True,
)


def main(args: Namespace):
    client = load_openai_client()
    dataset = load_dataset(args.dataset, "main", split=args.split)
    df = dataset.to_pandas()

    output_path = build_dataset_filepath(
        dataset=args.dataset,
        split=args.split,
        field=args.field,
        model=args.model,
    )

    if output_path.exists():
        with output_path.open("rb") as f:
            data = pickle.load(f)["data"]

        if len(data) == len(dataset):
            logging.info(
                f"All {len(dataset)} embeddings for dataset: {args.dataset}, "
                f"split: {args.split}, field: {args.field} have already been "
                f"computed with model: {args.model} and saved to {output_path}\n "
                f"Ending now..."
            )
            return

    logging.info(
        f"Computing embeddings for dataset: {args.dataset}, split: {args.split}, "
        f"field: {args.field}, model: {args.model}, batch size: {args.batch_size}"
    )
    embeddings = compute_embeddings(
        client,
        df[args.field],
        model=args.model,
        batch_size=args.batch_size,
    )

    output = {
        "dataset": args.dataset,
        "split": args.split,
        "field": args.field,
        "model": args.model,
        "data": [
            {
                "idx": i,
                "text": text,
                "embedding": emb,
            }
            for i, (text, emb) in enumerate(zip(dataset, embeddings))
        ],
    }

    with open(output_path, "wb") as f:
        pickle.dump(output, f)
    logging.info(f"Saved embeddings for {len(embeddings)} texts to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--field", type=str, default="answer")
    parser.add_argument("--model", type=str, default="text-embedding-3-large")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    main(args)
