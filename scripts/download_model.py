import argparse
import logging

from huggingface_hub import snapshot_download


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%b %d, %Y %I:%M:%S%p",
    force=True,
)

def main(args: argparse.Namespace):
    logging.info(f"Downloading model from Hugging Face Hub: {args.repo_id}")
    snapshot_download(repo_id=args.repo_id)
    logging.info(f"Model {args.repo_id} successfully downloaded!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
    )
    args = parser.parse_args()
    main(args)

