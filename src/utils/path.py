from pathlib import Path


def resolve_results_dirpath() -> Path:
    """Resolve the path to the results directory at the project root."""
    return Path(__file__).resolve().parent.parent.parent / "results"


def build_experiment_results_filepath(
    experiment: str,
    dataset: str,
    split: str,
    field: str,
    model: str,
    extension: str,
) -> Path:
    """Build the path to the experiment results file for a given experiment,
    dataset, split, field, model, and extension formatted as:

    results/{experiment}/{dataset}/{split}/{field}/{model}.{extension}

    If the file already exists, it will return the existing path.
    """
    filepath = (
        resolve_results_dirpath()
        / experiment
        / dataset
        / split
        / field
        / f"{model}.{extension}"
    )
    filepath.parent.mkdir(parents=True, exist_ok=True)
    return filepath


def resolve_datasets_dirpath(dataset: str) -> Path:
    """Resolve the path to results/datasets/{dataset}, directory and create
    it if it doesn't exist."""
    dirpath = resolve_results_dirpath() / "datasets" / dataset
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


def build_dataset_filepath(
    dataset: str,
    split: str,
    field: str,
    model: str,
    extension: str = "pkl",
) -> Path:
    """Build the path to the datasets file for a given dataset, split, field,
    model, and extension formatted as:

    results/datasets/{dataset}/{split}/{field}/{model}.{extension}
    """
    filepath = (
        resolve_datasets_dirpath(dataset) / split / field / f"{model}.{extension}"
    )
    filepath.parent.mkdir(parents=True, exist_ok=True)
    return filepath


def resolve_model_snapshot(
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
) -> Path:
    hf_dirpath = Path("/scratch/bcqc/mgee2/hf/hub/")

    # Default to Meta Llama-3.3-70B-Instruct if no model name is provided
    default_model_dir = "models--meta-llama--Llama-3.3-70B-Instruct"

    return {
        "meta-llama/Llama-3.3-70B-Instruct": hf_dirpath
        / "models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b",
    }.get(model_name, hf_dirpath / default_model_dir)
    