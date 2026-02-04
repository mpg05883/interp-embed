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


def resolve_embeddings_dirpath(dataset: str) -> Path:
    """Resolve the path to results/embeddings/{dataset}, directory and create
    it if it doesn't exist."""
    dirpath = resolve_results_dirpath() / "embeddings" / dataset
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


def build_embeddings_filepath(
    dataset: str,
    split: str,
    field: str,
    model: str,
    extension: str = "pkl",
) -> Path:
    """Build the path to the embeddings file for a given dataset, split, field,
    model, and extension formatted as:

    results/embeddings/{dataset}/{split}/{field}/{model}.{extension}
    """
    filepath = (
        resolve_embeddings_dirpath(dataset) / split / field / f"{model}.{extension}"
    )
    filepath.parent.mkdir(parents=True, exist_ok=True)
    return filepath
