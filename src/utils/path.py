from pathlib import Path


def resolve_results_dirpath() -> Path:
    """Resolve the path to the results directory at the project root."""
    return Path(__file__).resolve().parent.parent.parent / "results"


def resolve_embeddings_dirpath(dataset: str) -> Path:
    """Resolve the path to results/embeddings/{dataset}, directory and create
    it if it doesn't exist."""
    dirpath = resolve_results_dirpath() / "embeddings" / dataset
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


def build_embeddings_path(
    dataset: str,
    split: str,
    field: str,
    model: str,
    extension: str = "pkl",
) -> Path:
    """Build the path to the embeddings file for a given dataset, split, field,
    model, and extension."""
    filepath = (
        resolve_embeddings_dirpath(dataset) / split / field / f"{model}.{extension}"
    )
    filepath.parent.mkdir(parents=True, exist_ok=True)
    return filepath
