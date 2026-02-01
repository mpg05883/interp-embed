import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


def load_openai_client(env_var: str = "OPENAI_KEY") -> OpenAI:
    """Load OpenAI client using API key from .env."""
    load_dotenv()
    api_key = os.getenv(env_var)

    if api_key is None:
        raise RuntimeError(f"{env_var} not found in environment or .env")

    return OpenAI(api_key=api_key)


def compute_embeddings(
    client: OpenAI,
    texts: list[str] | pd.Series,
    model: str = "text-embedding-3-large",
    batch_size: int = 256,
    verbose: bool = True,
) -> list[list[float]]:
    """
    Compute embeddings for a list of texts using OpenAI's API. Returns
    embeddings in the same order as texts.
    """
    if isinstance(texts, pd.Series):
        texts = texts.tolist()

    embeddings = []

    kwargs = {
        "desc": "Computing embeddings",
        "total": len(texts),
        "unit": "texts",
        "disable": not verbose,
    }

    for i in tqdm(range(0, len(texts), batch_size), **kwargs):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch,
        )
        embeddings.extend([item.embedding for item in response.data])

    assert len(embeddings) == len(texts)
    return embeddings
