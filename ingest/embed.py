"""Step 1.4: the embedding model (bge small or bge large) and the token counter.

Both come from the same model name in config.py (EMBED_PROFILE: small or large), so the token counts match
what the model sees. The first run downloads the model (small: about 130 MB, large: about 1.3 GB).
The model runs on the GPU when the machine has one (Colab) and on the CPU
otherwise (a laptop). The vectors are the same either way.
Nothing is loaded until a function is called, so the tests stay fast.
"""
import os
from functools import lru_cache

# Quiet two harmless warnings on Windows. Set before the libraries load.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import config  # noqa: E402

BATCH = 32  # texts per model call


def _name(model):
    """The model to use: the one given, else the one of config.py (EMBED_PROFILE)."""
    return model or config.EMBED_MODEL


@lru_cache(maxsize=2)
def _tokenizer(name):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.model_max_length = 10**6  # we only count here, so no "longer than 512" warning
    return tokenizer


def count_tokens(texts, model=None):
    """The number of tokens of each text, without the two special start and end tokens."""
    tokenizer = _tokenizer(_name(model))
    sizes = []
    for i in range(0, len(texts), 256):
        encoded = tokenizer(list(texts[i : i + 256]), add_special_tokens=False)["input_ids"]
        sizes.extend(len(ids) for ids in encoded)
    return sizes


@lru_cache(maxsize=1)
def _device():
    """"cuda" when the machine has a GPU, "cpu" when it does not."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def _model(name):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=_device())


def model_on_disk(name=None):
    """True when the model is already downloaded on this computer (so nothing has to be downloaded to use it)."""
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(_name(name), "config.json"), str)
    except Exception:  # noqa: BLE001 - when in doubt the answer is no
        return False


def all_finite(vectors):
    """True when every number of every vector is a real number (no NaN, no infinity)."""
    import numpy as np

    return bool(np.isfinite(np.asarray(vectors, dtype="float64")).all())


def embed_texts(texts, model=None):
    """One vector (384 numbers for small, 1024 for large, length 1) for each text. Use it for pieces and for questions."""
    vectors = _model(_name(model)).encode(list(texts), batch_size=BATCH, normalize_embeddings=True, show_progress_bar=False)
    if not all_finite(vectors):
        raise RuntimeError(
            "The embedding model gave numbers that are not valid (NaN). Nothing was uploaded. "
            "Test the model with:  python scripts\\check_large.py --allow-download   or use EMBED_PROFILE = small."
        )
    return [vector.tolist() for vector in vectors]
