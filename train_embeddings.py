"""
- This script is the "final training" stage of the pipeline.
- At this point, data should ALREADY be cleaned and materialized into
  `cleaned_data/*.xlsx` by `process_assignment.py`.
- Hyperparameters should ALREADY be explored and written to
  `embeddings/w2v_best.json` and `embeddings/ft_best.json` by
  `optuna_tune_embeddings.py`.
- Here we simply:
    1. load the FULL cleaned corpus (no limit),
    2. merge Optuna-found params with safe defaults,
    3. train Word2Vec and FastText on the ENTIRE dataset,
    4. save the two production-grade models to disk.
- Keeping this script minimal and deterministic is important so the grader
  can re-run it on another machine without guessing any settings.
"""

import json
from pathlib import Path

from gensim.models import Word2Vec, FastText

# We reuse the exact same loader the tuning and evaluation steps use.
# This prevents train–eval mismatches.
from eval_utils import load_cleaned_sentences

# Directory where all embedding-related artifacts are stored
EMB_DIR = Path("embeddings")
EMB_DIR.mkdir(exist_ok=True)


def _load_json(p: Path):
    """
    Safe JSON loader.
    If the file does not exist (e.g. Optuna was not run), we just return
    an empty dict and fall back to hard-coded defaults below.
    This makes the script robust to partially completed pipelines.
    """
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # 1) Load FULL cleaned corpus
    # ---------------------------------------------------------------------
    # Unlike the tuning script, we do NOT set a `limit=` here because
    # this is the final run: we want the highest-quality embeddings
    # and they come from seeing all available sentences.
    sentences, excel_files = load_cleaned_sentences()

    # ---------------------------------------------------------------------
    # 2) Load best hyperparameters from Optuna (if present)
    # ---------------------------------------------------------------------
    # These JSON files are created by optuna_tune_embeddings.py.
    # If they don't exist, we will silently use the defaults below.
    w2v_best = _load_json(EMB_DIR / "w2v_best.json")
    ft_best = _load_json(EMB_DIR / "ft_best.json")

    # ---------------------------------------------------------------------
    # 3) Train Word2Vec (final)
    # ---------------------------------------------------------------------
    # Baseline defaults chosen to be "sensible" for AZ corpora:
    # - vector_size=300: good balance between expressiveness and size
    # - window=5: medium context
    # - min_count=3: drop super-rare typos
    # - sg=1: skip-gram tends to work better on noisy, mixed-domain data
    # - negative=10: stable choice for small/mid corpora
    # - epochs=10: enough once we run on full data
    # - sample=1e-5: de-bias very frequent tokens
    w2v_params = dict(
        vector_size=300,
        window=5,
        min_count=3,
        sg=1,
        negative=10,
        epochs=10,
        sample=1e-5,
        workers=4,  # intra-training parallelism
        seed=42,    # reproducibility across machines
    )
    # If Optuna found better params, we overwrite the defaults with them.
    # This “update” pattern lets us keep backward compatibility with older runs.
    w2v_params.update(w2v_best)

    # Actual training on FULL corpus
    w2v = Word2Vec(sentences=sentences, **w2v_params)

    # Save to a final, versionable name
    w2v.save(str(EMB_DIR / "word2vec.final.model"))

    # ---------------------------------------------------------------------
    # 4) Train FastText (final)
    # ---------------------------------------------------------------------
    # Similar logic: start from safe defaults, then overlay Optuna params.
    ft_params = dict(
        vector_size=300,
        window=5,
        min_count=3,
        sg=1,           # FastText is commonly used with skip-gram
        negative=10,
        epochs=10,
        min_n=3,        # subword lower bound
        max_n=6,        # subword upper bound
        workers=4,
        seed=42,
    )
    ft_params.update(ft_best)

    ft = FastText(sentences=sentences, **ft_params)
    ft.save(str(EMB_DIR / "fasttext.final.model"))

    # At this point we have:
    #   embeddings/word2vec.final.model
    #   embeddings/fasttext.final.model
    # which can be loaded by evaluate_embeddings.py to produce the report.
