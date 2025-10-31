"""
- This script is intentionally tiny and "read-only".
- It assumes that:
    1. the data cleaning pipeline has already been executed
       (i.e. `process_assignment.py` has produced `cleaned_data/*.xlsx`),
    2. the final embedding models have already been trained
       (i.e. `train_embeddings.py` has produced
        `embeddings/word2vec.final.model` and/or
        `embeddings/fasttext.final.model`).

- Its single responsibility is to:
    - load the available models,
    - evaluate them with the **same** metric used during Optuna tuning,
    - print the results in a format that can be pasted directly into the report.

- Because we reuse `eval_utils.evaluate_model(...)`, we guarantee that
  tuning-time scores and report-time scores are perfectly comparable.
  This is the key to a clean, auditable assignment pipeline.
"""

from pathlib import Path
from gensim.models import Word2Vec, FastText

# We reuse the unified loader and evaluator
from eval_utils import load_cleaned_sentences, evaluate_model


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1) Load the same cleaned Excel files we trained on
    # ------------------------------------------------------------------
    # We do NOT set a limit here, because for evaluation we want to see
    # coverage over the entire assignment dataset.
    _, excel_files = load_cleaned_sentences()

    # Final model locations (produced by train_embeddings.py)
    w2v_path = Path("embeddings/word2vec.final.model")
    ft_path = Path("embeddings/fasttext.final.model")

    # ------------------------------------------------------------------
    # 2) Evaluate Word2Vec (if present)
    # ------------------------------------------------------------------
    if w2v_path.exists():
        # Load the trained model from disk
        w2v = Word2Vec.load(str(w2v_path))

        # Run the shared evaluation: returns dict with
        # {syn_mean, ant_mean, separation, coverage}
        metrics = evaluate_model(w2v, excel_files)

        # Print in a JSON-like format so it is easy to copy into README.md
        print("Word2Vec:", metrics)

    # ------------------------------------------------------------------
    # 3) Evaluate FastText (if present)
    # ------------------------------------------------------------------
    if ft_path.exists():
        ft = FastText.load(str(ft_path))
        metrics = evaluate_model(ft, excel_files)
        print("FastText:", metrics)
