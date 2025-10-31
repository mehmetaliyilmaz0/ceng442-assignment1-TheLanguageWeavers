"""
- Provide a SINGLE source of truth for:
    1. loading the cleaned Azerbaijani sentences produced by process_assignment.py
    2. computing semantic-quality metrics (synonym vs. antonym similarity)
    3. computing lexical coverage on the assignment’s 5 cleaned Excel files

Why this matters:
- Both the tuning script (optuna_tune_embeddings.py) and the final evaluation
  script (evaluate_embeddings.py) must use EXACTLY the same logic, otherwise
  we would "optimize for one metric and report another".
- Centralizing this here eliminates that class of bugs.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm

# ---------------------------------------------------------------------------
# 1) DEFAULT INPUTS
# ---------------------------------------------------------------------------
# We explicitly list the 5 cleaned Excel files that process_assignment.py
# is supposed to generate. If one of them is missing, we will simply skip it.
# This makes the code slightly more robust in classroom environments.
DEFAULT_FILES = [
    "cleaned_data/labeled-sentiment_2col.xlsx",
    "cleaned_data/test__1__2col.xlsx",
    "cleaned_data/train__3__2col.xlsx",
    "cleaned_data/train-00000-of-00001_2col.xlsx",
    "cleaned_data/merged_dataset_CSV__1__2col.xlsx",
]

# ---------------------------------------------------------------------------
# 2) PROBING PAIRS
# ---------------------------------------------------------------------------
# These lists define WHAT we will probe in the embedding space.
# They are task- / language-specific and can be extended later.
# We intentionally picked frequent, sentiment-relevant Azerbaijani items so
# that small corpora still produce meaningful cosine values.

# Azerbaijani synonym pairs → we expect HIGH cosine similarities
SYN_PAIRS = [
    ("yaxşı", "əla"),
    ("bahalı", "qiymətli"),
    ("ucuz", "sərfəli"),
    ("pis", "bərbad"),
    ("gözəl", "qəşəng"),
]

# Azerbaijani antonym pairs → we expect LOWER cosine similarities
ANT_PAIRS = [
    ("yaxşı", "pis"),
    ("bahalı", "ucuz"),
    ("gözəl", "çirkin"),
    ("sevirəm", "nifrət"),
]

# Optional: seed words we may want to explore in notebooks
SEED_WORDS = [
    "yaxşı", "pis", "çox", "bahalı", "ucuz",
    "mükəmməl", "dəhşət",
    "<PRICE>", "<RATING_POS>",
    "gözəl", "yox",
]


# ---------------------------------------------------------------------------
# 3) DATA LOADER
# ---------------------------------------------------------------------------
def load_cleaned_sentences(files=None, limit=None):
    """
    Load tokenized sentences from the 5 cleaned Excel files.

    Returns:
        - all_sents: list[list[str]] → directly feedable to gensim
        - excel_paths: list[Path]    → for coverage calculation later

    Args:
        files: custom list of Excel paths; if None, DEFAULT_FILES is used.
        limit: if set, truncates the number of sentence-lists returned.
               (useful for fast hyperparameter tuning)

    Failure mode:
        If NO sentences can be loaded, we raise RuntimeError with a
        clear message, because tuning/evaluation without data is meaningless.
    """
    if files is None:
        files = DEFAULT_FILES

    all_sents = []
    excel_paths = []

    for f in files:
        p = Path(f)
        if not p.exists():
            # In classroom settings some files may be missing; we just skip.
            continue

        # We only need the cleaned_text column; other metadata are irrelevant
        df = pd.read_excel(p, usecols=["cleaned_text"])

        # Each row is a string → we split to get a list of tokens
        sents = df["cleaned_text"].astype(str).str.split().tolist()
        all_sents.extend(sents)
        excel_paths.append(p)

    if limit is not None:
        all_sents = all_sents[:limit]

    if not all_sents:
        # This is a hard error on purpose → signals that process_assignment.py
        # was not run or output paths are wrong.
        raise RuntimeError("No sentences found. Run process_assignment.py first.")

    return all_sents, excel_paths


# ---------------------------------------------------------------------------
# 4) COSINE SIMILARITY (LOW-LEVEL)
# ---------------------------------------------------------------------------
def _cos(a, b):
    """
    Simple cosine similarity between two numpy vectors.
    We isolate it here to make pair_sim() clearer.
    """
    return float(dot(a, b) / (norm(a) * norm(b)))


# ---------------------------------------------------------------------------
# 5) PAIRWISE SIMILARITY
# ---------------------------------------------------------------------------
def pair_sim(model, pairs):
    """
    Compute the average cosine similarity over a list of (w1, w2) pairs.

    - We first check if BOTH words exist in the model's vocabulary.
    - If at least one side is missing, we skip that pair.
    - If ALL pairs are missing, we return NaN so the caller can decide.

    This function is used for BOTH synonym and antonym probes.
    """
    vals = []
    for a, b in pairs:
        if a in model.wv and b in model.wv:
            vals.append(_cos(model.wv[a], model.wv[b]))
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


# ---------------------------------------------------------------------------
# 6) LEXICAL COVERAGE
# ---------------------------------------------------------------------------
def lexical_coverage(model, excel_files):
    """
    Measure how much of the ACTUAL assignment data is covered by the model.

    Definition:
        coverage = (#tokens that are in model vocabulary) / (total #tokens)

    We compute this PER FILE and then take the mean.
    This is important because some of the five files are more "review-like"
    and some are more "general". Averaging smooths that variability.
    """
    covs = []
    vocab = model.wv.key_to_index  # fast membership test

    for p in excel_files:
        df = pd.read_excel(p, usecols=["cleaned_text"])
        # Flatten tokens from all rows
        toks = [t for row in df["cleaned_text"].astype(str) for t in row.split()]
        hit = sum(1 for t in toks if t in vocab)
        covs.append(hit / max(1, len(toks)))

    return float(np.mean(covs)) if covs else 0.0


# ---------------------------------------------------------------------------
# 7) TOP-LEVEL EVAL
# ---------------------------------------------------------------------------
def evaluate_model(model, excel_files):
    """
    Produce the 4 numbers we need in the final report:

        - syn_mean   : avg cosine over SYN_PAIRS
        - ant_mean   : avg cosine over ANT_PAIRS
        - separation : syn_mean - ant_mean  (higher → better semantic structure)
        - coverage   : lexical_coverage(...)

    We also guard against the "all pairs missing" case:
    if either syn_mean or ant_mean is NaN, we return separation = -1.0.
    This makes bad models obviously bad in the logs.
    """
    syn = pair_sim(model, SYN_PAIRS)
    ant = pair_sim(model, ANT_PAIRS)

    if np.isnan(syn) or np.isnan(ant):
        sep = -1.0
    else:
        sep = syn - ant

    cov = lexical_coverage(model, excel_files)

    return {
        "syn_mean": syn,
        "ant_mean": ant,
        "separation": sep,
        "coverage": cov,
    }
