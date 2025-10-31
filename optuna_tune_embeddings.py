"""


- We intentionally:
  1. load the cleaned sentences ONCE (to avoid I/O on every trial),
  2. evaluate all trials with the SAME evaluation function (to guarantee comparability),
  3. store the best params as JSON (so a separate, full-training script can reuse them),
  4. parallelize trials via `n_jobs=4` (so tuning finishes in a reasonable time on 1 machine).

- Important constraint:
  This is NOT a full AutoML pipeline. We deliberately keep the search space small,
  reproducible, and understandable so that it fits a course assignment and can be
  inspected by the grader.

Prerequisites:
- `eval_utils.py` must define: load_cleaned_sentences(...), evaluate_model(...)
- `cleaned_data/*.xlsx` must already exist (i.e. `process_assignment.py` must have run)
"""

from pathlib import Path
import json

import optuna
from optuna.pruners import MedianPruner
from gensim.models import Word2Vec, FastText

# We reuse the EXACT same loader and evaluator that will be used later during
# final reporting. This is crucial to avoid "trained with X, evaluated with Y"
# inconsistencies.
from eval_utils import load_cleaned_sentences, evaluate_model

# ---------------------------------------------------------------------------
# 1) GLOBAL DATA LOAD
# ---------------------------------------------------------------------------
# We deliberately load only a SUBSET of the corpus (200k sentences).
# Rationale:
# - Hyperparameter search is exploratory: we don't need full corpus every time.
# - Small-but-representative sample makes each trial faster.
# - After we find good params, we will re-train on the FULL corpus in train_embeddings.py.
GLOBAL_SENTENCES, GLOBAL_EXCELS = load_cleaned_sentences(limit=200_000)

# All artifacts from tuning go here
EMB_DIR = Path("embeddings")
EMB_DIR.mkdir(exist_ok=True)


def _save_best_params(path: Path, params: dict) -> None:
    """
    Persist the best hyperparameters for a model to disk, as JSON.
    The follow-up script (train_embeddings.py) will read these and
    run a FULL training on the entire corpus.
    """
    with path.open("w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 2) WORD2VEC OBJECTIVE
# ---------------------------------------------------------------------------
def objective_w2v(trial: optuna.Trial) -> float:
    """
    Optuna objective for Word2Vec.

    We keep the search space narrow but meaningful:
    - vector_size: capacity of the embedding
    - window: amount of context to consider
    - min_count: lower bound on token frequency (controls vocab size)
    - sg: CBOW (0) vs Skip-gram (1)
    - negative: negative sampling
    - epochs: number of passes over the sampled corpus
    - sample: subsampling of frequent words

    We return a SINGLE scalar "score" that Optuna will try to maximize.
    Here, score = separation + 0.2 * coverage
    which prioritizes semantic quality but keeps lexical coverage in the loop.
    """
    # --- 2.1 Sample hyperparameters from search space ---
    vector_size = trial.suggest_int("vector_size", 150, 400, step=50)
    window = trial.suggest_int("window", 3, 10)
    min_count = trial.suggest_int("min_count", 1, 4)
    sg = trial.suggest_categorical("sg", [0, 1])
    negative = trial.suggest_int("negative", 5, 15)
    epochs = trial.suggest_int("epochs", 5, 12)
    sample = trial.suggest_float("sample", 1e-5, 1e-3, log=True)

    # --- 2.2 Train a temporary Word2Vec model on the sampled corpus ---
    # Note: workers=4 enables intra-trial parallelism on a typical laptop/WS
    model = Word2Vec(
        sentences=GLOBAL_SENTENCES,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        negative=negative,
        sample=sample,
        epochs=epochs,
        workers=4,
        seed=42,  # make trials reproducible
    )

    # --- 2.3 Evaluate with the course-provided metric ---
    # This calls back into eval_utils → SAME code used by evaluate_embeddings.py
    metrics = evaluate_model(model, GLOBAL_EXCELS)
    sep = metrics["separation"]
    cov = metrics["coverage"]

    # Higher is better. We slightly reward coverage so suboptimal
    # vocab truncation does not win purely on separation.
    score = sep + 0.2 * cov

    # Store raw metrics inside the trial, useful for later analysis
    trial.set_user_attr("metrics", metrics)

    # --- 2.4 Save best-so-far model & params ---
    # First trial has no completed sibling, so best_value is not defined.
    study = trial.study
    is_best = False
    try:
        # If at least one trial is completed, we can compare
        if score > study.best_value:
            is_best = True
    except ValueError:
        # No completed trials yet → this one is automatically the best so far
        is_best = True

    if is_best:
        # Save model checkpoint
        model.save(str(EMB_DIR / "word2vec.optuna.model"))
        # Save best hyperparameters for later full training
        _save_best_params(
            EMB_DIR / "w2v_best.json",
            {
                "vector_size": vector_size,
                "window": window,
                "min_count": min_count,
                "sg": sg,
                "negative": negative,
                "epochs": epochs,
                "sample": sample,
            },
        )

    # --- 2.5 Respect Optuna pruning (not very effective with gensim but OK) ---
    if trial.should_prune():
        raise optuna.TrialPruned()

    return score


# ---------------------------------------------------------------------------
# 3) FASTTEXT OBJECTIVE
# ---------------------------------------------------------------------------
def objective_ft(trial: optuna.Trial) -> float:
    """
    Optuna objective for FastText.

    Differences vs Word2Vec:
    - We always use sg=1 (Skip-gram) since subword + skip-gram tends to work
      better for low-resource, morphologically rich languages.
    - We search over (min_n, max_n) because character n-grams are THE reason to
      choose FastText over Word2Vec here.
    """
    # --- 3.1 Sample hyperparameters ---
    vector_size = trial.suggest_int("vector_size", 150, 400, step=50)
    window = trial.suggest_int("window", 3, 10)
    min_count = trial.suggest_int("min_count", 1, 4)
    negative = trial.suggest_int("negative", 5, 15)
    epochs = trial.suggest_int("epochs", 5, 12)
    min_n = trial.suggest_int("min_n", 3, 4)
    max_n = trial.suggest_int("max_n", 5, 6)
    # Safety: ensure min_n <= max_n
    if min_n > max_n:
        min_n, max_n = max_n, min_n

    # --- 3.2 Train temporary FastText model ---
    model = FastText(
        sentences=GLOBAL_SENTENCES,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,  # fixed: skip-gram
        negative=negative,
        epochs=epochs,
        min_n=min_n,
        max_n=max_n,
        workers=4,
        seed=42,
    )

    # --- 3.3 Evaluate with EXACTLY the same metric ---
    metrics = evaluate_model(model, GLOBAL_EXCELS)
    sep = metrics["separation"]
    cov = metrics["coverage"]
    score = sep + 0.2 * cov

    trial.set_user_attr("metrics", metrics)

    # --- 3.4 Save best-so-far model & params ---
    study = trial.study
    is_best = False
    try:
        if score > study.best_value:
            is_best = True
    except ValueError:
        # first completed trial
        is_best = True

    if is_best:
        model.save(str(EMB_DIR / "fasttext.optuna.model"))
        _save_best_params(
            EMB_DIR / "ft_best.json",
            {
                "vector_size": vector_size,
                "window": window,
                "min_count": min_count,
                "negative": negative,
                "epochs": epochs,
                "min_n": min_n,
                "max_n": max_n,
            },
        )

    if trial.should_prune():
        raise optuna.TrialPruned()

    return score


# ---------------------------------------------------------------------------
# 4) MAIN ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # -------------------------------
    # 4.1 Word2Vec parallel tuning
    # -------------------------------
    # We use MedianPruner to cut off clearly bad trials early.
    # We also set n_jobs=4 so 4 trials can run concurrently on a single machine.
    study_w2v = optuna.create_study(
        direction="maximize",
        study_name="w2v_az",
        pruner=MedianPruner(n_warmup_steps=5),
    )
    study_w2v.optimize(objective_w2v, n_trials=20, n_jobs=4)
    print("Best W2V:", study_w2v.best_trial.params, study_w2v.best_value)

    # -------------------------------
    # 4.2 FastText parallel tuning
    # -------------------------------
    study_ft = optuna.create_study(
        direction="maximize",
        study_name="ft_az",
        pruner=MedianPruner(n_warmup_steps=5),
    )
    study_ft.optimize(objective_ft, n_trials=20, n_jobs=4)
    print("Best FT:", study_ft.best_trial.params, study_ft.best_value)
