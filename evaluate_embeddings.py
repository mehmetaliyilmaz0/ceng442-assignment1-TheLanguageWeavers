"""
- Loads final Word2Vec / FastText models (trained on full corpus).
- Evaluates them with the SAME metric used during Optuna (syn/ant/separation/coverage).
- Additionally runs a simple, assignment-style qualitative probe:
  * per-dataset lexical coverage
  * synonym / antonym similarity comparison
  * nearest neighbors for a fixed Azerbaijani seed list

This matches the assignment’s “9) Compare Word2Vec vs FastText (simple metrics)”
section, so the TA can run ONLY this file and see everything needed.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from gensim.models import Word2Vec, FastText

from eval_utils import load_cleaned_sentences, evaluate_model

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
FILES = [
    "cleaned_data/labeled-sentiment_2col.xlsx",
    "cleaned_data/test__1__2col.xlsx",
    "cleaned_data/train__3__2col.xlsx",
    "cleaned_data/train-00000-of-00001_2col.xlsx",
    "cleaned_data/merged_dataset_CSV__1__2col.xlsx",
]

SEED_WORDS = [
    "yaxşı", "pis", "çox", "bahalı", "ucuz",
    "mükəmməl", "dəhşət", "<PRICE>", "<RATING_POS>"
]

SYN_PAIRS = [
    ("yaxşı", "əla"),
    ("bahalı", "qiymətli"),
    ("ucuz", "sərfəli"),
]

ANT_PAIRS = [
    ("yaxşı", "pis"),
    ("bahalı", "ucuz"),
]


def _lexical_coverage(model, tokens):
    """Per-dataset coverage used in the assignment-style snippet."""
    vocab = model.wv.key_to_index
    return sum(1 for t in tokens if t in vocab) / max(1, len(tokens))


def _read_tokens(path):
    """Read a cleaned Excel and flatten into a token list."""
    df = pd.read_excel(path, usecols=["cleaned_text"])
    return [tok for row in df["cleaned_text"].astype(str) for tok in row.split()]


def _cos(a, b):
    return float(dot(a, b) / (norm(a) * norm(b)))


def _pair_sim(model, pairs):
    vals = []
    for a, b in pairs:
        try:
            vals.append(model.wv.similarity(a, b))
        except KeyError:
            # skip missing pairs
            continue
    return sum(vals) / len(vals) if vals else float("nan")


def _neighbors(model, word, k=5):
    try:
        return [w for w, _ in model.wv.most_similar(word, topn=k)]
    except KeyError:
        return []


if __name__ == "__main__":
    # ==============================================================
    # 1) Load cleaned excel paths (for coverage)
    # ==============================================================
    # we don't need sentences here, only the file list
    _, excel_files = load_cleaned_sentences()

    # ==============================================================
    # 2) Load models if they exist
    # ==============================================================
    w2v_path = Path("embeddings/word2vec.final.model")
    ft_path = Path("embeddings/fasttext.final.model")

    w2v = Word2Vec.load(str(w2v_path)) if w2v_path.exists() else None
    ft = FastText.load(str(ft_path)) if ft_path.exists() else None

    # ==============================================================
    # 3) Primary metric (the one we report in README)
    # ==============================================================
    if w2v is not None:
        m_w2v = evaluate_model(w2v, excel_files)
        print("Word2Vec (final):", m_w2v)
    else:
        m_w2v = None

    if ft is not None:
        m_ft = evaluate_model(ft, excel_files)
        print("FastText (final):", m_ft)
    else:
        m_ft = None

    # ==============================================================
    # 4) Assignment-style comparison block (your screenshot)
    # ==============================================================
    if w2v is not None and ft is not None:
        print("\n== Lexical coverage (per dataset) ==")
        for f in FILES:
            p = Path(f)
            if not p.exists():
                print(f"- {f} → missing, skipped")
                continue
            toks = _read_tokens(p)
            cov_w2v = _lexical_coverage(w2v, toks)
            cov_ft = _lexical_coverage(ft, toks)
            print(f"{f}: W2V={cov_w2v:.3f}, FT(vocab)={cov_ft:.3f}  # FT still embeds OOV via subwords")

        print("\n== Similarity (higher better for synonyms; lower better for antonyms) ==")
        syn_w2v = _pair_sim(w2v, SYN_PAIRS)
        syn_ft = _pair_sim(ft, SYN_PAIRS)
        ant_w2v = _pair_sim(w2v, ANT_PAIRS)
        ant_ft = _pair_sim(ft, ANT_PAIRS)

        print(f"Synonyms: W2V={syn_w2v:.3f}, FT={syn_ft:.3f}")
        print(f"Antonyms: W2V={ant_w2v:.3f}, FT={ant_ft:.3f}")
        print(f"Separation (Syn − Ant): W2V={(syn_w2v - ant_w2v):.3f}, FT={(syn_ft - ant_ft):.3f}")

        print("\n== Nearest neighbors (qualitative) ==")
        for w in SEED_WORDS:
            print(f"  W2V NN for '{w}': { _neighbors(w2v, w, k=5) }")
            print(f"  FT  NN for '{w}': { _neighbors(ft,  w, k=5) }")
    else:
        print("\n[INFO] Only one model found. Skipping pairwise W2V vs FT comparison.")
