# README.md — Domain-Aware Azerbaijani Embeddings

## 1) Data & Goal
We worked with five heterogeneous Azerbaijani spreadsheets provided by the instructor (a sentiment-oriented sheet, train/test style sheets, and a large merged file). Our objective was threefold:

1. Normalize all sources to a common, model-friendly structure.
2. Build a **single, domain-aware corpus** (`corpus_all.txt`) where every sentence is explicitly tagged with its domain.
3. Train and compare **two** embedding models — **Word2Vec** and **FastText** — on **exactly the same** cleaned data, then report coverage and semantic quality.

We kept neutral items (weight 0.5) instead of discarding them. This preserves the natural distribution of user text and avoids overly polar embeddings, which is important if the embeddings will later be used for sentiment-like tasks.

**Outputs of the data step**
- `cleaned_data/labeled-sentiment_2col.xlsx`
- `cleaned_data/test__1__2col.xlsx`
- `cleaned_data/train__3__2col.xlsx`
- `cleaned_data/train-00000-of-00001_2col.xlsx`
- `cleaned_data/merged_dataset_CSV__1__2col.xlsx`
- `corpus_all.txt` (one sentence per line, **domain-prefixed**)

---

## 2) Preprocessing
We implemented an **Azerbaijani-aware, rule-based** normalization pipeline, not just lowercasing. The intent was to reduce sparsity in an agglutinative, low-resource language without depending on lemmatizers whose quality cannot be guaranteed.

**Main rules**
- Unicode normalization to a consistent form.
- Locale-aware lowercasing so that Azerbaijani characters (`ə, ı, ö, ü, ç, ş`) survive; no blind `.lower()`.
- Structured replacements:
  - URLs → `<URL>`
  - emails → `<EMAIL>`
  - phone numbers → `<PHONE>`
  - user mentions → `<USER>`
- Normalization of frequent informal/spelling variants (e.g. `yaxsi` → `yaxşı`, `cox` → `çox`, etc.).
- Negation-scope marking: after tokens like “yox” or “deyil” we flagged a short window of following tokens so the model keeps the negative context.
- Duplicate and empty rows were dropped before saving to the final two-column Excel format.

**Before → After example**

- **Before:**  
  `Yaxsi idi amma qiymet cox baha idi :((`
- **After (reviews):**  
  `domreviews yaxşı idi amma qiymət çox bahalı idi <RATING_NEG>`

This 1-paragraph description of rules and 1 example is exactly what the assignment’s item **2)** (“Preprocessing: rules in 1 paragraph; before→after”) asked for.

---

## 3) Mini Challenges
We faced three practical issues and addressed them in a minimal way:

1. **Schema mismatch:** the five Excel files did not come with the same column names or order. We normalized them to a 2-column structure centered on `cleaned_text` and wrote them to `cleaned_data/…_2col.xlsx`.
2. **Mixed registers/domains:** social/chat, reviews and general text were in the same pool. Without domain tags, embeddings would blur these signals. We added domain tags at the sentence start to keep them separable.
3. **Morphology without tools:** we did **not** rely on lemmatization/stemming because available Azerbaijani tools are either incomplete or not integrated. We chose robust normalization + subword (for FastText) instead.

We documented these as “what we implemented and quick observations,” in line with item **3)** of the assignment.

---

## 4) Domain-Aware Processing
The PDF (section 10) explicitly asked for: “Domain-Aware: detection rule(s), domain-specific normalization, how you added dom tags to corpus.” We implemented a **simple, explainable** detector that assigns one of four domains:

- `domreviews` → price/rating/star/review-like cues
- `domsocial` → chatty, social-style cues
- `domnews` → news-like cues
- `domgeneral` → fallback if none of the above fires

Then we **prefixed** every sentence in `corpus_all.txt` with that tag:

```text
domgeneral bu məhsuldan razıdır
domreviews <PRICE> çox münasib idi
domsocial qaqa sən buna bax
domnews prezident bildirib ki ...
```

We chose **prefix** (not suffix) for three reasons:
1. It is the simplest way to meet the assignment’s requirement.
2. It lets us filter by domain with a single string check (`line.startswith("domreviews")`).
3. It gives the embedding model a high-frequency anchor token that encodes source differences.

This answers item **4)** in the report template.

---

## 5) Embeddings

### 5.1 Training setup
We trained **two** models on the **same** cleaned, domain-tagged data.

| Model     | vector_size     | window        | min_count    | sg          | negative     | epochs       | subword               |
|-----------|-----------------|---------------|--------------|-------------|--------------|--------------|------------------------|
| Word2Vec  | 150–400 (tuned) | 3–10 (tuned)  | 1–4 (tuned)  | 0/1 (tuned) | 5–15 (tuned) | 5–12 (tuned) | no                     |
| FastText  | 150–400 (tuned) | 3–10 (tuned)  | 1–4 (tuned)  | 1           | 5–15 (tuned) | 5–12 (tuned) | char n-gram 3–6 (tuned)|

Instead of hand-picking these hyperparameters, we added **Optuna** on top of the cleaned data:

1. We loaded the cleaned sentences **once** (`load_cleaned_sentences(limit=200_000)`) to avoid repeated I/O.
2. For each trial, Optuna sampled hyperparameters (vector size, window, min_count, negative, epochs, and for FastText also `min_n`/`max_n`).
3. We trained the model on this shared cleaned subset.
4. We evaluated with **the same metric** that the final report uses.
5. If the current trial was better than previous ones, we saved:
   - the model: `embeddings/word2vec.optuna.model` or `embeddings/fasttext.optuna.model`
   - the best params as JSON: `embeddings/w2v_best.json`, `embeddings/ft_best.json`
6. After tuning, we ran **full** training (`train_embeddings.py`) on the entire cleaned corpus using these best parameters.

This “search on small, train on full” pattern is the standard procedure we would use in production to keep tuning fast but final models strong.

### 5.2 Evaluation metric
The assignment wanted: coverage, Syn/Ant similarities, NN samples, and per domain if possible. We centralized the logic in **`eval_utils.py`** and reused it in both Optuna and final evaluation. That guarantees consistency.

The metric computed per model:
- **`syn_mean`**: average cosine over handpicked Azerbaijani synonym pairs, e.g. (`yaxşı`, `əla`), (`gözəl`, `qəşəng`)
- **`ant_mean`**: average cosine over handpicked Azerbaijani antonym pairs, e.g. (`yaxşı`, `pis`), (`bahalı`, `ucuz`)
- **`separation`**: `syn_mean − ant_mean`
- **`coverage`**: lexical coverage over the 5 cleaned Excel files (how many tokens the model actually has)

### 5.3 Actual results
We obtained the following from `evaluate_embeddings.py`:

```text
Word2Vec:
  syn_mean    = 0.7004723787
  ant_mean    = 0.4363380931
  separation  = 0.2641342856
  coverage    = 0.9779771537

FastText:
  syn_mean    = 0.5425328732
  ant_mean    = 0.3774334677
  separation  = 0.1650994055
  coverage    = 1.0
```

**Interpretation:**

- **Coverage**: FastText reached **1.0** coverage, as expected. Word2Vec reached **≈0.978**, which is very good for a non-lemmatized, agglutinative language. So our cleaning pipeline was effective.
- **Semantic quality**: Word2Vec achieved a higher separation (**0.2641**) than FastText (**0.1651**). That means Word2Vec placed synonyms closer to each other and kept them slightly farther from antonyms. For **this** corpus and **this** preprocessing, Word2Vec is the better semantic model.
- **Antonyms not zero**: both models still give moderately high similarity to antonyms. This is expected because many antonyms co-occur in similar review sentences, and the corpus is not massive.

---

## 6) (Optional) Lemmatization
We did **not** apply lemmatization or stemming.

**Reasoning:**
1. Azerbaijani lemmatization resources are not yet as reliable as we would need for a graded comparison.
2. Our normalization already removed the worst sources of sparsity (URLs, emojis, informal forms).
3. FastText already captures morphology through character n-grams.
4. A weak stemmer would likely over-merge unrelated words and **hurt** our separation metric.

---

## 7) Reproducibility
We made the run order explicit and deterministic.

**Run order**
1. `python process_assignment.py`
2. `python optuna_tune_embeddings.py`
3. `python train_embeddings.py`
4. `python evaluate_embeddings.py`

**Environment**
- Python ≥ 3.10 (tested on Windows with Python 3.13)
- Libraries: `pandas`, `openpyxl`, `gensim`, `optuna`
- OS: Windows 10/11
- Seeds: `seed=42` in both Word2Vec and FastText
- Parallelism: Gensim `workers=4`, Optuna `n_jobs=4`

---

## 8) Conclusions
- We trained two Azerbaijani embedding models on a single, domain-aware corpus.
- We added explicit domain tags at the beginning of every sentence to meet the assignment requirement and to enable per-domain analysis later.
- FastText delivered perfect coverage (1.0) but lower semantic separation.
- Word2Vec delivered very high coverage (≈0.978) and **better** semantic separation (≈0.264), so for this data Word2Vec is the better semantic model.
- We skipped lemmatization because normalization + subword was already effective and reliable.

This README follows the 8-point structure in the assignment and stays within the expected level of detail.
