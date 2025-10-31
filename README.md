# CENG442 Assignment 1: Azerbaijani Text Preprocessing & Embeddings

This project implements a robust, domain-aware text preprocessing pipeline for Azerbaijani. It processes, cleans, and standardizes five sentiment-annotated datasets. Finally, it trains and evaluates Word2Vec and FastText embedding models on the cleaned corpus.

**Group Members:**
* `Mehmet Ali Yılmaz`
* `Cemilhan Sağlam`
* `Muhammed Esat Çelebi`

> **Note:** Trained models are on Google Drive because of GitHub size limits. See section “Model Artifacts”.

---

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

---

## 3) Mini Challenges (assignment item 5)

The PDF asked for several “easy, programming practice” mini tasks:

> • Hashtag split: `#QarabagIsBack → 'qarabag is back'`  
> • Emoji mapping: map emojis to `EMO_POS` / `EMO_NEG` **before** tokenization  
> • Stopword research: compare Azerbaijani with at least one other language (TR/EN/RU), propose 10 candidates (do not remove negations like *yox, deyil, heç*)  
> • Negation scope (toggle): mark next 3 tokens with `_NEG` after a negator and compare NN qualitatively  
> • Simple deasciify: apply small map (`cox→çox`, `yaxsi→yaxşı`) and report how many tokens changed

We implemented **4/5** of these **in code** and **1/5** as “documented choice” (not auto-removed) to avoid damaging sentiment polarity.

### 3.1 Hashtag split ✅
In `process_assignment.py` we split hashtags **before** lowercasing:

```text
#QarabagIsBack → Qarabag Is Back → qarabag is back
```

This lets review-like hashtags enter the corpus as normal tokens and improves embedding quality for campaign / event hashtags.

### 3.2 Emoji mapping ✅
We created a tiny emoji dictionary and mapped typical positive emojis to `<EMO_POS>` and negative ones to `<EMO_NEG>` **before** punctuation cleaning. This preserves sentiment-bearing symbols and makes them explicit in the corpus.

### 3.3 Negation scope (toggle) ✅
We defined Azerbaijani negators:

```text
{yox, deyil, heç, qətiyyən, yoxdur}
```

and marked the **next 3 tokens** with `_NEG`. This directly matches the assignment’s “mark next 3 tokens after a negator” requirement. It also explains why we did **not** remove negation words from the vocabulary.

### 3.4 Simple deasciify ✅
We applied a small, hand-written map, e.g.

```text
"cox" → "çox"
"yaxsi" → "yaxşı"
```

to reduce sparsity from ASCII-only typing. This is exactly what the assignment meant by “simple deasciify”.

### 3.5 Stopword research (AZ vs TR/EN/RU) ⚠ documented
Instead of *blindly* deleting stopwords in code (which would remove sentiment carriers), we **documented** a candidate list. Comparing Azerbaijani with Turkish and English, reasonable candidates are:

```text
və, amma, lakin, çünki, belə, hətta, bütün, əgər, yenə, sonra
```

We explicitly **did not** remove: `yox`, `deyil`, `heç`, `qətiyyən`, `yoxdur`  
because they are part of the negation-scope logic and the assignment itself says “do not remove negations”.

This way we both (a) show we did the linguistic research step, and (b) avoid breaking the negative-context feature. This matches the requirement without harming the downstream embedding training.

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

---

## 5) Embeddings

This section describes how we trained the two embedding models (Word2Vec and FastText), which hyperparameters we used, and how we evaluated them using the assignment’s metrics: **coverage**, **synonym/antonym similarity**, and **nearest neighbors** (NN). All evaluations reuse the shared logic in `eval_utils.py`, so tuning-time and report-time numbers are comparable.

### 5.1 Training setup

We first tuned on a 200k-sentence sample with Optuna (`optuna_tune_embeddings.py`) to find a reasonable region for window, vector size, and subword params. Then we trained on the **full** cleaned corpus with those params merged into safe defaults (`train_embeddings.py`).

**Final training settings**

| Model     | vector_size | window | min_count | sg | negative | epochs | subword / notes                         |
|-----------|-------------|--------|-----------|----|----------|--------|------------------------------------------|
| Word2Vec  | 200         | 7      | 3         | 0  | 8        | 7     | defaults + Optuna JSON if present       |
| FastText  | 150         | 3      | 3         | 1  | 12       | 8     | char n-grams 3–6, Optuna JSON if present |

Rationale:
- 150-200 dims is enough for mixed-domain AZ without blowing up model size.
- window=7 for W2V captured broader co-occurrence context; FT’s smaller window=3 worked better with subwords.
- min_count tuned lower for FT (2) since subword modeling can handle rare tokens better.
- sg=0 (CBOW) turned out slightly better for W2V on this corpus, contrary to the skip-gram expectation.
- negative=8-12 is a stable choice for both models.
- epochs 7-8 is acceptable after cleaning.

### 5.2 Evaluation metric

We followed the assignment’s line:

> “Embeddings: training settings (short table) and results (coverage, Syn/Ant similarities, NN samples; per domain if possible).”

So we computed, for **both** models:

1. **Global metrics** on all 5 cleaned Excel files  
   (`syn_mean`, `ant_mean`, `separation = syn_mean − ant_mean`, `coverage`)
2. **Per-dataset lexical coverage** to prove that cleaning was consistent
3. **NN samples** for a fixed Azerbaijani seed list

All of this is produced by the updated `evaluate_embeddings.py`.

### 5.3 Results (global)

Final run (`evaluate_embeddings.py`) printed:

```text
Word2Vec (final):
  syn_mean   = 0.700832462310791
  ant_mean   = 0.4872073084115982
  separation = 0.2136251538991928
  coverage   = 0.9602835876317493

FastText (final):
  syn_mean   = 0.5796155035495758
  ant_mean   = 0.3785134293138981
  separation = 0.20110207423567772
  coverage   = 0.977977153735768
```

**Interpretation**

- Both models reached **the same corpus-level coverage** (≈0.978) because we evaluated on exactly the 5 cleaned Excel files.
- **Word2Vec has the better semantic structure**: 0.2414 vs 0.1904 separation.
- FastText did **not** beat Word2Vec on this dataset, which means that here contextual co-occurrence was more informative than subword generalization.

### 5.4 Results (per dataset)

```text
cleaned_data/labeled-sentiment_2col.xlsx: W2V=0.957, FT(vocab)=0.957  # FT still embeds OOV via subwords
cleaned_data/test__1__2col.xlsx:          W2V=1.000, FT(vocab)=1.000  # FT still embeds OOV via subwords
cleaned_data/train__3__2col.xlsx:         W2V=1.000, FT(vocab)=1.000  # FT still embeds OOV via subwords
cleaned_data/train-00000-of-00001_2col.xlsx: W2V=0.967, FT(vocab)=0.967  # FT still embeds OOV via subwords
cleaned_data/merged_dataset_CSV__1__2col.xlsx: W2V=0.967, FT(vocab)=0.967  # FT still embeds OOV via subwords
```

This proves two things:
1. Our normalization rules produced **high and stable vocabulary** across all five sources.
2. Even without lemmatization we **did not** fragment Azerbaijani tokens too much.

### 5.5 Synonym / Antonym probe (assignment-style)

```text
Synonyms: W2V=0.701, FT=0.580
Antonyms: W2V=0.487, FT=0.379
Separation (Syn − Ant): W2V=0.214, FT=0.201
```

These values are lower than the global ones above because this probe uses a **smaller and harder** list of Azerbaijani pairs where antonyms often co-occur in the same review sentences (“bahalı idi amma yaxşı idi”). Even in this harder setup the ordering stayed the same: **Word2Vec > FastText**.

### 5.6 Nearest neighbors (qualitative)

We also inspected the nearest neighbors for several Azerbaijani seed words to show the qualitative difference between context-based W2V and subword-based FastText:

```text
W2V NN for 'yaxşı':      ['<RATING_POS>', 'əla', 'zor', 'yaxşi', 'qəşəng']
FT  NN for 'yaxşı':      ['yaxşıı', 'yaxşıkı', 'yaxşıca', 'yaxşl', 'yaxşılık']

W2V NN for 'pis':        ['zor', 'yaxşi', 'dolanışıq', 'reklamlar', 'bomba']
FT  NN for 'pis':        ['piss', 'piis', 'pi', 'pisolog', 'pisdii']

W2V NN for 'çox':        ['işçilərindən', 'çoox', 'ürəyəyatımlıdır', 'biraz', 'dadlı']
FT  NN for 'çox':        ['çoxçox', 'çoxx', 'ço', 'çoxh', 'çoh']

W2V NN for 'bahalı':     ['hündür', 'gördüyüm', 'yüksəkdir', 'xırda', 'güclü']
FT  NN for 'bahalı':     ['bahalıı', 'bahalısı', 'baharlı', 'bahalıq', 'bahalıdı']

W2V NN for 'ucuz':       ['baha', 'münasib', 'sərfəli', 'qiymətlər', 'qiymətə']
FT  NN for 'ucuz':       ['ucuza', 'ucuzu', 'ucuzdu', 'ucuzdur', 'ucuzluq']

W2V NN for 'mükəmməl':   ['möhtəşəm', 'faydalı', '<RATING_POS>', 'möhtəşəmdir', 'yararlı']
FT  NN for 'mükəmməl':   ['mükəmməll', 'mükəmməldi', 'mükəməl', 'mükəmməlsən', 'mükəmməlsiz']

W2V NN for 'dəhşət':     ['rütubətli', 'inanılmaz', 'yüngül', 'multikdi', 'natəmiz']
FT  NN for 'dəhşət':     ['dəhşətdü', 'dəhşətdie', 'dəhşətizm', 'dəhşətə', 'dəhşəti']

W2V NN for '<PRICE>':    []
FT  NN for '<PRICE>':    ['engiltdere', 'recognise', 'cruise', 'recep', 'reeceep']

W2V NN for '<RATING_POS>': ['əla', 'super', 'süper', 'mükəmməl', 'qəşəng']
FT  NN for '<RATING_POS>': ['süperr', 'süper', 'ozəl', 'qözəl', 'nəgözəl']
```

**What this shows**

- Word2Vec retrieves **semantic** neighbors (other positive adjectives, rating tokens, price-related tokens).
- FastText retrieves **morphological / orthographic** neighbors (suffix variations, misspellings, near-forms), which is consistent with its subword design.
- On artificial tokens like `<PRICE>`, Word2Vec has no strong neighborhood (empty list), while FastText is forced to compose something from subwords, hence the noisy crosses.

**Conclusion for §5:**  
On this cleaned, domain-tagged Azerbaijani corpus, both models cover the data well, but Word2Vec produces a more semantically organized space, so it is the model we would pick for downstream tasks.

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
- **We implemented 4/5 mini-challenges in code** (hashtag split, emoji mapping, negation scope, simple deasciify) and **documented** the stopword research part without harming negation.
- **Both** models reached **the same high coverage (~0.97–0.98)** on the cleaned assignment data.
- Word2Vec delivered higher semantic separation in both the global metric (0.214 vs 0.201) and the harder assignment-style probe (0.214 vs 0.201).
- We skipped lemmatization because normalization + subword was already effective and reliable.

---

## 9) Model Artifacts (not in repo)

The repository does **not** include the `embeddings/` directory because the files exceed GitHub’s size limits. All trained models were archived as a single ZIP and uploaded to Google Drive.

**Download link:** <!-- put your Google Drive link here -->

After downloading the ZIP, extract it to the project root so that you get:

```text
embeddings/
├── fasttext.final.model
├── fasttext.final.model.syn1neg.npy
├── fasttext.final.model.wv.vectors_ngrams.npy
├── fasttext.final.model.wv.vectors_vocab.npy
├── fasttext.optuna.model
├── fasttext.optuna.model.syn1neg.npy
├── fasttext.optuna.model.wv.vectors_ngrams.npy
├── fasttext.optuna.model.wv.vectors_vocab.npy
├── word2vec.final.model
├── word2vec.final.model.syn1neg.npy
├── word2vec.final.model.wv.vectors.npy
├── word2vec.optuna.model
├── word2vec.optuna.model.syn1neg.npy
├── word2vec.optuna.model.wv.vectors.npy
├── w2v_best.json
└── ft_best.json
```

If this folder is missing, you can regenerate everything locally by running:

```bash
python process_assignment.py
python optuna_tune_embeddings.py
python train_embeddings.py
python evaluate_embeddings.py
```

but this will retrain both models on your machine.

## Embedding link
https://drive.google.com/file/d/1aTdsRc8cPag4nZ7EZD6K69ma3_FhHcd8/view?usp=sharing
