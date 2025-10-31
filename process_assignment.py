#DESCRIPTION:
# This script performs the full ETL (Extract, Transform, Load) pipeline for
# CENG442 Assignment 1. It reads five raw Azerbaijani text datasets,
# cleans and standardizes them using a domain-aware pipeline,
# and outputs two sets of deliverables:
# 1. Five 2-column (cleaned_text, sentiment_value) Excel files.
# 2. A single 'corpus_all.txt' file for embedding training.

# =================================================================
# STAGE 1: LIBRARIES AND CONFIGURATION
# =================================================================
import pandas as pd
import os
import re
import html           # For unescaping HTML entities like &amp;
import unicodedata    # For complex Unicode normalization (e.g., NFC)
from pathlib import Path # For robust cross-platform path handling
from ftfy import fix_text

# =================================================================
# STAGE 2: CORE HELPER FUNCTIONS
# =================================================================

def lower_az(s: str) -> str:
    """
    Performs Azerbaijani-aware lowercasing.
    Standard .lower() fails on 'I' (maps to 'i', not 'ı').
    """
    if not isinstance(s, str): return ""
    # Normalize composite characters (e.g., 'İ' as 'I' + '̇') first
    # to ensure consistent replacements.
    s = unicodedata.normalize("NFC", s)
    # Manually map Turkic capitals *before* standard lowercasing.
    s = s.replace("I", "ı").replace("İ", "i")
    s = s.lower()
    return s

# =================================================================
# STAGE 3: REGEX CONSTANTS AND DOMAIN RULES
# =================================================================

# --- Base Noise Regex ---
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\+?[\d\s\-\(\)]{6,}\d") # A robust regex for various phone formats
USER_RE = re.compile(r"@\w+")
MULTI_PUNCT = re.compile(r"([!?.,;:])\1{1,}") # e.g., "!!!" -> "!"
MULTI_SPACE = re.compile(r"\s+")
REPEAT_CHARS = re.compile(r"(.)\1{2,}", re.UNICODE) # e.g., "cooool" -> "cool"

# --- Core Tokenizer Regex ---
# This regex acts as a "positive filter" or "whitelist".
# We use .findall() with this to *only* keep valid words and our special tokens,
# implicitly dropping any junk or punctuation we missed.
TOKEN_RE = re.compile(
    r"[A-Za-zəƏĞğlıİıÖöÜüÇçŞşXxQq]+"  # Azerbaijani letters
    r"|<NUM>|<URL>|<EMAIL>|<PHONE>|<USER>|<EMO_POS>|<EMO_NEG>" # Special tokens
)

# --- Mini-Challenge Dictionaries ---
# NOTE: These are part of the assignment's mini-challenges.
# The lists are representative, not exhaustive.
EMO_MAP = {  } # 5 emojis will be tagged as positive, 5 as negative
EMO_MAP.update({e: "<EMO_POS>" for e in ["😊", "😄", "😍", "👍", "😁"]})
EMO_MAP.update({e: "<EMO_NEG>" for e in ["😞", "😠", "😟", "😢", "😡"]})

SLANG_MAP = {"sim":"salam","tmm":"tamam", "sagol":"sağol", "cox":"çox", "yaxsi":"yaxşı"}

# --- Negation Handling (Mini-Challenge) ---
NEGATORS = {"yox", "deyil", "heç", "qətiyyən", "yoxdur"}

# --- Domain Detection Rules (Business Logic) ---
NEWS_HINTS = re.compile(r"\b(apa|trend|azertac|reuters|bloomberg|dha)\b", re.I)
SOCIAL_HINTS = re.compile(r"\b(rt)\b|@|#") # @ and # are strong signals
REV_HINTS = re.compile(r"\b(azn|manat|qiymət|aldım|ulduz|çox yaxşı|çox pis)\b", re.I)

# --- Domain-Specific Normalization Rules (Reviews) ---
PRICE_RE = re.compile(r"\b\d+\s*(azn|manat)\b", re.I)
STARS_RE = re.compile(r"\b([1-5])\s*ulduz\b", re.I)
POS_RATE = re.compile(r"\bçox yaxşı\b")
NEG_RATE = re.compile(r"\bçox pis\b")

def detect_domain(text: str) -> str:
    """Detects the domain of the text using simple regex hints."""
    if not isinstance(text, str): return "general"
    # Search on the lowercased raw text for max effectiveness
    s = text.lower()
    if NEWS_HINTS.search(s): return "news"
    if SOCIAL_HINTS.search(s): return "social"
    if REV_HINTS.search(s): return "reviews"
    return "general"

def domain_specific_normalize(cleaned: str, domain: str) -> str:
    """
    Applies extra normalization rules based on the detected domain.
    Currently only implemented for 'reviews'.
    """
    if domain == "reviews":
        # Standardize price and star ratings to single tokens
        s = PRICE_RE.sub(" <PRICE> ", cleaned)
        s = STARS_RE.sub(lambda m: f" <STARS_{m.group(1)}> ", s)
        s = POS_RATE.sub(" <RATING_POS> ", s)
        s = NEG_RATE.sub(" <RATING_NEG> ", s)
        # Clean up potential extra spaces introduced by .sub()
        return " ".join(s.split())
    return cleaned

# =================================================================
# STAGE 4: MAIN NORMALIZATION PIPELINE
# =================================================================
def normalize_text_az(s: str, numbers_to_token=True, keep_sentence_punct=False) -> str:
    """
    Full text cleaning pipeline for Azerbaijani.
    This function defines the ordered steps of normalization.
    """
    if not isinstance(s, str): return ""

    # --- Pipeline Step 1: Emoji Mapping ---
    # Convert emojis to tokens *before* punctuation removal
    for emo, tag in EMO_MAP.items():
        s = s.replace(emo, f" {tag} ")

    # --- Pipeline Step 2: Encoding and HTML Cleaning ---
    s = fix_text(s)      # Fix encoding issues (e.g., 'Ã¼' -> 'ü')
    s = html.unescape(s) # Fix HTML entities (e.g., '&amp;' -> '&')
    s = HTML_TAG_RE.sub(" ", s) # Remove HTML tags

    # --- Pipeline Step 3: Standardize Special Tokens ---
    s = URL_RE.sub(" <URL> ", s)
    s = EMAIL_RE.sub(" <EMAIL> ", s)
    s = PHONE_RE.sub(" <PHONE> ", s)
    s = USER_RE.sub(" <USER> ", s)
    
    # --- Pipeline Step 4: Hashtag Splitting (Mini-Challenge) ---
    # Handle hashtags: remove '#' and split CamelCase.
    # e.g., #QarabagIsBack -> 'qarabag is back'
    s = re.sub(r"#([A-Za-z0-9_]+)", 
               lambda m: " " + re.sub(r'([a-z])([A-Z])', r'\1 \2', m.group(1)) + " ", 
               s)

    # --- Pipeline Step 5: Language-Specific Lowercasing ---
    # Apply this *after* special token/hashtag logic
    s = lower_az(s)
    
    # --- Pipeline Step 6: Fix Repeated Punctuation ---
    s = MULTI_PUNCT.sub(r"\1", s)

    # --- Pipeline Step 7: Number Normalization ---
    if numbers_to_token:
        s = re.sub(r"\d+", " <NUM> ", s)

    # --- Pipeline Step 8: Punctuation Removal ---
    # This is a critical branching step
    if keep_sentence_punct:
        # Mode 1: Used by build_corpus_txt
        # Preserve sentence-ending punctuation to allow sentence splitting.
        s = re.sub(r"[^\w\s<>'əğıöşüçƏĞIİÖŞÜÇxqXQ.!?]", " ", s)
    else:
        # Mode 2: Default for 2-column Excel files
        # Remove all punctuation, keeping only valid token characters.
        s = re.sub(r"[^\w\s<>'əğıöşüçƏĞIİÖŞÜÇxqXQ]", " ", s)

    # --- Pipeline Step 9: Whitespace Cleanup ---
    s = MULTI_SPACE.sub(" ", s).strip()

    # --- Pipeline Step 10: Tokenization and Final Fixes ---
    # Use the whitelist regex to extract *only* valid tokens
    toks = TOKEN_RE.findall(s)
    
    norm = []
    mark_neg = 0 # Flag for negation scope
    for t in toks:
        # Fix repeated characters (e.g., "coool" -> "cool")
        t = REPEAT_CHARS.sub(r"\1\1", t)
        
        # Apply slang/de-asciify mapping
        t = SLANG_MAP.get(t, t)
        
        # Apply negation scope (Mini-Challenge)
        if t in NEGATORS:
            norm.append(t)
            mark_neg = 3 # Mark next 3 tokens
            continue
            
        if mark_neg > 0 and not t.startswith("<"): # Don't negate special tokens
            norm.append(t + "_NEG")
            mark_neg -= 1
        else:
            norm.append(t)
            
    # Remove single-letter tokens (except 'o' and 'e', as per assignment)
    norm = [t for t in norm if not (len(t) == 1 and t not in ("o", "e"))]
    
    return " ".join(norm).strip()

# =================================================================
# STAGE 5: SENTIMENT MAPPING UTILITY
# =================================================================
def map_sentiment_value(v, scheme: str):
    """
    Converts various label formats (binary, tri) into a
    standard float (0.0, 0.5, 1.0).
    """
    if scheme == "binary":
        try: return 1.0 if int(v) == 1 else 0.0
        except Exception: return None 
    
    # Handle 3-class (tri) scheme
    s = str(v).strip().lower() 
    if s in ("pos", "positive", "1", "müsbət", "good", "pozitiv"): return 1.0
    if s in ("neu", "neutral", "2", "neytral"): return 0.5
    if s in ("neg", "negative", "0", "mənfi", "bad", "negativ"): return 0.0
    return None # Return None for unmappable values

# =================================================================
# STAGE 6: FILE PROCESSING FUNCTIONS
# =================================================================

def process_file(in_path, text_col, label_col, scheme, out_two_col_path):
    """
    Main worker function for Task 1 (2-column Excels).
    Loads, processes, and saves a single Excel file.
    """
    print(f"\n[Processing]: {in_path}")
    
    if not os.path.exists(in_path):
        print(f"ERROR: '{in_path}' file not found.")
        return 0 # Return 0 processed rows
        
    try:
        df = pd.read_excel(in_path)
        
        # Drop common junk columns from CSV->Excel conversion
        for c in ["Unnamed: 0", "index"]:
            if c in df.columns: df = df.drop(columns=[c])

        if text_col not in df.columns or label_col not in df.columns:
            print(f"WARNING: Expected columns ('{text_col}', '{label_col}') not found. Found: {list(df.columns)}")
            return 0
            
        # --- Pre-processing Cleanup ---
        df = df.dropna(subset=[text_col, label_col])
        df = df[df[text_col].astype(str).str.strip().str.len() > 0]
        # Deduplicate based on the *original* raw text
        df = df.drop_duplicates(subset=[text_col])
        
        # --- Main Cleaning Pipeline ---
        
        # 1. Detect domain from *original* text (cleaning might remove hints)
        df["domain"] = df[text_col].astype(str).apply(detect_domain)
        
        # 2. Apply base cleaning (Mode 2: remove all punctuation)
        print(f"Starting text cleaning for '{in_path}' (this may take a moment)...")
        df["cleaned_text"] = df[text_col].astype(str).apply(
            lambda s: normalize_text_az(s, numbers_to_token=True, keep_sentence_punct=False)
        )
        
        # 3. Apply domain-specific tweaks (e.g., <PRICE>)
        df["cleaned_text"] = df.apply(
            lambda r: domain_specific_normalize(r["cleaned_text"], r["domain"]), 
            axis=1
        )
        
        # 4. Standardize labels to 0.0, 0.5, 1.0
        df["sentiment_value"] = df[label_col].apply(lambda v: map_sentiment_value(v, scheme=scheme))
        
        # --- Post-processing Cleanup ---
        # Drop rows where mapping failed or text became empty after cleaning
        df = df.dropna(subset=["sentiment_value", "cleaned_text"])
        df = df[df["cleaned_text"].str.len() > 0]

        # --- Save Output ---
        out_df = df[["cleaned_text", "sentiment_value"]].reset_index(drop=True)
        
        # Ensure output directory exists
        Path(out_two_col_path).parent.mkdir(parents=True, exist_ok=True)
        
        out_df.to_excel(out_two_col_path, index=False)
        
        print(f"SUCCESS: '{out_two_col_path}' file saved. (Total {len(out_df)} rows)")
        return len(out_df)

    except Exception as e:
        print(f"CRITICAL ERROR while processing '{in_path}': {e}")
        return 0

# ---
def build_corpus_txt(input_files_config, out_txt="corpus_all.txt"):
    """
    Main worker function for Task 2 (corpus_all.txt).
    Reads original files, detects domain, splits into sentences,
    and creates a domain-tagged, punctuation-free .txt file.
    """
    print(f"\n--- Task 2: Creating '{out_txt}' ---")
    lines = []
    
    # Get (filename, text_column) info from config
    for f, text_col in input_files_config:
        print(f"Reading '{f}' for corpus...")
        if not os.path.exists(f):
            print(f"WARNING: '{f}' not found, skipping for corpus.")
            continue
            
        df = pd.read_excel(f)
        
        # CRITICAL DESIGN CHOICE:
        # We re-read the *original* files, not the 'cleaned_data' files.
        # This is because we need:
        # 1. The *original* text to run domain detection (cleaning might remove hints).
        # 2. To run a *different* normalization (keep_sentence_punct=True)
        #    to allow for sentence splitting.
        for raw in df[text_col].dropna().astype(str):
            dom = detect_domain(raw)
            
            # 1. Normalize BUT keep '.!?' for splitting (Mode 1)
            s_with_punct = normalize_text_az(raw, numbers_to_token=True, keep_sentence_punct=True)
            
            # 2. Split text into sentences
            parts = re.split(r"[.!?]+", s_with_punct)
            
            # 3. Process each part (sentence)
            for p in parts:
                p = p.strip()
                if not p: continue # Skip empty strings
                
                # 4. Now, remove the sentence punctuation we kept
                p_cleaned = re.sub(r"[^\w\s<>'əğıöşüçƏĞIİÖŞÜÇxqXQ]", "", p)
                p_cleaned = MULTI_SPACE.sub(" ", p_cleaned).strip().lower()
                
                if p_cleaned:
                    # 5. Add domain tag
                    lines.append(f"dom{dom} {p_cleaned}")
                    
    # 6. Write all lines to a single .txt file
    with open(out_txt, "w", encoding="utf-8") as w:
        for ln in lines:
            w.write(ln + "\n")
            
    print(f"SUCCESS: '{out_txt}' file saved. (Total {len(lines)} rows/sentences)")

# =================================================================
# STAGE 7: MAIN EXECUTION BLOCK
# =================================================================
def main():
    """Main entry-point for the script."""
    print("--- Starting Full Pipeline (.py script) ---")
    
    OUTPUT_DIR = Path("cleaned_data")
    
    # Configuration list defining the 5 datasets and their properties
    CFG = [
        ("labeled-sentiment.xlsx", "text", "sentiment", "tri"),
        ("test__1_.xlsx", "text", "label", "binary"),
        ("train__3_.xlsx", "text", "label", "binary"),
        ("train-00000-of-00001.xlsx", "text", "labels", "tri"),
        ("merged_dataset_CSV__1_.xlsx", "text", "labels", "binary"),
    ]

    # --- TASK 1: CREATE TWO-COLUMN EXCEL FILES ---
    print("\n--- Task 1: Creating Two-Column Excel Files ---")
    total_rows = 0
    for f_name, t_col, l_col, l_scheme in CFG:
        # Define output filename (e.g., 'labeled-sentiment_2col.xlsx')
        out_f_name = f"{Path(f_name).stem}_2col.xlsx"
        out_path = OUTPUT_DIR / out_f_name
        
        rows = process_file(f_name, t_col, l_col, l_scheme, out_path)
        total_rows += rows
    print(f"\nTask 1 Complete. Processed {total_rows} total rows.")

    # --- TASK 2: CREATE 'corpus_all.txt' ---
    # build_corpus_txt expects a list of (filename, text_column)
    corpus_config = [(c[0], c[1]) for c in CFG]
    build_corpus_txt(corpus_config, out_txt="corpus_all.txt")

    print("\n--- Pipeline Complete ---")

# --- SCRIPT ENTRY POINT ---
# This standard Python construct ensures that main() is called
# only when the script is executed directly.
if __name__ == "__main__":
    main()