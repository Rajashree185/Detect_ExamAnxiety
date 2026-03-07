"""
preprocess.py - Data Preprocessing Script
==========================================
Cleans and preprocesses the exam anxiety dataset for BERT training.
Handles text cleaning, label encoding, and train/validation splitting.
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_FILE = os.path.join(DATA_DIR, "anxiety_dataset.csv")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VAL_FILE = os.path.join(DATA_DIR, "val.csv")

LABEL_MAP = {"Low": 0, "Moderate": 1, "High": 2}
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ─── Text Cleaning ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean and normalize input text."""
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'\"-]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Main Processing Pipeline ───────────────────────────────────────────────

def preprocess_dataset():
    """Load, clean, encode, split, and save the dataset."""
    print("=" * 60)
    print("  Exam Anxiety Dataset Preprocessing")
    print("=" * 60)

    # 1. Load raw data
    print(f"\n[1/5] Loading dataset from: {RAW_FILE}")
    df = pd.read_csv(RAW_FILE)
    print(f"      Loaded {len(df)} samples")

    # 2. Drop missing values
    print("[2/5] Removing missing/null entries...")
    initial_count = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip() != ""]
    removed = initial_count - len(df)
    print(f"      Removed {removed} invalid entries, {len(df)} remaining")

    # 3. Clean text
    print("[3/5] Cleaning text data...")
    df["text"] = df["text"].apply(clean_text)

    # 4. Encode labels
    print("[4/5] Encoding labels...")
    df["label_encoded"] = df["label"].map(LABEL_MAP)

    # Verify all labels mapped correctly
    unmapped = df["label_encoded"].isna().sum()
    if unmapped > 0:
        print(f"      WARNING: {unmapped} samples have unknown labels!")
        df = df.dropna(subset=["label_encoded"])
    df["label_encoded"] = df["label_encoded"].astype(int)

    # Print class distribution
    print("\n  Class Distribution:")
    for label_name, label_id in LABEL_MAP.items():
        count = (df["label_encoded"] == label_id).sum()
        pct = count / len(df) * 100
        print(f"    {label_name:>10} (id={label_id}): {count:>4} samples ({pct:.1f}%)")

    # 5. Train/Validation split
    print(f"\n[5/5] Splitting data ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} train/val)...")
    train_df, val_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label_encoded"]
    )

    # Save processed data
    train_df.to_csv(TRAIN_FILE, index=False)
    val_df.to_csv(VAL_FILE, index=False)

    print(f"      Train set: {len(train_df)} samples -> {TRAIN_FILE}")
    print(f"      Val set:   {len(val_df)} samples  -> {VAL_FILE}")
    print("\n" + "=" * 60)
    print("  Preprocessing complete!")
    print("=" * 60)

    return train_df, val_df


if __name__ == "__main__":
    preprocess_dataset()
