import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
# -----------------------------
# 1. Directory configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # Project root directory
RAW_DIR = BASE_DIR / "raw_data"
PROCESSED_DIR = BASE_DIR / "processed"

TRAIN_FILE = RAW_DIR / "train.csv"
TEST_FILE = RAW_DIR / "test.csv"
CLEANED_TRAIN_FILE = PROCESSED_DIR / "train_clean_v1.csv"
CLEANED_TEST_FILE = PROCESSED_DIR / "test_clean_v1.csv"


# -----------------------------
# 2. Helper functions
# -----------------------------
def clean_text(text: str) -> str:
    """
    Perform basic text cleaning operations:
    - Convert to lowercase
    - Remove HTML tags and URLs
    - Remove special characters
    - Normalize whitespace
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)              # Remove HTML tags
    text = re.sub(r"http\S+|www.\S+", " ", text)      # Remove URLs
    text = re.sub(r"[^a-z0-9\s.,%-]", " ", text)      # Keep alphanumeric and punctuation
    text = re.sub(r"\s+", " ", text).strip()          # Normalize whitespace
    return text


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in the DataFrame.
    - For object columns: replace NaN with empty string
    - For numeric columns: replace NaN with median
    """
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("")
        else:
            df[col] = df[col].fillna(df[col].median())
    return df


def remove_price_outliers(df: pd.DataFrame, price_col: str = "price",
                          lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    """
    Remove outliers based on quantile thresholds for the 'price' column.
    Parameters:
        price_col: Name of the price column.
        lower_q, upper_q: Quantile thresholds for filtering.
    """
    if price_col in df.columns:
        lower, upper = df[price_col].quantile([lower_q, upper_q])
        df = df[(df[price_col] >= lower) & (df[price_col] <= upper)]
    return df.reset_index(drop=True)


# -----------------------------
# 3. Main cleaning pipeline
# -----------------------------
def clean_dataset(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Execute the complete data cleaning pipeline:
        1. Handle missing values
        2. Clean text-based columns
        3. Remove outliers for training data (if applicable)
    """
    print(f"Cleaning dataset (train={is_train}) ...")

    # Handle missing values
    df = fill_missing_values(df)

    # Clean text-based fields
    text_cols = [c for c in df.columns if "content" in c or "description" in c or "title" in c]
    for col in tqdm(text_cols, desc="Cleaning text columns"):
        df[col + "_clean"] = df[col].apply(clean_text)

    # Remove outliers if this is training data
    if is_train and "price" in df.columns:
        df = remove_price_outliers(df)

    print(f"Completed cleaning: {len(df)} rows, {len(df.columns)} columns.")
    return df
if __name__ == "__main__":
    print("Reading raw data ...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    # Perform cleaning
    train_clean = clean_dataset(train, is_train=True)
    test_clean = clean_dataset(test, is_train=False)

    # Save cleaned datasets
    print("Saving cleaned files ...")
    train_clean.to_csv(CLEANED_TRAIN_FILE, index=False)
    test_clean.to_csv(CLEANED_TEST_FILE, index=False)

    print(f"Data cleaning completed.\n"
          f"Cleaned training data: {CLEANED_TRAIN_FILE}\n"
          f"Cleaned test data: {CLEANED_TEST_FILE}")
