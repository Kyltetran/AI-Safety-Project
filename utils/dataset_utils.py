# utils/dataset_utils.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_four_csvs(base_dir="data/processed"):
    files = {
        "gossipcop_fake.csv": 1,
        "gossipcop_real.csv": 0,
        "politifact_fake.csv": 1,
        "politifact_real.csv": 0,
    }

    dfs = []
    for filename, label in files.items():
        path = Path(base_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        df = pd.read_csv(path)
        # keep id, title, and add numeric label
        df = df.rename(columns={col: col.strip() for col in df.columns})
        if "title" not in df.columns:
            raise ValueError(f"{path} does not contain 'title' column")
        df = df[["id", "title"]].copy()
        df["label"] = label
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)
    full = full.dropna(subset=["title"]).reset_index(drop=True)
    return full


def create_splits(df, base_dir="data/processed", test_size=0.3, val_size=0.5, random_state=42):
    """
    Splits into train / val / test.
    First split into train / temp (temp = val+test with size test_size),
    then split temp into val and test equally (val_size fraction of temp).
    """
    train_df, temp_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=random_state)
    val_df, test_df = train_test_split(
        temp_df, test_size=val_size, stratify=temp_df["label"], random_state=random_state)

    Path(base_dir).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(Path(base_dir) / "train.csv", index=False)
    val_df.to_csv(Path(base_dir) / "val.csv", index=False)
    test_df.to_csv(Path(base_dir) / "test.csv", index=False)
    full_path = Path(base_dir) / "full_dataset.csv"
    df.to_csv(full_path, index=False)
    return train_df, val_df, test_df
