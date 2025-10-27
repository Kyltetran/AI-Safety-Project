import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_and_combine_datasets(base_dir="data/processed"):
    files = {
        "gossipcop_fake.csv": "fake",
        "gossipcop_real.csv": "real",
        "politifact_fake.csv": "fake",
        "politifact_real.csv": "real",
    }

    dfs = []
    for filename, label in files.items():
        path = Path(base_dir) / filename
        df = pd.read_csv(path)
        df["label"] = label
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df[["id", "title", "label"]]
    full_df.to_csv(Path(base_dir) / "full_dataset.csv", index=False)
    return full_df


def split_dataset(full_df, base_dir="data/processed"):
    train_df, temp_df = train_test_split(
        full_df, test_size=0.3, random_state=42, stratify=full_df["label"])
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

    train_df.to_csv(f"{base_dir}/train.csv", index=False)
    val_df.to_csv(f"{base_dir}/val.csv", index=False)
    test_df.to_csv(f"{base_dir}/test.csv", index=False)

    print(f"✅ Saved splits to {base_dir}/train.csv, val.csv, test.csv")
