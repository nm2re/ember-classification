import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class EMBERDataLoader:
    """
    Loads and processes EMBER JSONL data.
    ----- NOT USED ANYMORE ---------
    UPDATED VERSION -> extract_and_save.py
    """

    def __init__(self, data_path=r"D:\CodingFiles\PersonalCoding\python\EmberClassification\data\raw"):
        self.data_path = Path(data_path)
        self.dataset_folders = sorted([
            f for f in self.data_path.iterdir()
            if f.is_dir() and f.name.startswith('ember_2018')
        ])

    def load_jsonl_with_progress(self, filepath):
        """
        Loads a single JSONL file and prints timing/throughput.
        """
        start = time.time()
        df = pd.read_json(filepath, lines=True)
        elapsed = time.time() - start
        size_mb = filepath.stat().st_size / (1024 * 1024)
        rate = size_mb / elapsed if elapsed > 0 else 0
        print(f"  Loaded {filepath.name}: {len(df)} rows, {size_mb:.0f}MB, {elapsed:.1f}s ({rate:.1f} MB/s)")
        return df

    def load_all_training_data(self):
        """
        Loads and concatenates all training JSONL files across dataset folders.
        """
        all_dfs = []

        for folder in self.dataset_folders:
            print(f"\n=== Processing {folder.name} ===")
            train_files = sorted(folder.glob("train_features_*.jsonl"))

            if not train_files:
                print(f"No training files found in {folder.name}")
                continue

            for file in train_files:
                df = self.load_jsonl_with_progress(file)
                all_dfs.append(df)

        if not all_dfs:
            print("No training data loaded")
            return None

        train_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\nTotal training samples loaded: {len(train_df)}")
        return train_df

    def load_test_data(self):
        """
        Loads and concatenates test JSONL files across dataset folders.
        """
        all_test_dfs = []

        for folder in self.dataset_folders:
            test_file = folder / "test_features.jsonl"
            if not test_file.exists():
                continue

            print(f"Loading test data from {folder.name}...")
            try:
                df = pd.read_json(test_file, lines=True)
                all_test_dfs.append(df)
                print(f"Loaded {len(df)} test samples")
            except Exception as e:
                print(f"Error: {e}")

        if not all_test_dfs:
            print("No test data found")
            return None

        test_df = pd.concat(all_test_dfs, ignore_index=True)
        print(f"\nTotal test samples: {len(test_df)}")
        return test_df

    def prepare_data(self):
        """
        Loads train/test data, splits train into train/val, and scales all features.
        """
        train_df = self.load_all_training_data()
        if train_df is None:
            print("Failed to load training data")
            return None

        test_df = self.load_test_data()

        print(f"\n=== Data Info ===")
        print(f"Columns: {train_df.columns.tolist()[:10]}... ({len(train_df.columns)} total)")

        label_col = 'label' if 'label' in train_df.columns else train_df.columns[-1]

        X_train = train_df.drop(label_col, axis=1, errors='ignore')
        y_train = train_df[label_col] if label_col in train_df.columns else None

        print(f"\nTraining data shape: {X_train.shape}")
        print(f"Number of features: {len(X_train.columns)}")

        if y_train is not None:
            print(f"\nLabel distribution:")
            print(y_train.value_counts())

        if test_df is not None:
            X_test = test_df.drop(label_col, axis=1, errors='ignore')
            y_test = test_df[label_col] if label_col in test_df.columns else None
        else:
            X_test = None
            y_test = None

        # simple 70/30 train/val split, not stratified
        n_samples = len(X_train)
        split_idx = int(0.7 * n_samples)

        X_train_split = X_train.iloc[:split_idx]
        y_train_split = y_train.iloc[:split_idx] if y_train is not None else None

        X_val = X_train.iloc[split_idx:]
        y_val = y_train.iloc[split_idx:] if y_train is not None else None

        print(f"\n=== Data Split ===")
        print(f"Train set: {len(X_train_split)} samples (70%)")
        print(f"Validation set: {len(X_val)} samples (30%)")
        if X_test is not None:
            print(f"Test set: {len(X_test)} samples")

        print(f"\n=== Normalizing Features ===")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_split)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test) if X_test is not None else None
        print("Features normalized")

        return {
            'X_train': X_train_scaled,
            'y_train': y_train_split.values if y_train_split is not None else None,
            'X_val': X_val_scaled,
            'y_val': y_val.values if y_val is not None else None,
            'X_test': X_test_scaled,
            'y_test': y_test.values if y_test is not None else None,
            'scaler': scaler,
            'feature_names': X_train.columns.tolist()
        }


if __name__ == "__main__":
    loader = EMBERDataLoader()
    train_df = loader.load_all_training_data()
    test_df = loader.load_test_data()

    if train_df is not None:
        print(f"\nSuccessfully loaded:")
        print(f"Train: {len(train_df)} rows, {train_df.columns.tolist()}")
        print(f"Label distribution:\n{train_df['label'].value_counts()}")
        print(f"\nTest: {len(test_df)} rows")
    else:
        print("Failed to load training data")