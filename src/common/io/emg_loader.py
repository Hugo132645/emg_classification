from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.io import loadmat
import re

def _search_csv_files(path: str) -> list[str]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    return [str(p) for p in root.rglob("*.csv")]


def _search_mat_files(path: str) -> list[Path]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    subject_dir_pattern = re.compile(r"^[sS]\d+$")
    mat_files = []

    for p in root.rglob("*.mat"):
        if any(subject_dir_pattern.match(parent.name) for parent in p.parents):
            mat_files.append(p)

    return sorted(mat_files)


def load_emg_files(
    input_path: str,
    output_dir_name: str = "csv_files",
    exclude: list[str] | None = None,
) -> None:
    root = Path(input_path)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {input_path}")

    exclude = set(exclude or [])
    mat_files = _search_mat_files(input_path)

    output_root = root / output_dir_name
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Saving CSVs into: {output_root}")

    for mat_path in mat_files:
        if (
            mat_path.name in exclude
            or mat_path.stem in exclude
            or str(mat_path) in exclude
        ):
            continue

        data = loadmat(mat_path)

        n = data["emg"].shape[0]
        df = pd.DataFrame({"sample_idx": range(n)})

        for i in range(data["emg"].shape[1]):
            df[f"emg_{i+1}"] = data["emg"][:, i]

        df["restimulus"] = data["restimulus"].squeeze()
        df["rerepetition"] = data["rerepetition"].squeeze()
        df["subject"] = int(data["subject"].squeeze())
        df["exercise"] = int(data["exercise"].squeeze())

        out_csv = output_root / f"{mat_path.stem}_emg_only.csv"
        df.to_csv(out_csv, index=False)

        print(f"Wrote {out_csv}")

def process_emg_for_windowing(
    csv_path: str,
    drop_rest: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, got: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"restimulus", "rerepetition", "subject", "exercise"}
    emg_cols = sorted(
        [c for c in df.columns if c.startswith("emg_")],
        key=lambda x: int(x.split("_")[1])
    )

    if not emg_cols:
        raise ValueError(f"No EMG columns found in {csv_path}")
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    x = df[emg_cols].to_numpy(dtype=np.float32)
    labels = df["restimulus"].to_numpy(dtype=np.int64)
    rerepetition = df["rerepetition"].to_numpy(dtype=np.int64)
    subject = int(df["subject"].iloc[0])
    exercise = int(df["exercise"].iloc[0])

    if drop_rest:
        mask = labels != 0
        x = x[mask]
        labels = labels[mask]
        rerepetition = rerepetition[mask]

    return x, labels, rerepetition, subject, exercise

if __name__ == "__main__":
    load_emg_files("/Users/norbertcesar/PycharmProjects/emg_classification/data/input_data")
