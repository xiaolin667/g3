from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class DatasetMeta:
    name: str
    train_path: Path
    test_path: Path
    univariate: bool
    equal_length: bool
    series_length: int | None
    train_size: int
    test_size: int


def _read_meta(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.lower() == "@data":
                break
            if s.startswith("@"):
                parts = s.split(maxsplit=1)
                meta[parts[0].lower()] = parts[1].strip() if len(parts) > 1 else ""
    return meta


def _count_rows(path: Path) -> int:
    in_data = False
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if not in_data:
                in_data = s.lower() == "@data"
                continue
            n += 1
    return n


def discover_univariate_datasets(dataset_root: str | Path) -> list[DatasetMeta]:
    root = Path(dataset_root)
    out: list[DatasetMeta] = []
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        train = ds_dir / f"{ds_dir.name}_TRAIN.ts"
        test = ds_dir / f"{ds_dir.name}_TEST.ts"
        if not train.exists() or not test.exists():
            continue
        meta = _read_meta(train)
        is_uni = meta.get("@univariate", "").lower() == "true"
        if not is_uni:
            continue
        length = meta.get("@serieslength")
        out.append(
            DatasetMeta(
                name=ds_dir.name,
                train_path=train,
                test_path=test,
                univariate=True,
                equal_length=meta.get("@equallength", "").lower() == "true",
                series_length=int(length) if length and length.lower() != "none" else None,
                train_size=_count_rows(train),
                test_size=_count_rows(test),
            )
        )
    return out


def write_manifest(metas: list[DatasetMeta], path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "dataset": m.name,
                "train_path": str(m.train_path),
                "test_path": str(m.test_path),
                "univariate": m.univariate,
                "equal_length": m.equal_length,
                "series_length": m.series_length,
                "train_size": m.train_size,
                "test_size": m.test_size,
            }
            for m in metas
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    path.with_suffix(".txt").write_text("\n".join(df["dataset"].tolist()) + "\n", encoding="utf-8")
    return df


def _to_float(value: str) -> float:
    value = value.strip()
    if value in {"?", "NaN", "nan", ""}:
        return float("nan")
    return float(value)


def _clean_series(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("empty time series")
    if np.isnan(arr).any():
        idx = np.arange(arr.size)
        good = ~np.isnan(arr)
        if good.any():
            arr = np.interp(idx, idx[good], arr[good]).astype(np.float32)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
    return arr


def load_ts_file(path: str | Path) -> tuple[list[np.ndarray], np.ndarray]:
    path = Path(path)
    in_data = False
    series: list[np.ndarray] = []
    targets: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if not in_data:
                in_data = s.lower() == "@data"
                continue
            parts = s.split(":")
            if len(parts) < 2:
                raise ValueError(f"Expected target label in {path}: {s[:120]}")
            target = _to_float(parts[-1])
            dims = parts[:-1]
            if len(dims) != 1:
                raise ValueError(f"{path} is not univariate at data row with {len(dims)} dimensions")
            values = [_to_float(x) for x in dims[0].split(",") if x.strip()]
            series.append(_clean_series(values))
            targets.append(target)
    return series, np.asarray(targets, dtype=np.float32)


def resample_series(series: list[np.ndarray], length: int) -> np.ndarray:
    tensors = []
    for arr in series:
        x = torch.from_numpy(arr).float().view(1, 1, -1)
        if arr.size != length:
            x = F.interpolate(x, size=length, mode="linear", align_corners=False)
        tensors.append(x.view(length).numpy())
    return np.stack(tensors).astype(np.float32)


@dataclass
class LoadedTser:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    y_mean: float
    y_std: float

    def y_train_std(self) -> np.ndarray:
        return ((self.y_train - self.y_mean) / self.y_std).astype(np.float32)

    def y_test_std(self) -> np.ndarray:
        return ((self.y_test - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_std + self.y_mean


def load_dataset(meta: DatasetMeta, input_length: int) -> LoadedTser:
    train_series, y_train = load_ts_file(meta.train_path)
    test_series, y_test = load_ts_file(meta.test_path)
    x_train = resample_series(train_series, input_length)
    x_test = resample_series(test_series, input_length)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std < 1e-8:
        y_std = 1.0
    return LoadedTser(meta.name, x_train, y_train, x_test, y_test, y_mean, y_std)


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, dataset_index: int = 0, bins: np.ndarray | None = None):
        self.x = torch.from_numpy(x).float().unsqueeze(1)
        self.y = torch.from_numpy(y).float()
        self.dataset_index = int(dataset_index)
        self.bins = torch.from_numpy(bins.astype(np.int64)) if bins is not None else None

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        if self.bins is None:
            return self.x[idx], self.y[idx], self.dataset_index
        return self.x[idx], self.y[idx], self.bins[idx], self.dataset_index
