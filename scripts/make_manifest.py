#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g3.data import discover_univariate_datasets, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/Users/snowsitu/Downloads/Research/data/raw/tsml_tser")
    parser.add_argument("--out", default="/Users/snowsitu/Downloads/Research/G3/data/manifests/univariate_tser_28.csv")
    args = parser.parse_args()
    metas = discover_univariate_datasets(args.dataset_root)
    df = write_manifest(metas, args.out)
    print(f"wrote {len(df)} datasets to {args.out}")


if __name__ == "__main__":
    main()
