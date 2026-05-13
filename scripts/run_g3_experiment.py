#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g3.data import discover_univariate_datasets, write_manifest
from g3.metrics import add_average_ranks
from g3.train import finetune_dataset, plot_losses, prepare_loaded_datasets, pretrain_g3, write_g1_style_outputs, write_run_summary
from g3.utils import JsonlLogger, copy_dataset_files, get_device, load_yaml, make_run_dir, save_yaml, set_seed


def resolve_metas(config: dict):
    all_metas = discover_univariate_datasets(config["dataset_root"])
    selected = config.get("selected_datasets", "all_univariate")
    if selected == "all_univariate":
        metas = all_metas
    else:
        wanted = set(selected)
        metas = [m for m in all_metas if m.name in wanted]
        missing = sorted(wanted - {m.name for m in metas})
        if missing:
            raise ValueError(f"Selected datasets are missing or not univariate: {missing}")
    if len(all_metas) != 28:
        raise ValueError(f"Expected 28 univariate datasets in TSER archive, found {len(all_metas)}")
    return metas, all_metas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    set_seed(int(config["seed"]))
    run_dir = make_run_dir(config["output_root"], config["run_name"])
    save_yaml(config, run_dir / "configs" / "resolved_config.yaml")
    logger = JsonlLogger(run_dir / "logs" / "train_events.jsonl")
    device = get_device(config.get("device", "auto"))
    logger.log("run_start", run_dir=str(run_dir), device=str(device), config=args.config)

    metas, all_metas = resolve_metas(config)
    manifest_df = write_manifest(all_metas, run_dir / "data" / "univariate_tser_28.csv")
    write_manifest(metas, run_dir / "data" / "selected_datasets.csv")
    (run_dir / "configs" / "selected_datasets.json").write_text(
        json.dumps({"datasets": [m.name for m in metas]}, indent=2),
        encoding="utf-8",
    )
    logger.log("manifest_written", all_univariate_count=len(manifest_df), selected_count=len(metas))
    if config.get("copy_raw_datasets", False):
        copy_dataset_files(Path(config["dataset_root"]), [m.name for m in metas], run_dir / "data" / "raw" / "tsml_tser")
        logger.log("raw_datasets_copied", count=len(metas), destination=str(run_dir / "data" / "raw" / "tsml_tser"))

    loaded = prepare_loaded_datasets(metas, int(config["model"]["input_length"]), logger)
    seeds = config.get("seeds", [config["seed"]])
    all_metrics = []
    all_losses = []
    for seed in seeds:
        logger.log("seed_start", seed=seed)
        pretrained_state = None
        bank = None
        if config["pretrain"].get("enabled", True):
            pretrained_state, bank, pre_losses = pretrain_g3(config, loaded, device, run_dir, logger, int(seed))
            all_losses.extend(pre_losses)
        for ds in loaded:
            metrics, losses, _ = finetune_dataset(config, ds, device, run_dir, logger, int(seed), pretrained_state, bank)
            all_metrics.append(metrics)
            all_losses.extend(losses)
            plot_losses(losses, run_dir, dataset=ds.name)
        logger.log("seed_complete", seed=seed)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = run_dir / "metrics" / "per_dataset_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    g1_paths = write_g1_style_outputs(run_dir, metrics_df)
    comparison = pd.read_csv(g1_paths["model_comparison"])
    comparison.to_csv(run_dir / "metrics" / "model_comparison.csv", index=False)
    ranks = pd.read_csv(g1_paths["model_average_rank"])
    ranks.to_csv(run_dir / "metrics" / "ranks.csv", index=False)
    plot_losses(all_losses, run_dir, dataset=None)
    write_run_summary(run_dir, config, metrics_df, all_losses)
    logger.log("run_complete", run_dir=str(run_dir), metrics=str(metrics_path))
    print(f"RUN_DIR={run_dir}")
    print(metrics_df.to_string(index=False))
    print("\nMODEL_COMPARISON")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
