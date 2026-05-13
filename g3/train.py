from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, random_split

from .data import ArrayDataset, DatasetMeta, LoadedTser, load_dataset
from .metrics import regression_metrics
from .model import G3Model, PrototypeBank, augment, interval_contrastive_loss, top_shape_loss
from .utils import JsonlLogger, set_seed


def make_interval_bins(values: np.ndarray, intervals: int) -> tuple[np.ndarray, np.ndarray]:
    boundaries = np.quantile(values, np.linspace(0, 1, intervals + 1))
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf
    bins = np.digitize(values, boundaries[1:-1], right=False).astype(np.int64)
    return bins, boundaries.astype(np.float32)


def prepare_loaded_datasets(metas: list[DatasetMeta], input_length: int, logger: JsonlLogger) -> list[LoadedTser]:
    loaded = []
    for meta in metas:
        ds = load_dataset(meta, input_length)
        loaded.append(ds)
        logger.log(
            "dataset_loaded",
            dataset=meta.name,
            train_size=int(ds.x_train.shape[0]),
            test_size=int(ds.x_test.shape[0]),
            y_mean=ds.y_mean,
            y_std=ds.y_std,
        )
    return loaded


def _pretrain_targets(loaded: list[LoadedTser], transform: str) -> np.ndarray:
    ys = []
    for ds in loaded:
        if transform == "raw":
            ys.append(ds.y_train.astype(np.float32))
        elif transform == "dataset_standardized":
            ys.append(ds.y_train_std())
        else:
            raise ValueError(f"Unknown pretrain target transform: {transform}")
    return np.concatenate(ys).astype(np.float32)


def pretrain_g3(
    config: dict,
    loaded: list[LoadedTser],
    device: torch.device,
    run_dir: Path,
    logger: JsonlLogger,
    seed: int,
) -> tuple[dict[str, torch.Tensor], PrototypeBank, list[dict]]:
    set_seed(seed)
    model = G3Model(config).to(device)
    intervals = int(config["pretrain"]["intervals"])
    all_targets = _pretrain_targets(loaded, config["pretrain"].get("target_transform", "dataset_standardized"))
    all_bins, boundaries = make_interval_bins(all_targets, intervals)
    np.save(run_dir / "data" / f"pretrain_interval_boundaries_seed{seed}.npy", boundaries)

    datasets = []
    offset = 0
    for i, ds in enumerate(loaded):
        n = ds.x_train.shape[0]
        target = ds.y_train if config["pretrain"].get("target_transform") == "raw" else ds.y_train_std()
        datasets.append(ArrayDataset(ds.x_train, target.astype(np.float32), i, all_bins[offset : offset + n]))
        offset += n
    loader = DataLoader(
        ConcatDataset(datasets),
        batch_size=int(config["pretrain"]["batch_size"]),
        shuffle=True,
        num_workers=0,
    )

    bank = PrototypeBank(intervals, int(config["model"]["hidden_dim"])).to(device)
    opt = torch.optim.Adam(list(model.encoder.parameters()) + list(model.ipm.parameters()), lr=float(config["pretrain"]["lr"]))
    losses = []
    pre_cfg = config["pretrain"]
    for epoch in range(1, int(pre_cfg["epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, _, bins, _ in loader:
            x = x.to(device)
            bins = bins.to(device)
            opt.zero_grad(set_to_none=True)
            total_loss = 0.0
            cls_for_ema = None
            for _ in range(2):
                x_aug = augment(x, float(pre_cfg["jitter_sigma"]), float(pre_cfg["scale_low"]), float(pre_cfg["scale_high"]))
                cls, shapes, attn = model.encode(x_aug)
                logits = model.ipm(cls)
                probs = torch.softmax(logits, dim=-1)
                ipm_loss = F.cross_entropy(logits, bins)
                ins_loss = interval_contrastive_loss(cls, probs, bins, bank.prototypes, float(pre_cfg["temperature"]))
                shape_loss = top_shape_loss(
                    shapes,
                    attn,
                    probs,
                    bins,
                    bank.prototypes,
                    float(pre_cfg["temperature"]),
                    float(pre_cfg["top_shape_ratio"]),
                )
                total_loss = total_loss + (1.0 - float(pre_cfg["lambda_shape"])) * ins_loss
                total_loss = total_loss + float(pre_cfg["lambda_shape"]) * shape_loss
                total_loss = total_loss + float(pre_cfg["gamma_ipm"]) * ipm_loss
                cls_for_ema = cls.detach()
            total_loss = total_loss / 2.0
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if cls_for_ema is not None:
                bank.ema_update(cls_for_ema, bins, float(pre_cfg["ema_beta"]))
            epoch_loss += float(total_loss.detach().cpu())
            n_batches += 1
        row = {"seed": seed, "stage": "pretrain", "epoch": epoch, "loss": epoch_loss / max(n_batches, 1)}
        losses.append(row)
        logger.log("pretrain_epoch", **row)

    ckpt_path = run_dir / "checkpoints" / f"pretrained_encoder_seed{seed}.pt"
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "ipm": model.ipm.state_dict(),
            "prototypes": bank.state_dict(),
            "config": config,
            "seed": seed,
        },
        ckpt_path,
    )
    logger.log("pretrain_complete", seed=seed, checkpoint=str(ckpt_path))
    return {"encoder": model.encoder.state_dict(), "ipm": model.ipm.state_dict()}, bank, losses


def _split_train_val(dataset: ArrayDataset, seed: int) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    n = len(dataset)
    if n < 5:
        return dataset, dataset
    val_n = max(1, int(round(n * 0.2)))
    train_n = n - val_n
    return random_split(dataset, [train_n, val_n], generator=torch.Generator().manual_seed(seed))


def finetune_dataset(
    config: dict,
    ds: LoadedTser,
    device: torch.device,
    run_dir: Path,
    logger: JsonlLogger,
    seed: int,
    pretrained_state: dict[str, torch.Tensor] | None,
    bank: PrototypeBank | None,
) -> tuple[dict, list[dict], pd.DataFrame]:
    set_seed(seed)
    model = G3Model(config).to(device)
    if pretrained_state is not None:
        model.encoder.load_state_dict(pretrained_state["encoder"])
        model.ipm.load_state_dict(pretrained_state["ipm"])
    if bank is not None:
        bank = bank.to(device)
    train_ds = ArrayDataset(ds.x_train, ds.y_train_std())
    train_split, val_split = _split_train_val(train_ds, seed)
    train_loader = DataLoader(train_split, batch_size=int(config["finetune"]["batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_split, batch_size=int(config["finetune"]["batch_size"]), shuffle=False, num_workers=0)
    test_loader = DataLoader(ArrayDataset(ds.x_test, ds.y_test_std()), batch_size=int(config["finetune"]["batch_size"]), shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=float(config["finetune"]["lr"]))
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    losses = []
    fit_start = time.perf_counter()
    for epoch in range(1, int(config["finetune"]["epochs"]) + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            if bank is not None and float(config["finetune"]["mu_shape"]) > 0:
                cls, shapes, attn = model.encode(x)
                logits = model.ipm(cls.detach())
                probs = torch.softmax(logits, dim=-1).detach()
                bins = probs.argmax(dim=1)
                aux = top_shape_loss(
                    shapes,
                    attn,
                    probs.detach(),
                    bins,
                    bank.prototypes,
                    float(config["pretrain"]["temperature"]),
                    float(config["pretrain"]["top_shape_ratio"]),
                )
                loss = loss + float(config["finetune"]["mu_shape"]) * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss.detach().cpu())
            batches += 1
        val_loss = evaluate_loss(model, val_loader, device)
        row = {
            "seed": seed,
            "dataset": ds.name,
            "stage": "finetune",
            "epoch": epoch,
            "train_loss": train_loss / max(batches, 1),
            "val_loss": val_loss,
        }
        losses.append(row)
        logger.log("finetune_epoch", **row)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(config["finetune"]["patience"]):
                logger.log("early_stop", seed=seed, dataset=ds.name, epoch=epoch, best_val=best_val)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    fit_time_seconds = time.perf_counter() - fit_start
    dataset_ckpt_dir = run_dir / "checkpoints" / ds.name
    dataset_ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = dataset_ckpt_dir / f"best_model_seed{seed}.pt"
    torch.save({"model": model.state_dict(), "config": config, "seed": seed, "dataset": ds.name}, model_path)
    if seed == int(config.get("seed", seed)):
        torch.save({"model": model.state_dict(), "config": config, "seed": seed, "dataset": ds.name}, dataset_ckpt_dir / "best_model.pt")

    predict_start = time.perf_counter()
    y_true_std, y_pred_std = predict(model, test_loader, device)
    predict_time_seconds = time.perf_counter() - predict_start
    y_true = ds.inverse_y(y_true_std)
    y_pred = ds.inverse_y(y_pred_std)
    metric_row = {
        "seed": seed,
        "dataset": ds.name,
        "model": config["model"]["name"],
        "checkpoint": str(model_path),
        "best_val_mse_std": best_val,
        "fit_time_seconds": fit_time_seconds,
        "predict_time_seconds": predict_time_seconds,
        **regression_metrics(y_true, y_pred),
    }
    pred_df = pd.DataFrame({"dataset": ds.name, "seed": seed, "y_true": y_true, "y_pred": y_pred, "error": y_pred - y_true})
    pred_path = run_dir / "predictions" / f"{ds.name}__{config['model']['name']}__seed{seed}.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.log("test_complete", **metric_row, predictions=str(pred_path))
    return metric_row, losses, pred_df


@torch.no_grad()
def evaluate_loss(model: G3Model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        total += float(F.mse_loss(pred, y, reduction="sum").detach().cpu())
        n += int(y.numel())
    return total / max(n, 1)


@torch.no_grad()
def predict(model: G3Model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    preds = []
    for x, y, _ in loader:
        x = x.to(device)
        pred = model(x).detach().cpu().numpy()
        preds.append(pred)
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(preds)


def plot_losses(losses: list[dict], run_dir: Path, dataset: str | None = None) -> None:
    df = pd.DataFrame(losses)
    if df.empty:
        return
    path_csv = run_dir / "metrics" / ("loss_history.csv" if dataset is None else f"{dataset}_loss_history.csv")
    df.to_csv(path_csv, index=False)
    if dataset is not None:
        sub = df[df.get("dataset", dataset) == dataset]
        if not sub.empty and "train_loss" in sub:
            plt.figure(figsize=(8, 4))
            plt.plot(sub["epoch"], sub["train_loss"], label="train")
            plt.plot(sub["epoch"], sub["val_loss"], label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dataset} loss curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(run_dir / "plots" / f"{dataset}_loss_curve.png", dpi=160)
            plt.close()


def write_run_summary(run_dir: Path, config: dict, metrics: pd.DataFrame, losses: list[dict]) -> None:
    datasets = sorted(metrics["dataset"].unique().tolist()) if not metrics.empty else []
    seeds = sorted(int(s) for s in metrics["seed"].unique().tolist()) if not metrics.empty else []
    expected = len(config.get("selected_datasets", datasets)) * len(seeds)
    summary = {
        "run_dir": str(run_dir),
        "selected_dataset_count": len(config.get("selected_datasets", datasets)),
        "completed_dataset_count": len(datasets),
        "remaining_dataset_count": max(0, len(config.get("selected_datasets", datasets)) - len(datasets)),
        "completed_datasets": datasets,
        "remaining_datasets": [d for d in config.get("selected_datasets", []) if d not in datasets],
        "models": [config["model"]["name"]],
        "seeds": seeds,
        "completed_record_count": int(len(metrics)),
        "expected_record_count": int(expected),
        "tables": {
            "dataset_model_summary": str(run_dir / "tables" / "dataset_model_summary.csv"),
            "dataset_ranks": str(run_dir / "tables" / "dataset_ranks.csv"),
            "average_ranks": str(run_dir / "tables" / "average_ranks.csv"),
            "best_models": str(run_dir / "tables" / "best_models.csv"),
        },
        "mean_rmse": float(metrics["rmse"].mean()) if "rmse" in metrics else None,
        "mean_mae": float(metrics["mae"].mean()) if "mae" in metrics else None,
    }
    (run_dir / "metrics" / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


G1_ALL_SEED_COLUMNS = [
    "dataset",
    "model",
    "seed",
    "mse",
    "rmse",
    "mae",
    "r2",
    "fit_time_seconds",
    "predict_time_seconds",
]

G1_SUMMARY_COLUMNS = [
    "dataset",
    "model",
    "mse_mean",
    "mse_std",
    "rmse_mean",
    "rmse_std",
    "mae_mean",
    "mae_std",
    "r2_mean",
    "r2_std",
    "fit_time_seconds_mean",
    "predict_time_seconds_mean",
    "seeds_ran",
]


def write_g1_style_outputs(run_dir: Path, metrics_df: pd.DataFrame) -> dict[str, Path]:
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    all_seed = metrics_df[G1_ALL_SEED_COLUMNS].sort_values(["dataset", "model", "seed"]).reset_index(drop=True)
    all_seed.to_csv(run_dir / "all_seed_metrics.csv", index=False)
    all_seed.to_csv(run_dir / "metrics" / "all_seed_metrics.csv", index=False)

    summary = (
        all_seed.groupby(["dataset", "model"], as_index=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            fit_time_seconds_mean=("fit_time_seconds", "mean"),
            predict_time_seconds_mean=("predict_time_seconds", "mean"),
            seeds_ran=("seed", "nunique"),
        )
        .sort_values(["dataset", "rmse_mean", "model"])
        .reset_index(drop=True)
    )
    summary = summary[G1_SUMMARY_COLUMNS]
    summary.to_csv(run_dir / "dataset_model_summary.csv", index=False)
    summary.to_csv(tables_dir / "dataset_model_summary.csv", index=False)

    ranks = summary.copy()
    ranks["rank"] = ranks.groupby("dataset")["rmse_mean"].rank(method="average", ascending=True)
    avg_rank = (
        ranks.groupby("model", as_index=False)
        .agg(average_rank=("rank", "mean"), datasets=("dataset", "nunique"), average_rmse=("rmse_mean", "mean"))
        .sort_values(["average_rank", "average_rmse", "model"])
        .reset_index(drop=True)
    )
    ranks = ranks.merge(avg_rank[["model", "average_rank"]], on="model", how="left")
    ranks.to_csv(run_dir / "model_comparison.csv", index=False)
    ranks.to_csv(tables_dir / "dataset_ranks.csv", index=False)

    avg_rank.to_csv(run_dir / "model_average_rank.csv", index=False)
    avg_rank.to_csv(tables_dir / "average_ranks.csv", index=False)

    best_seed_records = (
        all_seed.sort_values(["dataset", "model", "rmse", "seed"])
        .groupby(["dataset", "model"], as_index=False)
        .first()
        [G1_ALL_SEED_COLUMNS]
    )
    best_seed_records.to_csv(run_dir / "best_seed_records.csv", index=False)

    best_models = (
        ranks.sort_values(["dataset", "rank", "rmse_mean", "model"])
        .groupby("dataset", as_index=False)
        .first()
        .rename(columns={"model": "best_model", "rank": "dataset_rank"})
        [["dataset", "best_model", "dataset_rank", "rmse_mean", "average_rank"]]
        .rename(columns={"average_rank": "model_average_rank"})
    )
    best_models.to_csv(tables_dir / "best_models.csv", index=False)
    return {
        "all_seed_metrics": run_dir / "all_seed_metrics.csv",
        "model_comparison": run_dir / "model_comparison.csv",
        "dataset_model_summary": run_dir / "dataset_model_summary.csv",
        "model_average_rank": run_dir / "model_average_rank.csv",
        "best_seed_records": run_dir / "best_seed_records.csv",
        "average_ranks": tables_dir / "average_ranks.csv",
        "best_models": tables_dir / "best_models.csv",
        "dataset_ranks": tables_dir / "dataset_ranks.csv",
        "table_dataset_model_summary": tables_dir / "dataset_model_summary.csv",
    }
