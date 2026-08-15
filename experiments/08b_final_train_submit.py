#!/usr/bin/env python3
"""Train the validated tree + TabM ensemble and build submit_04.zip.

The schema-3 LightGBM/CatBoost R/F route is reproduced exactly.  A TabM model
is trained on all 2019-2024 regular-season rows with the experiment-08 Brier
recipe.  On R rows, the final prediction is a 70:30 probability blend of the
tree route and TabM; F rows remain exactly equal to the schema-3 route.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import platform
import shutil
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from features import (
    FeatureConfig,
    ID_COL,
    TARGET_COL,
    add_rowwise_features,
    categorical_columns,
)


def load_python_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import Python module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


V3 = load_python_module(
    "aimers_submission_v3_builder",
    PROJECT_ROOT / "experiments/07c_train_submit.py",
)
CV = load_python_module(
    "aimers_tabm_cv_recipe",
    PROJECT_ROOT / "experiments/08_tabm_cv.py",
)
BASE = V3.BASE


PACKAGE_SCHEMA_VERSION = 4
EXPECTED_TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
REFERENCE_SEASON = 2025
EXPECTED_TABM_SIGNATURE = "a960bb6f17e7"
EXPECTED_TABM_CANDIDATE = "base_tabm_brier_w30"
EXPECTED_TABM_COLUMN = "blend_brier_w30"
TABM_LOSS = "brier"
TABM_WEIGHT = 0.30
TREE_WEIGHT = 0.70
SUBMISSION_FILENAME = "submit_04.zip"
SUBMISSION_REQUIREMENTS = (
    "# Evaluation image supplies numpy, pandas, and joblib.\n"
    "lightgbm==4.6.0\n"
    "catboost==1.2.8\n"
    "tabm==0.0.3\n"
)
EXPECTED_ZIP_NAMES = {
    "model/lgbm_model.txt",
    "model/catboost_base.cbm",
    "model/catboost_f_route.cbm",
    "model/tabm_state.pt",
    "model/artifacts.joblib",
    "script.py",
    "requirements.txt",
}
NEURAL_TOLERANCE = 3e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path", type=Path, default=PROJECT_ROOT / "data/train.csv"
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/final_submission_v4",
    )
    parser.add_argument(
        "--routing-latest-path",
        type=Path,
        default=PROJECT_ROOT / "outputs/f_routing_v1/latest_run.json",
    )
    parser.add_argument(
        "--tabm-latest-path",
        type=Path,
        default=PROJECT_ROOT / "outputs/tabm_cv_v1/latest_run.json",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--lgbm-estimators", type=int, default=500)
    parser.add_argument("--catboost-iterations", type=int, default=85)
    parser.add_argument(
        "--catboost-task-type", choices=["CPU", "GPU"], default="GPU"
    )
    parser.add_argument(
        "--tabm-device", choices=["auto", "cuda", "cpu"], default="auto"
    )
    parser.add_argument("--tabm-batch-size", type=int, default=4096)
    parser.add_argument("--tabm-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"cannot interpret boolean value: {value!r}")


def resolve_recorded_run(latest_path: Path, run_dir_value: str) -> Path:
    recorded = Path(run_dir_value)
    candidates = [recorded, latest_path.parent / recorded.name]
    run_dir = next((path for path in candidates if path.exists()), None)
    if run_dir is None:
        raise FileNotFoundError(f"recorded run directory is missing: {candidates}")
    return run_dir


def load_and_validate_tabm_selection(latest_path: Path) -> dict:
    if not latest_path.exists():
        raise FileNotFoundError(
            f"08 latest result is missing: {latest_path}. Run notebook 08 first."
        )
    with latest_path.open(encoding="utf-8") as handle:
        latest = json.load(handle)
    summary = latest.get("summary", {})
    checks = {
        "signature": summary.get("signature") == EXPECTED_TABM_SIGNATURE,
        "selected_candidate": (
            summary.get("selected_candidate") == EXPECTED_TABM_CANDIDATE
        ),
        "selection_status": (
            summary.get("selection_status")
            == "best_predeclared_candidate_passed"
        ),
        "selected_prediction_column": (
            summary.get("selected_prediction_column") == EXPECTED_TABM_COLUMN
        ),
        "source_07b_signature": (
            summary.get("source_metadata", {}).get("source_signature")
            == V3.EXPECTED_ROUTING_SIGNATURE
        ),
        "source_07b_route": (
            summary.get("source_metadata", {}).get("selected_route")
            == V3.SELECTED_ROUTE
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"08 selection validation failed: {failed}")

    run_dir = resolve_recorded_run(latest_path, latest["run_dir"])
    required = [
        run_dir / "configuration.json",
        run_dir / "blend_summary.csv",
        run_dir / "integrity_audit.json",
        run_dir / "summary.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"08 audit artifacts are missing: {missing}")
    with (run_dir / "configuration.json").open(encoding="utf-8") as handle:
        configuration = json.load(handle)
    with (run_dir / "integrity_audit.json").open(encoding="utf-8") as handle:
        integrity = json.load(handle)
    blend_summary = pd.read_csv(run_dir / "blend_summary.csv")
    selected_rows = blend_summary[
        blend_summary["model"].eq(EXPECTED_TABM_CANDIDATE)
    ]
    if len(selected_rows) != 1:
        raise RuntimeError("expected one selected TabM blend row")
    selected = selected_rows.iloc[0]
    config_checks = {
        "losses": configuration.get("losses") == ["bce", "brier"],
        "half_life": float(configuration.get("half_life", -1.0)) == 1.5,
        "k": int(configuration.get("k", -1)) == 32,
        "n_blocks": int(configuration.get("n_blocks", -1)) == 3,
        "d_block": int(configuration.get("d_block", -1)) == 256,
        "dropout": float(configuration.get("dropout", -1.0)) == 0.10,
        "learning_rate": (
            float(configuration.get("learning_rate", -1.0)) == 0.002
        ),
        "weight_decay": (
            float(configuration.get("weight_decay", -1.0)) == 0.0003
        ),
        "training_scope": configuration.get("training_scope") == "R_only",
        "script_hash": (
            configuration.get("script_sha256")
            == BASE.sha256_file(PROJECT_ROOT / "experiments/08_tabm_cv.py")
        ),
    }
    integrity_checks = {
        "integrity_status": integrity.get("status") == "passed",
        "season_excluded": integrity.get("season_feature_excluded_all_folds") is True,
        "game_type_excluded": (
            integrity.get("game_type_feature_excluded_all_folds") is True
        ),
        "temporal_te": (
            integrity.get("temporal_target_encoding_passed_all_folds") is True
        ),
        "f_exact": integrity.get("F_predictions_exact") is True,
        "selected_gate": _as_bool(selected["passes_fixed_gate"]),
        "selected_weight": EXPECTED_TABM_CANDIDATE.endswith("w30"),
    }
    failed = [
        name
        for name, passed in {**config_checks, **integrity_checks}.items()
        if not passed
    ]
    if failed:
        raise RuntimeError(f"08 configuration/integrity validation failed: {failed}")
    metrics = {
        "recent_r_brier": float(selected["recent_r_brier"]),
        "recent_r_bss_raw": float(selected["recent_r_bss_raw"]),
        "delta_recent_r_brier_vs_base": float(
            selected["delta_recent_r_brier_vs_base"]
        ),
        "worst_fold_delta": float(selected["worst_fold_delta"]),
        "fold_r_brier": {
            str(season): float(selected[f"r_brier_{season}"])
            for season in (2022, 2023, 2024)
        },
        "fold_delta_r_brier_vs_base": {
            str(season): float(selected[f"delta_r_brier_{season}_vs_base"])
            for season in (2022, 2023, 2024)
        },
    }
    return {
        "latest": latest,
        "run_dir": run_dir,
        "selection_checks": checks,
        "configuration_checks": config_checks,
        "integrity_checks": integrity_checks,
        "selected_metrics": metrics,
        "configuration": configuration,
        "integrity": integrity,
    }


def resolve_tabm_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TabM CUDA training was requested, but CUDA is unavailable")
    return torch.device(requested)


def fit_final_target_encoding_statistics(
    train: pd.DataFrame,
    *,
    reference_season: int,
    half_life: float,
) -> tuple[dict, float]:
    keys = CV.target_encoding_keys(train)
    target = train[TARGET_COL].to_numpy(dtype=np.float32)
    weights = CV.recency_weights(
        train["season"],
        reference_season=reference_season,
        half_life=half_life,
        normalize=False,
    )
    prior = CV.weighted_mean(target, weights)
    groups = {}
    for group_name, definition in CV.TARGET_ENCODING_GROUPS.items():
        columns = list(definition["columns"])
        stats = CV.fit_group_statistics(keys, target, weights, columns)
        groups[group_name] = {
            "columns": columns,
            "strength": float(definition["strength"]),
            "stats": stats,
        }
    return groups, prior


def prepare_final_tabm_training_features(
    train: pd.DataFrame,
    *,
    reference_season: int,
    half_life: float,
    architecture: dict,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    train = train.reset_index(drop=True)
    raw_weights = CV.recency_weights(
        train["season"],
        reference_season=reference_season,
        half_life=half_life,
        normalize=False,
    )
    prior = CV.weighted_mean(train[TARGET_COL].to_numpy(), raw_weights)
    rowwise = add_rowwise_features(
        train, target_prior=prior, config=CV.FEATURE_CONFIG
    )
    categorical = categorical_columns(rowwise)
    numeric = [column for column in rowwise.columns if column not in categorical]
    unexpected = [
        column
        for column in numeric
        if not pd.api.types.is_numeric_dtype(rowwise[column])
    ]
    if unexpected:
        raise TypeError(f"unexpected TabM non-numeric columns: {unexpected}")

    # Training rows receive encodings from strictly earlier seasons.  The
    # one-row probe is ignored; it lets the shared CV helper build the train matrix.
    train_te, _, temporal_audit = CV.build_temporal_target_encodings(
        train,
        train.iloc[[0]],
        valid_season=reference_season,
        half_life=half_life,
    )
    if temporal_audit["status"] != "passed":
        raise RuntimeError("final temporal target-encoding audit failed")
    groups, inference_prior = fit_final_target_encoding_statistics(
        train,
        reference_season=reference_season,
        half_life=half_life,
    )
    if abs(prior - inference_prior) > 1e-12:
        raise RuntimeError("TabM weighted priors do not match")

    numeric_frame = rowwise.loc[:, numeric].replace([np.inf, -np.inf], np.nan)
    numeric_frame = pd.concat(
        [numeric_frame.reset_index(drop=True), train_te], axis=1
    )
    pre_missing_columns = numeric_frame.columns.tolist()
    medians = numeric_frame.median(axis=0).fillna(0.0)
    missing_sources = [
        column for column in pre_missing_columns if numeric_frame[column].isna().any()
    ]
    filled = numeric_frame.fillna(medians)
    if missing_sources:
        indicator_names = {
            column: f"{column}__missing" for column in missing_sources
        }
        indicators = (
            numeric_frame.loc[:, missing_sources]
            .isna()
            .astype("float32")
            .rename(columns=indicator_names)
        )
        filled = pd.concat([filled, indicators], axis=1)
    final_numeric_columns = filled.columns.tolist()
    means = filled.mean(axis=0)
    stds = filled.std(axis=0, ddof=0).fillna(0.0)
    safe_stds = stds.mask(stds < 1e-6, 1.0)
    x_num = ((filled - means) / safe_stds).clip(-10.0, 10.0).to_numpy(
        dtype=np.float32, copy=True
    )
    if not np.isfinite(x_num).all():
        raise RuntimeError("non-finite final TabM numerical feature")

    x_cat = np.empty((len(train), len(categorical)), dtype=np.int32)
    category_maps = {}
    cardinalities = []
    embedding_dimensions = []
    category_audit = {}
    for index, column in enumerate(categorical):
        values = rowwise[column].astype("string").fillna("__MISSING__")
        categories = values.unique().tolist()
        codes = pd.Categorical(values, categories=categories).codes.astype("int32") + 1
        cardinality = len(categories) + 1
        dimension = CV.embedding_dimension(cardinality)
        x_cat[:, index] = codes
        category_maps[column] = categories
        cardinalities.append(cardinality)
        embedding_dimensions.append(dimension)
        category_audit[column] = {
            "cardinality_with_unknown": cardinality,
            "embedding_dimension": dimension,
        }

    metadata = {
        "model_filename": "tabm_state.pt",
        "feature_bundle": "all_new_plus_temporal_target_encoding",
        "feature_config": asdict(CV.FEATURE_CONFIG),
        "target_prior": prior,
        "rowwise_feature_columns": rowwise.columns.tolist(),
        "rowwise_numeric_columns": numeric,
        "pre_missing_numeric_columns": pre_missing_columns,
        "numeric_feature_columns": final_numeric_columns,
        "numeric_medians": {column: float(medians[column]) for column in medians.index},
        "numeric_means": {column: float(means[column]) for column in means.index},
        "numeric_stds": {
            column: float(safe_stds[column]) for column in safe_stds.index
        },
        "missing_indicator_sources": missing_sources,
        "categorical_columns": categorical,
        "category_maps": category_maps,
        "cardinalities": cardinalities,
        "embedding_dimensions": embedding_dimensions,
        "n_num_features": int(x_num.shape[1]),
        "target_encoding": {
            "prior": prior,
            "groups": groups,
            "fit_rows": int(len(train)),
            "max_source_season": int(train["season"].max()),
            "reference_season": reference_season,
            "uses_training_labels_only": True,
        },
        "architecture": architecture,
        "loss": TABM_LOSS,
        "inference_batch_size_gpu": 4096,
        "inference_batch_size_cpu": 1024,
    }
    audit = {
        "status": "passed",
        "training_rows": int(len(train)),
        "training_seasons": sorted(int(value) for value in train["season"].unique()),
        "training_game_type": "R",
        "target_prior": prior,
        "rowwise_feature_count": int(rowwise.shape[1]),
        "numeric_feature_count": int(x_num.shape[1]),
        "categorical_feature_count": int(x_cat.shape[1]),
        "category_audit": category_audit,
        "temporal_training_encodings_passed": bool(
            temporal_audit["training_encodings_use_strictly_earlier_seasons"]
        ),
        "inference_statistics_max_source_season": int(train["season"].max()),
        "inference_reference_season": reference_season,
    }
    return x_num, x_cat, metadata, audit


def prepare_final_tabm_inference_features(
    frame: pd.DataFrame,
    metadata: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Independently reproduce the frozen final-inference transformation.

    Training rows use strictly historical target encodings, whereas inference
    rows use statistics fitted on every available R training row.  Keeping this
    builder-side implementation separate from ``script.py`` makes the package
    parity check meaningful without incorrectly comparing those two regimes.
    """

    rowwise = add_rowwise_features(
        frame,
        target_prior=float(metadata["target_prior"]),
        config=FeatureConfig(**metadata["feature_config"]),
    )
    if list(rowwise.columns) != metadata["rowwise_feature_columns"]:
        raise RuntimeError("TabM inference rowwise feature order changed")

    target_encoding = metadata["target_encoding"]
    keys = CV.target_encoding_keys(frame)
    encoded_columns: dict[str, np.ndarray] = {}
    prior = float(target_encoding["prior"])
    for group_name, definition in target_encoding["groups"].items():
        mean, count = CV.apply_group_statistics(
            keys,
            definition["stats"],
            group_columns=list(definition["columns"]),
            prior=prior,
            strength=float(definition["strength"]),
        )
        encoded_columns[f"te_{group_name}_mean"] = mean
        encoded_columns[f"te_{group_name}_delta"] = mean - prior
        encoded_columns[f"te_{group_name}_log_count"] = np.log1p(count)

    numeric = rowwise.loc[:, metadata["rowwise_numeric_columns"]].replace(
        [np.inf, -np.inf], np.nan
    )
    numeric = pd.concat(
        [numeric.reset_index(drop=True), pd.DataFrame(encoded_columns)], axis=1
    )
    pre_missing_columns = metadata["pre_missing_numeric_columns"]
    if list(numeric.columns) != pre_missing_columns:
        raise RuntimeError("TabM inference numerical feature order changed")
    medians = pd.Series(metadata["numeric_medians"], dtype="float64").reindex(
        pre_missing_columns
    )
    filled = numeric.fillna(medians)
    missing_sources = metadata["missing_indicator_sources"]
    if missing_sources:
        indicators = (
            numeric.loc[:, missing_sources]
            .isna()
            .astype("float32")
            .rename(columns={c: f"{c}__missing" for c in missing_sources})
        )
        filled = pd.concat([filled, indicators], axis=1)
    final_columns = metadata["numeric_feature_columns"]
    if list(filled.columns) != final_columns:
        raise RuntimeError("TabM inference post-imputation feature order changed")
    means = pd.Series(metadata["numeric_means"], dtype="float64").reindex(
        final_columns
    )
    stds = pd.Series(metadata["numeric_stds"], dtype="float64").reindex(
        final_columns
    )
    x_num = ((filled - means) / stds).clip(-10.0, 10.0).to_numpy(
        dtype=np.float32, copy=True
    )
    if not np.isfinite(x_num).all():
        raise RuntimeError("non-finite final TabM inference numerical feature")

    categorical = metadata["categorical_columns"]
    x_cat = np.empty((len(frame), len(categorical)), dtype=np.int32)
    for index, column in enumerate(categorical):
        values = rowwise[column].astype("string").fillna("__MISSING__")
        categories = metadata["category_maps"][column]
        x_cat[:, index] = (
            pd.Categorical(values, categories=categories).codes.astype("int32") + 1
        )
    return x_num, x_cat


@torch.inference_mode()
def predict_tabm_arrays(
    model,
    x_num: np.ndarray,
    x_cat: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(x_num), batch_size):
        stop = min(start + batch_size, len(x_num))
        batch_num = torch.from_numpy(x_num[start:stop]).to(
            device, non_blocking=True
        )
        batch_cat = torch.from_numpy(x_cat[start:stop]).to(
            device, dtype=torch.long, non_blocking=True
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(batch_num, batch_cat).squeeze(-1)
        output.append(torch.sigmoid(logits.float()).mean(dim=1).cpu().numpy())
    return np.clip(
        np.concatenate(output).astype(np.float64),
        CV.PROBABILITY_EPS,
        1.0 - CV.PROBABILITY_EPS,
    )


def train_final_tabm(
    data: pd.DataFrame,
    *,
    args: argparse.Namespace,
    model_path: Path,
    selection: dict,
) -> tuple[
    dict,
    dict,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    dict,
]:
    train = data[data["game_type"].eq("R")].reset_index(drop=True)
    configuration = selection["configuration"]
    architecture = {
        "k": int(configuration["k"]),
        "n_blocks": int(configuration["n_blocks"]),
        "d_block": int(configuration["d_block"]),
        "dropout": float(configuration["dropout"]),
    }
    half_life = float(configuration["half_life"])
    x_num, x_cat, metadata, preprocessing_audit = (
        prepare_final_tabm_training_features(
            train,
            reference_season=REFERENCE_SEASON,
            half_life=half_life,
            architecture=architecture,
        )
    )
    target = train[TARGET_COL].to_numpy(dtype=np.float32)
    train_weight = CV.recency_weights(
        train["season"],
        reference_season=REFERENCE_SEASON,
        half_life=half_life,
        normalize=True,
    )
    device = resolve_tabm_device(args.tabm_device)
    seed = int(configuration["seed"]) + REFERENCE_SEASON
    CV.set_global_seed(seed)
    model = CV.EmbeddingTabM(
        n_num_features=x_num.shape[1],
        cardinalities=metadata["cardinalities"],
        embedding_dimensions=metadata["embedding_dimensions"],
        k=architecture["k"],
        n_blocks=architecture["n_blocks"],
        d_block=architecture["d_block"],
        dropout=architecture["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(configuration["learning_rate"]),
        weight_decay=float(configuration["weight_decay"]),
    )
    scaler = CV.make_grad_scaler(device.type == "cuda")
    dataset = TensorDataset(
        torch.from_numpy(x_num),
        torch.from_numpy(x_cat),
        torch.from_numpy(target),
        torch.from_numpy(train_weight.astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=args.tabm_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(
        f"Training TabM Brier | R rows={len(train):,} "
        f"numeric={x_num.shape[1]} categorical={x_cat.shape[1]} "
        f"device={device} epochs={args.tabm_epochs}"
    )
    started = time.perf_counter()
    history = []
    for epoch in range(1, args.tabm_epochs + 1):
        model.train()
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        epoch_started = time.perf_counter()
        for batch_num, batch_cat, batch_target, batch_weight in loader:
            batch_num = batch_num.to(device, non_blocking=True)
            batch_cat = batch_cat.to(device, dtype=torch.long, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            batch_weight = batch_weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(batch_num, batch_cat)
                loss = CV.tabm_loss(
                    logits, batch_target, batch_weight, TABM_LOSS
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            batch_weight_sum = float(batch_weight.sum().detach().cpu())
            weighted_loss_sum += float(loss.detach().cpu()) * batch_weight_sum
            weight_sum += batch_weight_sum
        history.append(
            {
                "epoch": epoch,
                "train_brier_loss": weighted_loss_sum / weight_sum,
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
        )
        print(
            f"  epoch={epoch:02d} "
            f"train_brier={history[-1]['train_brier_loss']:.6f} "
            f"seconds={history[-1]['epoch_seconds']:.1f}"
        )

    audit_indices = np.unique(
        np.linspace(0, len(train) - 1, num=min(256, len(train)), dtype=int)
    )
    audit_raw = train.iloc[audit_indices].copy()
    audit_num, audit_cat = prepare_final_tabm_inference_features(
        audit_raw, metadata
    )
    audit_reference = predict_tabm_arrays(
        model,
        audit_num,
        audit_cat,
        batch_size=args.tabm_batch_size,
        device=device,
    )
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    torch.save(state, model_path)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    elapsed = time.perf_counter() - started
    summary = {
        "train_rows": int(len(train)),
        "train_seasons": sorted(int(value) for value in train["season"].unique()),
        "train_game_type": "R",
        "target_rate": float(train[TARGET_COL].mean()),
        "weighted_target_prior": float(metadata["target_prior"]),
        "reference_season": REFERENCE_SEASON,
        "half_life": half_life,
        "loss": TABM_LOSS,
        "epochs": args.tabm_epochs,
        "seed": seed,
        "device": str(device),
        "parameter_count": int(parameter_count),
        "numeric_feature_count": int(x_num.shape[1]),
        "categorical_feature_count": int(x_cat.shape[1]),
        "architecture": architecture,
        "history": history,
        "train_seconds": elapsed,
        "preprocessing_audit": preprocessing_audit,
    }
    del model, optimizer, scaler, loader, dataset, state, x_num, x_cat
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        metadata,
        summary,
        audit_reference,
        audit_raw,
        audit_num,
        audit_cat,
        preprocessing_audit,
    )


def max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.float64)
                - np.asarray(right, dtype=np.float64)
            )
        )
    )


def verify_tabm_feature_parity(
    *,
    runtime,
    raw_rows: pd.DataFrame,
    reference_num: np.ndarray,
    reference_cat: np.ndarray,
    metadata: dict,
) -> dict:
    runtime_num, runtime_cat = runtime.prepare_tabm_features(raw_rows, metadata)
    numeric_diff = max_abs_diff(reference_num, runtime_num)
    categorical_exact = np.array_equal(reference_cat, runtime_cat)
    if numeric_diff > 1e-6 or not categorical_exact:
        raise RuntimeError(
            "TabM final-inference/runtime feature mismatch: "
            f"numeric={numeric_diff}, categorical_exact={categorical_exact}"
        )
    return {
        "status": "passed",
        "rows": int(len(raw_rows)),
        "numeric_max_abs_difference": numeric_diff,
        "categorical_exact": categorical_exact,
    }


def verify_serialized_ensemble(
    *,
    script_path: Path,
    model_dir: Path,
    full_data: pd.DataFrame,
    lgbm_audit_raw: pd.DataFrame,
    lgbm_reference: np.ndarray,
    base_audit_raw: pd.DataFrame,
    base_reference: np.ndarray,
    f_route_audit_raw: pd.DataFrame,
    f_route_reference: np.ndarray,
    tabm_audit_raw: pd.DataFrame,
    tabm_reference: np.ndarray,
) -> dict:
    runtime = load_python_module("aimers_v4_runtime", script_path)
    predictor = runtime.load_predictor(model_dir)
    lgbm_reloaded, _ = runtime.predict_components(lgbm_audit_raw, predictor)
    base_reloaded = runtime.predict_catboost_model(
        base_audit_raw, predictor, "catboost_base"
    )
    f_route_reloaded = runtime.predict_catboost_model(
        f_route_audit_raw, predictor, "catboost_f_route"
    )
    tabm_reloaded = runtime.predict_tabm_probabilities(tabm_audit_raw, predictor)
    reload_diffs = {
        "lgbm": max_abs_diff(lgbm_reference, lgbm_reloaded),
        "catboost_base": max_abs_diff(base_reference, base_reloaded),
        "catboost_f_route": max_abs_diff(f_route_reference, f_route_reloaded),
        "tabm": max_abs_diff(tabm_reference, tabm_reloaded),
    }
    if max(
        reload_diffs["lgbm"],
        reload_diffs["catboost_base"],
        reload_diffs["catboost_f_route"],
    ) > 1e-12 or reload_diffs["tabm"] > NEURAL_TOLERANCE:
        raise RuntimeError(f"serialized model reload mismatch: {reload_diffs}")

    generator = np.random.default_rng(2026)
    r_positions = np.flatnonzero(
        full_data["game_type"].eq("R").to_numpy(dtype=np.bool_)
    )
    f_positions = np.flatnonzero(
        full_data["game_type"].eq("F").to_numpy(dtype=np.bool_)
    )
    audit = pd.concat(
        [
            full_data.iloc[
                generator.choice(r_positions, size=min(32, len(r_positions)), replace=False)
            ],
            full_data.iloc[
                generator.choice(f_positions, size=min(32, len(f_positions)), replace=False)
            ],
        ],
        ignore_index=True,
    )
    lgbm_probability, routed_catboost = runtime.predict_components(audit, predictor)
    base_tree = runtime.blend_logit(
        lgbm_probability,
        routed_catboost,
        lgbm_weight=V3.LGBM_WEIGHT,
    )
    is_r = audit["game_type"].eq("R").to_numpy(dtype=np.bool_)
    is_f = np.logical_not(is_r)
    expected = base_tree.copy()
    expected[is_r] = (
        TREE_WEIGHT * base_tree[is_r]
        + TABM_WEIGHT
        * runtime.predict_tabm_probabilities(audit.iloc[np.flatnonzero(is_r)], predictor)
    )
    actual = runtime.predict_probabilities(audit, predictor)
    composition_diff = max_abs_diff(actual, expected)
    f_exact_diff = max_abs_diff(actual[is_f], base_tree[is_f])
    if composition_diff > NEURAL_TOLERANCE or f_exact_diff != 0.0:
        raise RuntimeError(
            f"final ensemble composition mismatch: {composition_diff}, F={f_exact_diff}"
        )

    shuffled = audit.sample(frac=1.0, random_state=2027)
    shuffled_prediction = runtime.predict_probabilities(shuffled, predictor)
    shuffled_by_id = pd.Series(
        shuffled_prediction, index=shuffled[ID_COL].astype("string").to_numpy()
    )
    reordered = shuffled_by_id.reindex(
        audit[ID_COL].astype("string").to_numpy()
    ).to_numpy()
    chunks = np.array_split(audit, min(7, len(audit)))
    chunked = np.concatenate(
        [runtime.predict_probabilities(chunk, predictor) for chunk in chunks if len(chunk)]
    )
    single = np.array(
        [
            runtime.predict_probabilities(audit.iloc[[index]], predictor)[0]
            for index in range(min(10, len(audit)))
        ]
    )
    independence_diffs = {
        "shuffle": max_abs_diff(actual, reordered),
        "chunk": max_abs_diff(actual, chunked),
        "single_row": max_abs_diff(actual[: len(single)], single),
    }
    if max(independence_diffs.values()) > NEURAL_TOLERANCE:
        raise RuntimeError(f"row-independence test failed: {independence_diffs}")
    return {
        "status": "passed",
        "native_reload_max_abs_difference": reload_diffs,
        "final_composition_max_abs_difference": composition_diff,
        "F_vs_schema3_base_max_abs_difference": f_exact_diff,
        "row_independence_max_abs_difference": independence_diffs,
        "audit_rows": int(len(audit)),
        "audit_r_rows": int(is_r.sum()),
        "audit_f_rows": int(is_f.sum()),
        "tabm_applied_only_to_R": True,
        "F_predictions_exact_schema3_base": True,
    }


def build_and_verify_zip(package_dir: Path, zip_path: Path) -> dict:
    source_by_name = {
        "model/lgbm_model.txt": package_dir / "model/lgbm_model.txt",
        "model/catboost_base.cbm": package_dir / "model/catboost_base.cbm",
        "model/catboost_f_route.cbm": package_dir / "model/catboost_f_route.cbm",
        "model/tabm_state.pt": package_dir / "model/tabm_state.pt",
        "model/artifacts.joblib": package_dir / "model/artifacts.joblib",
        "script.py": package_dir / "script.py",
        "requirements.txt": package_dir / "requirements.txt",
    }
    for name, source in source_by_name.items():
        if not source.is_file():
            raise FileNotFoundError(f"missing package member {name}: {source}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source in source_by_name.items():
            archive.write(source, arcname=name)
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        if names != EXPECTED_ZIP_NAMES:
            raise RuntimeError(
                f"ZIP structure mismatch: missing={sorted(EXPECTED_ZIP_NAMES - names)}, "
                f"extra={sorted(names - EXPECTED_ZIP_NAMES)}"
            )
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise RuntimeError(f"corrupt ZIP member: {corrupt_member}")
        members = {
            info.filename: {
                "bytes": info.file_size,
                "compressed_bytes": info.compress_size,
            }
            for info in archive.infolist()
        }
    if zip_path.stat().st_size >= 10 * 1024**3:
        raise RuntimeError("submission ZIP exceeds the 10 GiB competition limit")
    return {
        "zip_path": str(zip_path),
        "zip_bytes": int(zip_path.stat().st_size),
        "zip_sha256": BASE.sha256_file(zip_path),
        "members": members,
    }


def main() -> None:
    args = parse_args()
    args.threads = max(1, min(8, args.threads))
    if args.tabm_epochs != 1:
        raise ValueError("08 recent folds selected one epoch; tabm_epochs must remain 1")
    output_dir = args.output_dir.resolve()
    package_dir = output_dir / "package"
    model_dir = package_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    routing_selection = V3.load_and_validate_routing_selection(
        args.routing_latest_path
    )
    tabm_selection = load_and_validate_tabm_selection(args.tabm_latest_path)
    runtime_template = SRC_DIR / "submission_entrypoint.py"
    if not runtime_template.exists():
        raise FileNotFoundError(f"submission runtime is missing: {runtime_template}")
    shutil.copy2(runtime_template, package_dir / "script.py")
    (package_dir / "requirements.txt").write_text(
        SUBMISSION_REQUIREMENTS, encoding="utf-8"
    )

    total_started = time.perf_counter()
    print(f"Loading {args.train_path} ...")
    data = pd.read_csv(args.train_path, encoding="utf-8-sig", low_memory=False)
    BASE.validate_training_data(data)
    seasons = sorted(int(value) for value in data["season"].unique())
    if seasons != EXPECTED_TRAIN_SEASONS:
        raise ValueError(
            f"unexpected training seasons: expected={EXPECTED_TRAIN_SEASONS}, got={seasons}"
        )
    if not set(data["game_type"].astype(str).unique()).issubset({"R", "F"}):
        raise ValueError("game_type must contain only R and F")
    raw_input_columns = [column for column in data.columns if column != TARGET_COL]
    print(
        f"rows={len(data):,} seasons={seasons} "
        f"selected_TabM={EXPECTED_TABM_CANDIDATE}"
    )
    tree_feature_parity = V3.verify_training_inference_feature_parity(
        data, runtime_template
    )

    (
        lgbm_metadata,
        lgbm_summary,
        lgbm_reference,
        lgbm_audit_raw,
        lgbm_importance,
    ) = V3.train_lgbm(
        data, args=args, model_path=model_dir / "lgbm_model.txt"
    )
    lgbm_importance.to_csv(
        output_dir / "lgbm_feature_importance.csv", index=False
    )
    del lgbm_importance

    (
        base_metadata,
        base_summary,
        base_reference,
        base_audit_raw,
        base_importance,
    ) = V3.train_catboost(
        data,
        args=args,
        model_path=model_dir / "catboost_base.cbm",
        model_label="CatBoost base all_new",
        training_policy="all_2019_2024_R_F",
    )
    base_importance.to_csv(
        output_dir / "catboost_base_feature_importance.csv", index=False
    )
    del base_importance

    f_route_train = V3.select_f_route_training_rows(data)
    (
        f_route_metadata,
        f_route_summary,
        f_route_reference,
        f_route_audit_raw,
        f_route_importance,
    ) = V3.train_catboost(
        f_route_train,
        args=args,
        model_path=model_dir / "catboost_f_route.cbm",
        model_label="CatBoost F route shared_latest_f1",
        training_policy="all_R_2019_2024_plus_F_2024",
    )
    f_route_importance.to_csv(
        output_dir / "catboost_f_route_feature_importance.csv", index=False
    )
    del f_route_importance, f_route_train
    gc.collect()

    (
        tabm_metadata,
        tabm_summary,
        tabm_reference,
        tabm_audit_raw,
        tabm_audit_num,
        tabm_audit_cat,
        tabm_preprocessing_audit,
    ) = train_final_tabm(
        data,
        args=args,
        model_path=model_dir / "tabm_state.pt",
        selection=tabm_selection,
    )

    artifacts = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "competition": "LG Aimers 9th",
        "submission_version": 4,
        "target_column": TARGET_COL,
        "raw_input_columns": raw_input_columns,
        "selected_pair": V3.SELECTED_PAIR,
        "selected_route": V3.SELECTED_ROUTE,
        "selected_neural_candidate": EXPECTED_TABM_CANDIDATE,
        "lgbm": lgbm_metadata,
        "catboost_base": base_metadata,
        "catboost_f_route": f_route_metadata,
        "tabm": tabm_metadata,
        "routing": {
            "column": "game_type",
            "base_value": "R",
            "base_model": "catboost_base",
            "f_route_value": "F",
            "f_route_model": "catboost_f_route",
            "row_independent": True,
        },
        "blend": {
            "method": V3.BLEND_METHOD,
            "lgbm_weight": V3.LGBM_WEIGHT,
            "catboost_weight": V3.CATBOOST_WEIGHT,
            "calibration": "none",
        },
        "neural_blend": {
            "method": "probability",
            "scope": "R_only",
            "tree_weight": TREE_WEIGHT,
            "tabm_weight": TABM_WEIGHT,
            "F_policy": "schema3_tree_route_exact",
        },
    }
    joblib.dump(artifacts, model_dir / "artifacts.joblib", compress=3)

    runtime = load_python_module("aimers_v4_feature_runtime", package_dir / "script.py")
    tabm_feature_parity = verify_tabm_feature_parity(
        runtime=runtime,
        raw_rows=tabm_audit_raw,
        reference_num=tabm_audit_num,
        reference_cat=tabm_audit_cat,
        metadata=tabm_metadata,
    )
    del tabm_audit_num, tabm_audit_cat

    audit = verify_serialized_ensemble(
        script_path=package_dir / "script.py",
        model_dir=model_dir,
        full_data=data,
        lgbm_audit_raw=lgbm_audit_raw,
        lgbm_reference=lgbm_reference,
        base_audit_raw=base_audit_raw,
        base_reference=base_reference,
        f_route_audit_raw=f_route_audit_raw,
        f_route_reference=f_route_reference,
        tabm_audit_raw=tabm_audit_raw,
        tabm_reference=tabm_reference,
    )
    del (
        lgbm_audit_raw,
        lgbm_reference,
        base_audit_raw,
        base_reference,
        f_route_audit_raw,
        f_route_reference,
        tabm_audit_raw,
        tabm_reference,
    )
    gc.collect()

    end_to_end = BASE.verify_end_to_end(package_dir, args.data_dir)
    zip_path = output_dir / SUBMISSION_FILENAME
    zip_summary = build_and_verify_zip(package_dir, zip_path)
    if len(zip_path.name) > 30:
        raise RuntimeError(
            f"submission filename exceeds 30 characters: {zip_path.name}"
        )

    summary = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "submission_filename": SUBMISSION_FILENAME,
        "selected_pair": V3.SELECTED_PAIR,
        "selected_route": V3.SELECTED_ROUTE,
        "selected_neural_candidate": EXPECTED_TABM_CANDIDATE,
        "train_path": str(args.train_path.resolve()),
        "train_file_bytes": int(args.train_path.stat().st_size),
        "train_file_sha256": BASE.sha256_file(args.train_path),
        "all_train_rows": int(len(data)),
        "all_train_seasons": seasons,
        "reference_season": REFERENCE_SEASON,
        "routing": artifacts["routing"],
        "tree_blend": artifacts["blend"],
        "neural_blend": artifacts["neural_blend"],
        "validation_reference": {
            "source_07b_signature": V3.EXPECTED_ROUTING_SIGNATURE,
            "source_08_signature": EXPECTED_TABM_SIGNATURE,
            "selection_seasons": [2023, 2024],
            "candidate": EXPECTED_TABM_CANDIDATE,
            **tabm_selection["selected_metrics"],
            "selection_checks": tabm_selection["selection_checks"],
            "configuration_checks": tabm_selection["configuration_checks"],
            "integrity_checks": tabm_selection["integrity_checks"],
            "note": "Frozen 08 OOF recipe; no test statistic was used.",
        },
        "lgbm": lgbm_summary,
        "catboost_base": base_summary,
        "catboost_f_route": f_route_summary,
        "tabm": tabm_summary,
        "tree_training_inference_feature_parity": tree_feature_parity,
        "tabm_final_inference_feature_parity": tabm_feature_parity,
        "tabm_preprocessing_audit": tabm_preprocessing_audit,
        "audit": audit,
        "end_to_end": end_to_end,
        "package": zip_summary,
        "model_files": {
            path.name: {
                "bytes": int(path.stat().st_size),
                "sha256": BASE.sha256_file(path),
            }
            for path in sorted(model_dir.iterdir())
            if path.is_file()
        },
        "source_files": {
            "training_script_sha256": BASE.sha256_file(Path(__file__)),
            "feature_source_sha256": BASE.sha256_file(SRC_DIR / "features.py"),
            "runtime_script_sha256": BASE.sha256_file(runtime_template),
            "cv_script_sha256": BASE.sha256_file(
                PROJECT_ROOT / "experiments/08_tabm_cv.py"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
            "lightgbm": __import__("lightgbm").__version__,
            "catboost": __import__("catboost").__version__,
            "torch": torch.__version__,
            "tabm": __import__("tabm").__version__,
            "catboost_task_type": args.catboost_task_type,
            "tabm_device": tabm_summary["device"],
            "threads": args.threads,
        },
        "total_seconds": time.perf_counter() - total_started,
    }
    with (output_dir / "training_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("\nAll checks passed.")
    print(f"Submission ZIP: {zip_path}")
    print(f"SHA-256: {zip_summary['zip_sha256']}")
    print(f"ZIP bytes: {zip_summary['zip_bytes']:,}")
    print(f"Total seconds: {summary['total_seconds']:.1f}")


if __name__ == "__main__":
    main()
