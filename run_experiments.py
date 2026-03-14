#!/usr/bin/env python3
"""
run_experiments.py -- Headless 10x Grind Protocol Runner (v2: Crash-Resilient)
===============================================================================

Consolidates Notebooks 01-03 into a single CLI application for SSH-resilient
execution on Vast.ai (or any headless GPU server).

CHECKPOINTING: Results are saved to disk after *every single run*. If the
process crashes or SSH drops, simply re-launch the same command. The script
will detect completed runs in the JSON files and skip them automatically.
At worst, you lose the one run that was in-flight at crash time.

Usage:
    python run_experiments.py --all              # Full pipeline
    python run_experiments.py --baselines        # NB01: LSTM, BiLSTM, XGBoost
    python run_experiments.py --swarms           # NB02: Pure AO, Pure PSO
    python run_experiments.py --hybrid           # NB03: Hybrid AO-PSO
    python run_experiments.py --baselines --hybrid

All logs are written to both the console AND results/execution_10x_grind.log
so progress survives SSH disconnections. Run Notebook 04 separately for
Wilcoxon tests and publication figures.
"""

import sys
import os
import json
import time
import gc
import logging
import argparse
from pathlib import Path

import numpy as np

# ── Resolve project root from this script's location ──────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


# =====================================================================
# Configuration
# =====================================================================
N_RUNS          = 10
WINDOW_SIZE     = 168
HORIZON         = 24
BATCH_SIZE      = 64
NUM_WORKERS     = 4       # Keep EPYC feeding the GPU

# Baseline DL hyperparameters
DL_HIDDEN_SIZE  = 128
DL_NUM_LAYERS   = 2
DL_DROPOUT      = 0.2
DL_EPOCHS       = 30
DL_LR           = 1e-3

# Swarm / Hybrid optimizer settings
POP_SIZE        = 20
MAX_ITER        = 50      # For standalone AO / PSO
TOTAL_ITER      = 50      # For Hybrid (AO + PSO combined)
AO_FRACTION     = 0.5
PROXY_EPOCHS    = 2
SUBSET_FRACTION = 0.3
VAL_SUBSET_FRAC = 0.1     # Validation subset for proxy evals (speed hack)

# Full TFT training after optimization
FULL_EPOCHS     = 30
PATIENCE        = 5

# Paths
MICRO_PATH  = "data/micro_household_dataset.csv"
MACRO_PATH  = "data/macro_grid_dataset.csv"
RESULTS_DIR = Path("results")

# JSON output files
BASELINE_JSON = RESULTS_DIR / "baseline_metrics.json"
SWARM_JSON    = RESULTS_DIR / "standalone_swarm_metrics.json"
HYBRID_JSON   = RESULTS_DIR / "hybrid_metrics.json"


# =====================================================================
# Logging (dual: console + file)
# =====================================================================
def setup_logging():
    RESULTS_DIR.mkdir(exist_ok=True)
    log_path = RESULTS_DIR / "execution_10x_grind.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return logging.getLogger("grind")


# =====================================================================
# Checkpoint helpers
# =====================================================================
def _load_json(path: Path) -> dict:
    """Load existing results JSON or return empty dict."""
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _is_run_done(data: dict, dataset: str, model_name: str, run: int) -> bool:
    """Check if a specific run already exists in the checkpoint."""
    run_list = data.get(dataset, {}).get(model_name, [])
    if not isinstance(run_list, list):
        return False
    return any(r.get("run") == run for r in run_list)


def _save_run(path: Path, data: dict, dataset: str, model_name: str,
              result: dict, config: dict):
    """Append one run result and flush to disk immediately."""
    data.setdefault(dataset, {})
    data[dataset].setdefault(model_name, [])
    data[dataset][model_name].append(result)
    data["config"] = config
    # Atomic-ish write: write to temp then rename
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def _count_completed(data: dict, dataset: str, model_name: str) -> int:
    """Count how many runs are already completed for a model/dataset."""
    run_list = data.get(dataset, {}).get(model_name, [])
    if not isinstance(run_list, list):
        return 0
    return len(run_list)


# =====================================================================
# Shared helpers
# =====================================================================
def _get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_pipeline(dataset_path, dataset_type, log):
    from src.dataset_utils import prepare_pipeline
    label = "Micro-Grid" if dataset_type == "micro" else "Macro-Grid"
    log.info("Loading %s dataset from %s ...", label, dataset_path)
    pipeline = prepare_pipeline(
        filepath=dataset_path,
        dataset_type=dataset_type,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    log.info("  n_continuous : %d", pipeline["n_continuous"])
    log.info("  target_col   : %s", pipeline["target_col"])
    log.info("  train batches: %d", len(pipeline["train_loader"]))
    log.info("  test  batches: %d", len(pipeline["test_loader"]))
    return pipeline


def _print_summary(log, label, data, dataset, model_name):
    """Print Mean +/- Std from the checkpoint data for a model/dataset."""
    results = data.get(dataset, {}).get(model_name, [])
    if not results:
        return
    rmses = [r["RMSE"] for r in results]
    maes  = [r["MAE"]  for r in results]
    mapes = [r["MAPE"] for r in results]
    log.info("=" * 60)
    log.info("  %s -- %d runs", label, len(results))
    log.info("=" * 60)
    log.info("  RMSE : %.4f +/- %.4f", np.mean(rmses), np.std(rmses))
    log.info("  MAE  : %.4f +/- %.4f", np.mean(maes), np.std(maes))
    log.info("  MAPE : %.2f%% +/- %.2f%%", np.mean(mapes), np.std(mapes))
    if "total_time_s" in results[0]:
        times = [r["total_time_s"] for r in results]
        log.info("  Time : %.1fs +/- %.1fs per run", np.mean(times), np.std(times))
    log.info("=" * 60)


# =====================================================================
# Training / Inference helpers
# =====================================================================
def train_dl_model(model, train_loader, val_loader, epochs, lr, device):
    """Train LSTM / BiLSTM with early stopping. Returns trained model."""
    import torch
    import torch.nn as nn

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x_cont, x_cat, y in train_loader:
            x_cont, y = x_cont.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x_cont)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_cont, x_cat, y in val_loader:
                x_cont, y = x_cont.to(device), y.to(device)
                preds = model(x_cont)
                val_losses.append(criterion(preds, y).item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_tft_full(model, train_loader, val_loader, epochs, lr, device):
    """Train BaseTFT with early stopping. Returns trained model."""
    import torch
    import torch.nn as nn

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x_cont, x_cat, y in train_loader:
            x_cont, y = x_cont.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x_cont)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_cont, x_cat, y in val_loader:
                x_cont, y = x_cont.to(device), y.to(device)
                preds = model(x_cont)
                val_losses.append(criterion(preds, y).item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_model(model, test_loader, device):
    """Run inference, return (preds, targets) as numpy arrays."""
    import torch
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_cont, x_cat, y in test_loader:
            x_cont = x_cont.to(device)
            preds = model(x_cont).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


# =====================================================================
# Phase 1: Baselines (Notebook 01)
# =====================================================================
def run_baselines(log):
    """Execute LSTM, BiLSTM, XGBoost on both datasets with checkpointing."""
    import torch
    from src.models import StandardLSTM, BiLSTM, XGBoostBaseline
    from src.metrics import (
        inverse_transform_predictions,
        calculate_forecasting_metrics,
        calculate_horizon_metrics,
    )

    device = _get_device()
    config = {
        "n_runs": N_RUNS, "window_size": WINDOW_SIZE, "horizon": HORIZON,
        "batch_size": BATCH_SIZE,
        "dl_config": {"hidden_size": DL_HIDDEN_SIZE, "num_layers": DL_NUM_LAYERS,
                       "dropout": DL_DROPOUT, "epochs": DL_EPOCHS, "lr": DL_LR},
    }

    log.info("=" * 70)
    log.info("  PHASE 1: BASELINE EVALUATION (LSTM, BiLSTM, XGBoost)")
    log.info("  Device: %s | N_RUNS: %d", device, N_RUNS)
    log.info("=" * 70)

    models_to_run = [
        ("StandardLSTM",    StandardLSTM,    False),
        ("BiLSTM",          BiLSTM,          False),
        ("XGBoost",         XGBoostBaseline, True),
    ]

    for ds_key, ds_path, ds_type in [
        ("micro", MICRO_PATH, "micro"),
        ("macro", MACRO_PATH, "macro"),
    ]:
        pipeline = _load_pipeline(ds_path, ds_type, log)
        train_loader = pipeline["train_loader"]
        val_loader   = pipeline["val_loader"]
        test_loader  = pipeline["test_loader"]
        scaler       = pipeline["scaler"]
        target_idx   = pipeline["target_idx"]
        n_features   = pipeline["n_continuous"]

        for model_name, model_class, is_xgb in models_to_run:
            log.info("\n--- %s (%s-Grid) ---", model_name, ds_key.capitalize())

            for run in range(1, N_RUNS + 1):
                # ── Checkpoint check ──
                data = _load_json(BASELINE_JSON)
                if _is_run_done(data, ds_key, model_name, run):
                    log.info("  [%s] Skipping Run %d/%d (already completed)",
                             model_name, run, N_RUNS)
                    continue

                t0 = time.time()

                if is_xgb:
                    model = model_class(horizon=HORIZON)
                    X_train_list, y_train_list = [], []
                    for x_cont, x_cat, y in train_loader:
                        X_train_list.append(x_cont.numpy())
                        y_train_list.append(y.numpy())
                    X_test_list, y_test_list = [], []
                    for x_cont, x_cat, y in test_loader:
                        X_test_list.append(x_cont.numpy())
                        y_test_list.append(y.numpy())
                    model.fit(np.concatenate(X_train_list),
                              np.concatenate(y_train_list))
                    preds_scaled   = model.predict(np.concatenate(X_test_list))
                    targets_scaled = np.concatenate(y_test_list)
                else:
                    model = model_class(
                        n_features=n_features,
                        hidden_size=DL_HIDDEN_SIZE,
                        num_layers=DL_NUM_LAYERS,
                        dropout=DL_DROPOUT,
                        horizon=HORIZON,
                    )
                    model = train_dl_model(
                        model, train_loader, val_loader,
                        epochs=DL_EPOCHS, lr=DL_LR, device=device,
                    )
                    preds_scaled, targets_scaled = predict_model(
                        model, test_loader, device,
                    )

                preds_real, targets_real = inverse_transform_predictions(
                    preds_scaled, targets_scaled, scaler, target_idx,
                )
                overall     = calculate_forecasting_metrics(targets_real, preds_real)
                per_horizon = calculate_horizon_metrics(targets_real, preds_real)
                elapsed = time.time() - t0

                result = {
                    "run":  run,
                    "RMSE": overall["RMSE"],
                    "MAE":  overall["MAE"],
                    "MAPE": overall["MAPE"],
                    "horizon_metrics": {str(h): m for h, m in per_horizon.items()},
                    "time_s": round(elapsed, 2),
                }

                # ── Save immediately ──
                data = _load_json(BASELINE_JSON)
                _save_run(BASELINE_JSON, data, ds_key, model_name, result, config)

                log.info(
                    "  [%s] Run %2d/%d | RMSE: %.4f | MAE: %.4f | "
                    "MAPE: %.2f%% | %.1fs | SAVED",
                    model_name, run, N_RUNS,
                    overall["RMSE"], overall["MAE"], overall["MAPE"], elapsed,
                )

                del model
                if not is_xgb:
                    torch.cuda.empty_cache()
                gc.collect()

            # Print summary for this model/dataset combo
            data = _load_json(BASELINE_JSON)
            _print_summary(log, f"{model_name} ({ds_key.capitalize()})",
                           data, ds_key, model_name)

        del pipeline
        torch.cuda.empty_cache()
        gc.collect()

    log.info("\n>>> BASELINE TESTING COMPLETE <<<")
    log.info("  Results: %s", BASELINE_JSON)


# =====================================================================
# Phase 2: Standalone Swarms (Notebook 02)
# =====================================================================
def run_swarms(log):
    """Execute Pure AO and Pure PSO on both datasets with checkpointing."""
    import torch
    from src.models import BaseTFT
    from src.optimizers import ObjectiveFunction, AquilaOptimizer, ParticleSwarm
    from src.metrics import (
        inverse_transform_predictions,
        calculate_forecasting_metrics,
        calculate_horizon_metrics,
    )

    device = _get_device()
    config = {
        "n_runs": N_RUNS, "pop_size": POP_SIZE, "max_iter": MAX_ITER,
        "proxy_epochs": PROXY_EPOCHS, "subset_fraction": SUBSET_FRACTION,
        "val_subset_fraction": VAL_SUBSET_FRAC,
        "full_epochs": FULL_EPOCHS, "patience": PATIENCE,
        "window_size": WINDOW_SIZE, "horizon": HORIZON, "batch_size": BATCH_SIZE,
    }

    log.info("=" * 70)
    log.info("  PHASE 2: STANDALONE SWARM EVALUATION (Pure AO, Pure PSO)")
    log.info("  Device: %s | N_RUNS: %d | POP: %d | ITER: %d",
             device, N_RUNS, POP_SIZE, MAX_ITER)
    log.info("  [System] Using val_subset_fraction=%.1f and num_workers=%d "
             "for proxy evaluations.", VAL_SUBSET_FRAC, NUM_WORKERS)
    log.info("=" * 70)

    optimizers_to_run = [
        ("AquilaOptimizer", AquilaOptimizer),
        ("ParticleSwarm",   ParticleSwarm),
    ]

    for ds_key, ds_path, ds_type in [
        ("micro", MICRO_PATH, "micro"),
        ("macro", MACRO_PATH, "macro"),
    ]:
        pipeline = _load_pipeline(ds_path, ds_type, log)
        train_loader = pipeline["train_loader"]
        val_loader   = pipeline["val_loader"]
        test_loader  = pipeline["test_loader"]
        scaler       = pipeline["scaler"]
        target_idx   = pipeline["target_idx"]
        n_features   = pipeline["n_continuous"]

        for opt_name, opt_class in optimizers_to_run:
            log.info("\n### %s (%s-Grid) -- 10x Grind ###",
                     opt_name, ds_key.capitalize())

            for run in range(1, N_RUNS + 1):
                # ── Checkpoint check ──
                data = _load_json(SWARM_JSON)
                if _is_run_done(data, ds_key, opt_name, run):
                    log.info("  [%s] Skipping Run %d/%d (already completed)",
                             opt_name, run, N_RUNS)
                    continue

                log.info("\n  [%s] Run %d/%d", opt_name, run, N_RUNS)
                t0 = time.time()

                # -- Optimization --
                obj_fn = ObjectiveFunction(
                    train_loader=train_loader, val_loader=val_loader,
                    n_encoder_features=n_features, window_size=WINDOW_SIZE,
                    horizon=HORIZON, proxy_epochs=PROXY_EPOCHS,
                    subset_fraction=SUBSET_FRACTION,
                    val_subset_fraction=VAL_SUBSET_FRAC, device=device,
                )
                swarm = opt_class(
                    objective_fn=obj_fn, pop_size=POP_SIZE,
                    max_iter=MAX_ITER, seed=run,
                )
                best_params, best_fitness, convergence = swarm.optimize()
                opt_time = time.time() - t0

                log.info("  Optimization: %.1fs | Proxy RMSE: %.6f",
                         opt_time, best_fitness)
                for k, v in best_params.items():
                    log.info("    %s: %s", k, v)

                # -- Full training --
                t1 = time.time()
                model = BaseTFT(
                    n_encoder_features=n_features, n_decoder_features=0,
                    n_static_categoricals=0,
                    d_model=best_params["d_model"],
                    n_heads=best_params["n_heads"],
                    num_encoder_layers=best_params["num_encoder_layers"],
                    dropout=best_params["dropout"],
                    horizon=HORIZON, window_size=WINDOW_SIZE,
                )
                model = train_tft_full(
                    model, train_loader, val_loader,
                    epochs=FULL_EPOCHS, lr=best_params["learning_rate"],
                    device=device,
                )
                train_time = time.time() - t1

                # -- Evaluation --
                preds_scaled, targets_scaled = predict_model(
                    model, test_loader, device,
                )
                preds_real, targets_real = inverse_transform_predictions(
                    preds_scaled, targets_scaled, scaler, target_idx,
                )
                overall     = calculate_forecasting_metrics(targets_real, preds_real)
                per_horizon = calculate_horizon_metrics(targets_real, preds_real)
                total_time  = time.time() - t0

                result = {
                    "run":            run,
                    "RMSE":           overall["RMSE"],
                    "MAE":            overall["MAE"],
                    "MAPE":           overall["MAPE"],
                    "horizon_metrics": {str(h): m for h, m in per_horizon.items()},
                    "best_params":    best_params,
                    "proxy_fitness":  best_fitness,
                    "convergence":    convergence.to_dict(),
                    "opt_time_s":     round(opt_time, 2),
                    "train_time_s":   round(train_time, 2),
                    "total_time_s":   round(total_time, 2),
                }

                # ── Save immediately ──
                data = _load_json(SWARM_JSON)
                _save_run(SWARM_JSON, data, ds_key, opt_name, result, config)

                log.info(
                    "  [%s] Run %2d COMPLETE | RMSE: %.4f | MAE: %.4f | "
                    "MAPE: %.2f%% | %.1fs | SAVED",
                    opt_name, run,
                    overall["RMSE"], overall["MAE"], overall["MAPE"], total_time,
                )

                del model, swarm, obj_fn, convergence
                torch.cuda.empty_cache()
                gc.collect()

            data = _load_json(SWARM_JSON)
            _print_summary(log, f"{opt_name} ({ds_key.capitalize()})",
                           data, ds_key, opt_name)

        del pipeline
        torch.cuda.empty_cache()
        gc.collect()

    log.info("\n>>> STANDALONE SWARM TESTING COMPLETE <<<")
    log.info("  Results: %s", SWARM_JSON)


# =====================================================================
# Phase 3: Hybrid AO-PSO Champion (Notebook 03)
# =====================================================================
def run_hybrid(log):
    """Execute Hybrid AO-PSO on both datasets with checkpointing."""
    import torch
    from src.models import BaseTFT
    from src.optimizers import ObjectiveFunction, Hybrid_AO_PSO
    from src.metrics import (
        inverse_transform_predictions,
        calculate_forecasting_metrics,
        calculate_horizon_metrics,
    )

    device = _get_device()
    config = {
        "n_runs": N_RUNS, "pop_size": POP_SIZE, "total_iter": TOTAL_ITER,
        "ao_fraction": AO_FRACTION, "proxy_epochs": PROXY_EPOCHS,
        "subset_fraction": SUBSET_FRACTION, "val_subset_fraction": VAL_SUBSET_FRAC,
        "full_epochs": FULL_EPOCHS, "patience": PATIENCE,
        "window_size": WINDOW_SIZE, "horizon": HORIZON, "batch_size": BATCH_SIZE,
    }

    log.info("=" * 70)
    log.info("  PHASE 3: CHAMPION EVALUATION (Hybrid AO-PSO)")
    log.info("  Device: %s | N_RUNS: %d | POP: %d | TOTAL_ITER: %d | AO_FRAC: %.1f",
             device, N_RUNS, POP_SIZE, TOTAL_ITER, AO_FRACTION)
    log.info("  [System] Using val_subset_fraction=%.1f and num_workers=%d "
             "for proxy evaluations.", VAL_SUBSET_FRAC, NUM_WORKERS)
    log.info("=" * 70)

    model_name = "Hybrid_AO_PSO"

    for ds_key, ds_path, ds_type in [
        ("micro", MICRO_PATH, "micro"),
        ("macro", MACRO_PATH, "macro"),
    ]:
        pipeline = _load_pipeline(ds_path, ds_type, log)
        train_loader = pipeline["train_loader"]
        val_loader   = pipeline["val_loader"]
        test_loader  = pipeline["test_loader"]
        scaler       = pipeline["scaler"]
        target_idx   = pipeline["target_idx"]
        n_features   = pipeline["n_continuous"]

        log.info("\n### Hybrid AO-PSO (%s-Grid) -- 10x Grind ###",
                 ds_key.capitalize())

        for run in range(1, N_RUNS + 1):
            # ── Checkpoint check ──
            data = _load_json(HYBRID_JSON)
            if _is_run_done(data, ds_key, model_name, run):
                log.info("  [Hybrid] Skipping Run %d/%d (already completed)",
                         run, N_RUNS)
                continue

            log.info("\n  [Hybrid AO-PSO] Run %d/%d", run, N_RUNS)
            t0 = time.time()

            # -- Optimization --
            obj_fn = ObjectiveFunction(
                train_loader=train_loader, val_loader=val_loader,
                n_encoder_features=n_features, window_size=WINDOW_SIZE,
                horizon=HORIZON, proxy_epochs=PROXY_EPOCHS,
                subset_fraction=SUBSET_FRACTION,
                val_subset_fraction=VAL_SUBSET_FRAC, device=device,
            )
            hybrid = Hybrid_AO_PSO(
                objective_fn=obj_fn, pop_size=POP_SIZE,
                total_iter=TOTAL_ITER, ao_fraction=AO_FRACTION, seed=run,
            )
            best_params, best_fitness, convergence = hybrid.optimize()
            opt_time = time.time() - t0

            ao_iters = sum(1 for r in convergence.history
                           if r.get("phase") == "AO")
            pso_iters = sum(1 for r in convergence.history
                            if r.get("phase") == "PSO")
            log.info("  Optimization: %.1fs | AO iters: %d | PSO iters: %d "
                     "| Proxy RMSE: %.6f",
                     opt_time, ao_iters, pso_iters, best_fitness)
            for k, v in best_params.items():
                log.info("    %s: %s", k, v)

            # -- Full training --
            t1 = time.time()
            model = BaseTFT(
                n_encoder_features=n_features, n_decoder_features=0,
                n_static_categoricals=0,
                d_model=best_params["d_model"],
                n_heads=best_params["n_heads"],
                num_encoder_layers=best_params["num_encoder_layers"],
                dropout=best_params["dropout"],
                horizon=HORIZON, window_size=WINDOW_SIZE,
            )
            model = train_tft_full(
                model, train_loader, val_loader,
                epochs=FULL_EPOCHS, lr=best_params["learning_rate"],
                device=device,
            )
            train_time = time.time() - t1

            # -- Evaluation --
            preds_scaled, targets_scaled = predict_model(
                model, test_loader, device,
            )
            preds_real, targets_real = inverse_transform_predictions(
                preds_scaled, targets_scaled, scaler, target_idx,
            )
            overall     = calculate_forecasting_metrics(targets_real, preds_real)
            per_horizon = calculate_horizon_metrics(targets_real, preds_real)
            total_time  = time.time() - t0

            result = {
                "run":            run,
                "RMSE":           overall["RMSE"],
                "MAE":            overall["MAE"],
                "MAPE":           overall["MAPE"],
                "horizon_metrics": {str(h): m for h, m in per_horizon.items()},
                "best_params":    best_params,
                "proxy_fitness":  best_fitness,
                "convergence":    convergence.to_dict(),
                "opt_time_s":     round(opt_time, 2),
                "train_time_s":   round(train_time, 2),
                "total_time_s":   round(total_time, 2),
            }

            # ── Save immediately ──
            data = _load_json(HYBRID_JSON)
            _save_run(HYBRID_JSON, data, ds_key, model_name, result, config)

            log.info(
                "  [Hybrid] Run %2d COMPLETE | RMSE: %.4f | MAE: %.4f | "
                "MAPE: %.2f%% | %.1fs | SAVED",
                run, overall["RMSE"], overall["MAE"], overall["MAPE"], total_time,
            )

            del model, hybrid, obj_fn, convergence
            torch.cuda.empty_cache()
            gc.collect()

        data = _load_json(HYBRID_JSON)
        _print_summary(log, f"Hybrid AO-PSO-TFT ({ds_key.capitalize()})",
                       data, ds_key, model_name)

        del pipeline
        torch.cuda.empty_cache()
        gc.collect()

    log.info("\n>>> CHAMPION HYBRID TESTING COMPLETE <<<")
    log.info("  Results: %s", HYBRID_JSON)


# =====================================================================
# CLI Entry Point
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="10x Grind Protocol -- Crash-resilient headless experiment "
                    "runner for Hybrid AO-PSO-TFT energy forecasting research.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python run_experiments.py --all              Full pipeline (hours on GPU)
  python run_experiments.py --baselines        Only LSTM/BiLSTM/XGBoost
  python run_experiments.py --swarms --hybrid  Optimizers only
  python run_experiments.py --hybrid           Resume hybrid (skips done runs)

Checkpointing: Results are saved after EVERY run. Re-launch the same
command after a crash -- completed runs are detected and skipped.

Logs:   results/execution_10x_grind.log
Plots:  Run Notebook 04 separately after all phases complete.
""",
    )
    parser.add_argument("--baselines", action="store_true",
                        help="Run LSTM, BiLSTM, XGBoost baselines (Notebook 01)")
    parser.add_argument("--swarms", action="store_true",
                        help="Run Pure AO and Pure PSO standalone (Notebook 02)")
    parser.add_argument("--hybrid", action="store_true",
                        help="Run Hybrid AO-PSO champion (Notebook 03)")
    parser.add_argument("--all", action="store_true",
                        help="Run the complete pipeline: baselines -> swarms -> hybrid")

    args = parser.parse_args()

    if not any([args.baselines, args.swarms, args.hybrid, args.all]):
        parser.print_help()
        sys.exit(0)

    log = setup_logging()
    RESULTS_DIR.mkdir(exist_ok=True)

    log.info("=" * 70)
    log.info("  10x GRIND PROTOCOL -- Crash-Resilient Experiment Runner v2")
    log.info("  Project root: %s", PROJECT_ROOT)
    log.info("=" * 70)

    # Report existing checkpoint state
    for label, path in [("Baseline", BASELINE_JSON),
                        ("Swarm", SWARM_JSON),
                        ("Hybrid", HYBRID_JSON)]:
        if path.exists():
            data = _load_json(path)
            counts = {}
            for ds in ["micro", "macro"]:
                for model_name in data.get(ds, {}):
                    n = _count_completed(data, ds, model_name)
                    if n > 0:
                        counts[f"{ds}/{model_name}"] = n
            if counts:
                log.info("  Checkpoint [%s]: %s", label, counts)

    wall_start = time.time()

    if args.all or args.baselines:
        run_baselines(log)

    if args.all or args.swarms:
        run_swarms(log)

    if args.all or args.hybrid:
        run_hybrid(log)

    wall_total = time.time() - wall_start
    hours = int(wall_total // 3600)
    minutes = int((wall_total % 3600) // 60)
    seconds = int(wall_total % 60)
    log.info("\n" + "=" * 70)
    log.info("  TOTAL WALL TIME: %dh %dm %ds", hours, minutes, seconds)
    log.info("  Logs: %s", RESULTS_DIR / "execution_10x_grind.log")
    log.info("  Run Notebook 04 for Wilcoxon tests and publication figures.")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
