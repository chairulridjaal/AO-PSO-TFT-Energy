"""
dataset_utils.py -- Data Pipeline for Multi-Horizon Energy Forecasting
======================================================================

Research: Hybrid AO-PSO Optimised Temporal Fusion Transformer
           for Smart Grid Multi-Horizon Energy Forecasting

This module provides the complete data pipeline from raw CSV to
GPU-ready PyTorch DataLoaders.  It handles two heterogeneous datasets:

    Micro-Grid (UCI)   -- Single household in Sceaux, France (~34k hourly)
    Macro-Grid (ISO-NE) -- Regional aggregate, New England  (~35k hourly)

Pipeline stages:
    1. load_and_preprocess_data  -- CSV ingest, timestamp normalisation,
                                    cyclical feature injection, static
                                    categorical column for TFT
    2. split_and_scale_data      -- Chronological train/val/test split
                                    with scaler fitted ONLY on train
                                    (prevents look-ahead data leakage)
    3. TimeSeriesDataset         -- Sliding-window PyTorch Dataset that
                                    cleanly separates continuous features,
                                    categorical features, and target
    4. create_dataloaders         -- Wraps datasets into DataLoaders
                                    with configurable batch size

Anti-leakage protocol:
    - Chronological splitting ONLY (no shuffling across time)
    - Scaler fitted exclusively on training data
    - Train DataLoader shuffles within its window-safe range;
      val/test DataLoaders iterate in strict temporal order

References
----------
[1] Lim, B. et al. (2021). "Temporal Fusion Transformers for
    Interpretable Multi-horizon Time Series Forecasting."
    International Journal of Forecasting, 37(4), pp. 1748--1764.
[2] Hebrail, G. & Berard, A. (2012). UCI Individual Household
    Electric Power Consumption Dataset.
[3] ISO New England. System Load and Weather Data (2021--2024).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler, LabelEncoder

__all__ = [
    "TimeSeriesDataset",
    "load_and_preprocess_data",
    "split_and_scale_data",
    "create_dataloaders",
]


# =====================================================================
# SECTION 1: DATASET CONFIGURATION REGISTRY
# =====================================================================
# Centralised metadata for each dataset type so that downstream code
# never hard-codes column names.  Adding a new dataset is as simple as
# appending an entry here.

DATASET_CONFIGS: Dict[str, dict] = {
    "micro": {
        # Target variable to forecast
        "target_col": "Global_active_power",
        # Static categorical identifier injected for TFT grouping
        "grid_id": "UCI",
        # Columns that exist in the raw CSV but should be dropped
        # before modelling (string/redundant columns)
        "drop_cols": [],
        # Whether the raw CSV already contains cyclical time features
        "has_cyclical_features": False,
    },
    "macro": {
        "target_col": "System_Load",
        "grid_id": "ISO_NE",
        "drop_cols": ["temp_check"],
        "has_cyclical_features": True,
    },
}


# =====================================================================
# SECTION 2: DATA LOADING & PREPROCESSING
# =====================================================================

def _inject_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sinusoidal time covariates if the dataframe lacks them.

    Cyclical encoding prevents the model from perceiving artificial
    discontinuities at period boundaries (e.g. hour 23 -> 0).

    Injected features:
        hour_sin, hour_cos   -- 24-hour daily cycle
        dow_sin,  dow_cos    -- 7-day weekly cycle
        month_sin, month_cos -- 12-month annual cycle

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with six new columns appended.
    """
    hour = df.index.hour
    dow = df.index.dayofweek
    month = df.index.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    return df


def load_and_preprocess_data(
    filepath: str,
    dataset_type: str,
) -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    """
    Load a CSV dataset and prepare it for the forecasting pipeline.

    Steps performed:
        1. Read CSV and parse the ``Datetime`` column as the index.
        2. Drop any redundant string columns (e.g. ``temp_check``).
        3. Inject cyclical time features if the dataset lacks them.
        4. Inject a static categorical ``grid_id`` column required by
           the TFT Variable Selection Network for group-aware gating.
        5. Classify every column as continuous or categorical.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the pre-processed CSV file
        (``micro_household_dataset.csv`` or ``macro_grid_dataset.csv``).
    dataset_type : str
        One of ``"micro"`` or ``"macro"``.  Controls which config
        entry from ``DATASET_CONFIGS`` is used.

    Returns
    -------
    df : pd.DataFrame
        Cleaned DataFrame with DatetimeIndex, ready for scaling.
    target_col : str
        Name of the target column.
    continuous_cols : list[str]
        Names of all continuous (numeric) feature columns.
    categorical_cols : list[str]
        Names of all categorical feature columns (currently just
        ``grid_id``).

    Raises
    ------
    ValueError
        If ``dataset_type`` is not a recognised key.
    FileNotFoundError
        If ``filepath`` does not exist.
    """
    # ── Validate dataset type ──────────────────────────────────────
    dataset_type = dataset_type.strip().lower()
    if dataset_type not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset_type '{dataset_type}'.  "
            f"Expected one of: {list(DATASET_CONFIGS.keys())}"
        )
    config = DATASET_CONFIGS[dataset_type]
    target_col: str = config["target_col"]

    # ── 1. Read CSV ────────────────────────────────────────────────
    df = pd.read_csv(filepath, parse_dates=["Datetime"], index_col="Datetime")
    df.sort_index(inplace=True)

    print(f"[dataset_utils] Loaded {dataset_type} dataset: {df.shape}")
    print(f"  Date range: {df.index.min()} --> {df.index.max()}")

    # ── 2. Drop redundant columns ─────────────────────────────────
    cols_to_drop = [c for c in config["drop_cols"] if c in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"  Dropped columns: {cols_to_drop}")

    # ── 3. Handle missing values ──────────────────────────────────
    n_missing = df.isna().sum().sum()
    if n_missing > 0:
        df.interpolate(method="linear", inplace=True)
        df.dropna(inplace=True)
        print(f"  Interpolated & dropped {n_missing} missing values")

    # ── 4. Inject cyclical time features (micro dataset only) ─────
    if not config["has_cyclical_features"]:
        df = _inject_cyclical_features(df)
        print("  Injected cyclical time features (hour/dow/month sin/cos)")

    # ── 5. Inject static categorical column for TFT grouping ─────
    # The TFT's Variable Selection Network and static context GRNs
    # require at least one static categorical input.  Since each
    # dataset represents a single grid, we inject a constant group
    # identifier that gets label-encoded to an integer below.
    df["grid_id"] = config["grid_id"]
    print(f"  Injected grid_id = '{config['grid_id']}'")

    # ── 6. Classify columns ───────────────────────────────────────
    categorical_cols = ["grid_id"]
    continuous_cols = [
        c for c in df.columns
        if c not in categorical_cols
    ]

    # Ensure target is among continuous columns
    assert target_col in continuous_cols, (
        f"Target '{target_col}' not found in continuous columns: "
        f"{continuous_cols}"
    )

    print(f"  Target:       {target_col}")
    print(f"  Continuous:   {len(continuous_cols)} features")
    print(f"  Categorical:  {categorical_cols}")

    return df, target_col, continuous_cols, categorical_cols


# =====================================================================
# SECTION 3: CHRONOLOGICAL SPLITTING & SCALING
# =====================================================================

def split_and_scale_data(
    df: pd.DataFrame,
    target_col: str,
    continuous_cols: List[str],
    categorical_cols: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    StandardScaler,
    LabelEncoder,
    int,
]:
    """
    Chronologically split into train/val/test and scale.

    Anti-leakage protocol:
        - The scaler is fitted EXCLUSIVELY on the training partition.
        - Validation and test sets are transformed (never fitted) with
          the same scaler, preventing any future information from
          leaking into the scaling statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``load_and_preprocess_data``.
    target_col : str
        Name of the target column.
    continuous_cols : list[str]
        Continuous feature column names.
    categorical_cols : list[str]
        Categorical feature column names.
    train_ratio : float
        Fraction of data for training (default 0.70).
    val_ratio : float
        Fraction of data for validation (default 0.15).
        Test ratio is implicitly ``1 - train_ratio - val_ratio``.

    Returns
    -------
    df_train : pd.DataFrame
        Scaled training partition.
    df_val : pd.DataFrame
        Scaled validation partition.
    df_test : pd.DataFrame
        Scaled test partition.
    scaler : StandardScaler
        Fitted scaler for inverse-transforming predictions back to
        original units (required for real-world RMSE/MAE/MAPE).
    label_encoder : LabelEncoder
        Fitted encoder for categorical columns.
    target_idx : int
        Integer index of the target column within ``continuous_cols``
        (needed by ``TimeSeriesDataset`` to extract the forecast
        target from the sliding window).
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, (
        f"test_ratio must be > 0, got {test_ratio:.4f} "
        f"(train={train_ratio}, val={val_ratio})"
    )

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    print(f"\n[dataset_utils] Chronological split (no shuffling):")
    print(f"  Train: {len(df_train):>7,} rows  "
          f"({df_train.index.min()} --> {df_train.index.max()})")
    print(f"  Val:   {len(df_val):>7,} rows  "
          f"({df_val.index.min()} --> {df_val.index.max()})")
    print(f"  Test:  {len(df_test):>7,} rows  "
          f"({df_test.index.min()} --> {df_test.index.max()})")

    # ── Label-encode categorical columns ──────────────────────────
    label_encoder = LabelEncoder()
    # Fit on the full vocabulary (all unique categories across the
    # entire dataset) so val/test never encounter unseen labels.
    label_encoder.fit(df[categorical_cols[0]])

    for split_df in [df_train, df_val, df_test]:
        for col in categorical_cols:
            split_df[col] = label_encoder.transform(split_df[col])

    # ── Fit scaler ONLY on training data (anti-leakage) ───────────
    scaler = StandardScaler()
    scaler.fit(df_train[continuous_cols].values)

    # Transform all splits with the training-fitted scaler
    df_train[continuous_cols] = scaler.transform(
        df_train[continuous_cols].values
    )
    df_val[continuous_cols] = scaler.transform(
        df_val[continuous_cols].values
    )
    df_test[continuous_cols] = scaler.transform(
        df_test[continuous_cols].values
    )

    # ── Locate the target column index within continuous features ─
    target_idx = continuous_cols.index(target_col)
    print(f"  Scaler fitted on training set ({len(continuous_cols)} "
          f"continuous features)")
    print(f"  Target column index: {target_idx} ('{target_col}')")

    return df_train, df_val, df_test, scaler, label_encoder, target_idx


# =====================================================================
# SECTION 4: PYTORCH SLIDING-WINDOW DATASET
# =====================================================================

class TimeSeriesDataset(Dataset):
    """
    Sliding-window PyTorch Dataset for multi-horizon forecasting.

    Each sample ``(x_cont, x_cat, y)`` is constructed by sliding a
    fixed-length window over the chronologically ordered data:

    ::

        |<--- window_size --->|<--- horizon --->|
        [...encoder features..][ target values  ]
              x_cont, x_cat          y

    The dataset cleanly separates:
        - **x_cont** : Continuous features for the encoder window.
                        Shape ``(window_size, n_continuous)``.
        - **x_cat**  : Categorical features (static, repeated per step).
                        Shape ``(window_size,)`` of int64 indices.
        - **y**       : Target values over the forecast horizon.
                        Shape ``(horizon,)``.

    Parameters
    ----------
    data : pd.DataFrame
        Scaled dataframe with continuous and categorical columns.
    continuous_cols : list[str]
        Names of continuous feature columns.
    categorical_cols : list[str]
        Names of categorical feature columns.
    target_idx : int
        Index of the target variable within ``continuous_cols``.
    window_size : int
        Number of past time steps in each encoder input window.
    horizon : int
        Number of future time steps to predict.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        continuous_cols: List[str],
        categorical_cols: List[str],
        target_idx: int,
        window_size: int = 168,
        horizon: int = 24,
    ):
        super().__init__()
        self.window_size = window_size
        self.horizon = horizon
        self.target_idx = target_idx

        # Pre-convert to numpy for fast __getitem__ indexing.
        # Continuous features: float32 for GPU efficiency.
        self.cont_data = data[continuous_cols].values.astype(np.float32)
        # Categorical features: int64 for nn.Embedding lookups.
        self.cat_data = data[categorical_cols].values.astype(np.int64)

        # Total number of valid sliding-window positions.
        # The last valid start index is len(data) - window_size - horizon.
        self.n_samples = len(data) - window_size - horizon + 1

        if self.n_samples <= 0:
            raise ValueError(
                f"Not enough data for window_size={window_size} + "
                f"horizon={horizon}.  Got {len(data)} rows, need at "
                f"least {window_size + horizon}."
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sliding-window sample.

        Parameters
        ----------
        idx : int
            Sample index (0-based window start position).

        Returns
        -------
        x_cont : torch.FloatTensor, shape (window_size, n_continuous)
            Continuous encoder features over the lookback window.
        x_cat : torch.LongTensor, shape (window_size,)
            Categorical encoder features (static group id repeated
            at every time step for TFT compatibility).
        y : torch.FloatTensor, shape (horizon,)
            Ground-truth target values over the prediction horizon.
        """
        # ── Encoder window: [idx, idx + window_size) ──────────────
        enc_start = idx
        enc_end = idx + self.window_size

        x_cont = torch.from_numpy(self.cont_data[enc_start:enc_end])
        # Flatten categorical to 1D (squeeze the last dim since we
        # currently have exactly one categorical column: grid_id).
        x_cat = torch.from_numpy(self.cat_data[enc_start:enc_end, 0])

        # ── Target horizon: [idx + window_size, idx + window_size + horizon)
        tgt_start = enc_end
        tgt_end = enc_end + self.horizon

        y = torch.from_numpy(
            self.cont_data[tgt_start:tgt_end, self.target_idx]
        )

        return x_cont, x_cat, y


# =====================================================================
# SECTION 5: DATALOADER FACTORY
# =====================================================================

def create_dataloaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    continuous_cols: List[str],
    categorical_cols: List[str],
    target_idx: int,
    window_size: int = 168,
    horizon: int = 24,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test splits.

    Training data is shuffled within its window range to reduce
    sequential correlation between consecutive mini-batches, which
    helps SGD generalisation.  Validation and test loaders iterate
    in strict chronological order for reproducible evaluation.

    Parameters
    ----------
    df_train, df_val, df_test : pd.DataFrame
        Scaled partitions from ``split_and_scale_data``.
    continuous_cols : list[str]
        Continuous feature column names.
    categorical_cols : list[str]
        Categorical feature column names.
    target_idx : int
        Target column index within ``continuous_cols``.
    window_size : int
        Lookback window length in hours (default 168 = 7 days).
    horizon : int
        Forecast horizon in hours (default 24 = 1 day).
    batch_size : int
        Mini-batch size (default 64).
    num_workers : int
        DataLoader worker processes (default 0 for Windows compat).
    pin_memory : bool
        Pin memory for faster CPU-to-GPU transfer (default True).

    Returns
    -------
    train_loader : DataLoader
        Shuffled training DataLoader.
    val_loader : DataLoader
        Sequential validation DataLoader.
    test_loader : DataLoader
        Sequential test DataLoader.
    """
    # ── Instantiate datasets ──────────────────────────────────────
    ds_kwargs = dict(
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        target_idx=target_idx,
        window_size=window_size,
        horizon=horizon,
    )

    train_ds = TimeSeriesDataset(data=df_train, **ds_kwargs)
    val_ds = TimeSeriesDataset(data=df_val, **ds_kwargs)
    test_ds = TimeSeriesDataset(data=df_test, **ds_kwargs)

    print(f"\n[dataset_utils] DataLoader summary "
          f"(window={window_size}, horizon={horizon}, "
          f"batch={batch_size}):")
    print(f"  Train samples: {len(train_ds):>7,}")
    print(f"  Val samples:   {len(val_ds):>7,}")
    print(f"  Test samples:  {len(test_ds):>7,}")

    # ── Common DataLoader kwargs ──────────────────────────────────
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # Training: shuffle=True within the window-safe index range.
    # This shuffles which *windows* appear in each mini-batch, but
    # each individual window's internal time order is preserved.
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)

    # Validation & Test: shuffle=False for deterministic evaluation.
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


# =====================================================================
# SECTION 6: CONVENIENCE END-TO-END PIPELINE
# =====================================================================

def prepare_pipeline(
    filepath: str,
    dataset_type: str,
    window_size: int = 168,
    horizon: int = 24,
    batch_size: int = 64,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    num_workers: int = 0,
) -> dict:
    """
    End-to-end convenience function: CSV -> DataLoaders.

    Chains ``load_and_preprocess_data`` -> ``split_and_scale_data`` ->
    ``create_dataloaders`` and returns all artefacts needed for
    training, evaluation, and inverse-scaling in a single dict.

    Parameters
    ----------
    filepath : str
        Path to the dataset CSV.
    dataset_type : str
        ``"micro"`` or ``"macro"``.
    window_size : int
        Encoder lookback window length.
    horizon : int
        Forecast horizon length.
    batch_size : int
        Mini-batch size for DataLoaders.
    train_ratio : float
        Training set fraction.
    val_ratio : float
        Validation set fraction.
    num_workers : int
        DataLoader worker processes.

    Returns
    -------
    dict with keys:
        ``train_loader``, ``val_loader``, ``test_loader`` -- DataLoaders
        ``scaler``         -- fitted StandardScaler (for inverse transform)
        ``label_encoder``  -- fitted LabelEncoder
        ``target_col``     -- target column name (str)
        ``target_idx``     -- target column index in continuous features
        ``continuous_cols`` -- list of continuous feature names
        ``categorical_cols`` -- list of categorical feature names
        ``n_continuous``   -- int, number of continuous features
        ``n_categorical``  -- int, number of categorical features
        ``window_size``    -- int
        ``horizon``        -- int
    """
    print("=" * 60)
    print(f"  prepare_pipeline: {dataset_type.upper()} dataset")
    print(f"  window_size={window_size}, horizon={horizon}, "
          f"batch_size={batch_size}")
    print("=" * 60)

    # Stage 1: Load & preprocess
    df, target_col, continuous_cols, categorical_cols = (
        load_and_preprocess_data(filepath, dataset_type)
    )

    # Stage 2: Split & scale
    df_train, df_val, df_test, scaler, label_encoder, target_idx = (
        split_and_scale_data(
            df, target_col, continuous_cols, categorical_cols,
            train_ratio=train_ratio, val_ratio=val_ratio,
        )
    )

    # Stage 3: Build DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        df_train, df_val, df_test,
        continuous_cols, categorical_cols, target_idx,
        window_size=window_size, horizon=horizon,
        batch_size=batch_size, num_workers=num_workers,
    )

    print("=" * 60)
    print("  Pipeline ready.")
    print("=" * 60 + "\n")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "target_col": target_col,
        "target_idx": target_idx,
        "continuous_cols": continuous_cols,
        "categorical_cols": categorical_cols,
        "n_continuous": len(continuous_cols),
        "n_categorical": len(categorical_cols),
        "window_size": window_size,
        "horizon": horizon,
    }
