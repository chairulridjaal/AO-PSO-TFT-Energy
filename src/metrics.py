"""
metrics.py -- Evaluation & Statistical Testing for Energy Load Forecasting
==========================================================================

Research: Hybrid AO-PSO Optimised Temporal Fusion Transformer
           for Smart Grid Multi-Horizon Energy Forecasting

This module provides all mathematical evaluation required by the testing
matrix defined in MASTER_BLUEPRINT.md:

    1. inverse_transform_predictions  -- Undo StandardScaler to recover
                                         real-world units (MW, kW, etc.)
    2. calculate_forecasting_metrics  -- Point-forecast error metrics
                                         (RMSE, MAE, MAPE)
    3. calculate_horizon_metrics      -- Per-horizon error breakdown
                                         (1h, 6h, 12h, 24h)
    4. run_wilcoxon_test              -- Wilcoxon signed-rank test for
                                         statistical significance across
                                         independent runs (10x Grind)

Integration
-----------
This module consumes:
    - ``StandardScaler`` and ``target_idx`` from ``dataset_utils.py``
    - Scaled prediction / target arrays from model inference loops
    - Error distributions (lists of per-run RMSE values) for Wilcoxon

References
----------
[1] Hyndman, R.J. & Koehler, A.B. (2006). "Another look at measures
    of forecast accuracy." International Journal of Forecasting, 22(4),
    pp. 679--688.
[2] Wilcoxon, F. (1945). "Individual comparisons by ranking methods."
    Biometrics Bulletin, 1(6), pp. 80--83.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

__all__ = [
    "inverse_transform_predictions",
    "calculate_forecasting_metrics",
    "calculate_horizon_metrics",
    "run_wilcoxon_test",
]


# =====================================================================
# SECTION 1: INVERSE SCALING
# =====================================================================

def inverse_transform_predictions(
    preds: np.ndarray,
    targets: np.ndarray,
    scaler,
    target_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse-transform scaled predictions and targets back to original
    real-world units using the fitted ``StandardScaler``.

    Because the scaler was fitted on *all* continuous features jointly,
    we cannot call ``scaler.inverse_transform`` on the target column
    alone.  Instead we construct a dummy array with the correct number
    of columns, place the target values into the appropriate column,
    inverse-transform the whole array, and extract just the target
    column back out.

    For ``StandardScaler`` this is equivalent to:

        x_original = x_scaled * scale_[target_idx] + mean_[target_idx]

    but using the full inverse_transform path is safer and works with
    any scikit-learn transformer (``MinMaxScaler``, ``RobustScaler``,
    etc.) without assumptions about internal attribute names.

    Parameters
    ----------
    preds : np.ndarray, shape (N, H) or (N,)
        Scaled model predictions.
    targets : np.ndarray, shape (N, H) or (N,)
        Scaled ground-truth values.
    scaler : sklearn.preprocessing scaler
        The fitted scaler from ``dataset_utils.split_and_scale_data``.
    target_idx : int
        Column index of the target variable within ``continuous_cols``
        (as returned by ``dataset_utils.split_and_scale_data``).

    Returns
    -------
    preds_original : np.ndarray, same shape as ``preds``
        Predictions in original units.
    targets_original : np.ndarray, same shape as ``targets``
        Ground truth in original units.
    """
    n_features = scaler.n_features_in_

    def _invert(scaled_array: np.ndarray) -> np.ndarray:
        original_shape = scaled_array.shape
        flat = scaled_array.flatten()

        # Build a dummy array where every column is zero except the
        # target column, so inverse_transform only affects that column.
        dummy = np.zeros((len(flat), n_features), dtype=np.float64)
        dummy[:, target_idx] = flat

        inverted = scaler.inverse_transform(dummy)[:, target_idx]
        return inverted.reshape(original_shape)

    return _invert(preds), _invert(targets)


# =====================================================================
# SECTION 2: POINT-FORECAST ERROR METRICS
# =====================================================================

def calculate_forecasting_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> Dict[str, float]:
    """
    Calculate standard point-forecasting error metrics.

    All inputs must be in **original (real-world) units** -- i.e.,
    after calling ``inverse_transform_predictions``.

    Metrics
    -------
    **RMSE -- Root Mean Squared Error**

        RMSE = sqrt( (1/N) * sum_{i=1}^{N} (y_true_i - y_pred_i)^2 )

    Penalises large errors more heavily than MAE due to the squared
    term.  Expressed in the same units as the target variable.

    **MAE -- Mean Absolute Error**

        MAE = (1/N) * sum_{i=1}^{N} |y_true_i - y_pred_i|

    Robust to outliers.  Provides a linear, interpretable measure of
    average deviation in original units.

    **MAPE -- Mean Absolute Percentage Error**

        MAPE = (100/N) * sum_{i=1}^{N} |y_true_i - y_pred_i| / |y_true_i|

    Scale-independent metric expressed as a percentage.  Caution:
    undefined when y_true_i == 0.  We guard against division by zero
    by adding a small ``epsilon`` to the denominator.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values in original units.
    y_pred : np.ndarray
        Predicted values in original units.
    epsilon : float
        Small constant added to the MAPE denominator to prevent
        division by zero (default 1e-8).

    Returns
    -------
    dict
        ``{"RMSE": float, "MAE": float, "MAPE": float}``
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    errors = y_true - y_pred

    # RMSE = sqrt( mean( (y_true - y_pred)^2 ) )
    rmse = np.sqrt(np.mean(errors ** 2))

    # MAE = mean( |y_true - y_pred| )
    mae = np.mean(np.abs(errors))

    # MAPE = 100 * mean( |y_true - y_pred| / (|y_true| + epsilon) )
    mape = 100.0 * np.mean(np.abs(errors) / (np.abs(y_true) + epsilon))

    return {"RMSE": float(rmse), "MAE": float(mae), "MAPE": float(mape)}


# =====================================================================
# SECTION 3: PER-HORIZON METRICS
# =====================================================================

def calculate_horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: List[int] | None = None,
    epsilon: float = 1e-8,
) -> Dict[int, Dict[str, float]]:
    """
    Calculate RMSE, MAE, and MAPE at specific forecasting horizons.

    This answers the thesis question: "How does forecast accuracy
    degrade as the prediction horizon increases?"

    For a horizon h, we evaluate the error on forecast step h only
    (column index h-1 in the prediction matrix), isolating the
    difficulty of predicting further into the future.

    Parameters
    ----------
    y_true : np.ndarray, shape (N, H)
        Ground-truth values in original units.  Each row is a full
        H-step forecast window; column j corresponds to the (j+1)-hour
        ahead prediction.
    y_pred : np.ndarray, shape (N, H)
        Predicted values in original units.
    horizons : list[int] or None
        Specific horizon steps to evaluate (1-indexed).
        Default: ``[1, 6, 12, 24]``.
    epsilon : float
        MAPE stability constant (default 1e-8).

    Returns
    -------
    dict[int, dict]
        Nested dictionary keyed by horizon, e.g.::

            {
                1:  {"RMSE": 150.2, "MAE": 112.5, "MAPE": 1.23},
                6:  {"RMSE": 210.3, "MAE": 158.1, "MAPE": 1.78},
                12: {"RMSE": 280.7, "MAE": 210.4, "MAPE": 2.31},
                24: {"RMSE": 350.1, "MAE": 265.0, "MAPE": 2.94},
            }
    """
    if horizons is None:
        horizons = [1, 6, 12, 24]

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    max_h = y_true.shape[1]
    results: Dict[int, Dict[str, float]] = {}

    for h in horizons:
        if h > max_h:
            raise ValueError(
                f"Requested horizon {h} exceeds the prediction width "
                f"{max_h}.  Available horizons: 1..{max_h}."
            )

        # Column index h-1 gives the h-step-ahead forecast
        col = h - 1
        metrics = calculate_forecasting_metrics(
            y_true[:, col], y_pred[:, col], epsilon=epsilon,
        )
        results[h] = metrics

    return results


# =====================================================================
# SECTION 4: STATISTICAL SIGNIFICANCE TESTING
# =====================================================================

def run_wilcoxon_test(
    errors_model_a: np.ndarray | List[float],
    errors_model_b: np.ndarray | List[float],
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test for paired samples (Wilcoxon, 1945).

    Used to determine whether the Champion Model (AO-PSO-TFT) produces
    statistically significantly different error distributions compared
    to a baseline (e.g., LSTM, Untuned TFT) across the 10x independent
    runs mandated by the testing matrix.

    The null hypothesis H_0 is that the median difference between the
    paired error observations is zero (i.e., both models perform
    equally).  A p-value below the significance threshold (typically
    alpha = 0.05) leads us to reject H_0 and conclude that the
    performance difference is statistically significant.

    Procedure
    ---------
    Given paired error samples (e_A_i, e_B_i) for i = 1..n:

    1. Compute differences d_i = e_A_i - e_B_i
    2. Rank |d_i| (discarding zeros)
    3. Assign signs to ranks based on sign(d_i)
    4. W = sum of signed ranks
    5. p-value from the Wilcoxon distribution (exact for n <= 25,
       normal approximation otherwise)

    Parameters
    ----------
    errors_model_a : array-like, shape (n_runs,)
        Per-run error metric (e.g., RMSE) for Model A (Champion).
    errors_model_b : array-like, shape (n_runs,)
        Per-run error metric for Model B (Baseline).
    alternative : str
        Sidedness of the test.  One of:
        - ``"two-sided"``: H_1: median difference != 0
        - ``"less"``:      H_1: Model A < Model B (Champion is better)
        - ``"greater"``:   H_1: Model A > Model B

    Returns
    -------
    dict
        ``{"W_statistic": float, "p_value": float}``

    Raises
    ------
    ValueError
        If the input arrays have different lengths or fewer than 5
        paired observations (insufficient for meaningful inference).
    """
    a = np.asarray(errors_model_a, dtype=np.float64)
    b = np.asarray(errors_model_b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(
            f"Input arrays must have the same shape, got "
            f"{a.shape} and {b.shape}."
        )

    n_pairs = len(a)
    if n_pairs < 5:
        raise ValueError(
            f"Wilcoxon test requires at least 5 paired observations "
            f"for meaningful inference, got {n_pairs}."
        )

    # scipy.stats.wilcoxon performs the signed-rank test on paired
    # samples.  When all differences are zero, it raises a ValueError;
    # we let that propagate since identical results mean significance
    # testing is not applicable.
    result = stats.wilcoxon(a, b, alternative=alternative)

    return {
        "W_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }
