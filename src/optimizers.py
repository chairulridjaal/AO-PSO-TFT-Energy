"""
optimizers.py -- Meta-Heuristic Hyperparameter Optimizers for BaseTFT
=====================================================================

Research: Hybrid AO-PSO Optimised Temporal Fusion Transformer
           for Smart Grid Multi-Horizon Energy Forecasting

This module implements three swarm-intelligence optimizers designed to
find optimal hyperparameters for the BaseTFT model:

    1. AquilaOptimizer  -- Pure Aquila Optimizer (Abualigah et al., 2021)
                           Global exploration via 4-phase hunting strategy.
    2. ParticleSwarm     -- Pure PSO (Kennedy & Eberhart, 1995)
                           Local exploitation via velocity-position updates.
    3. Hybrid_AO_PSO     -- Champion Algorithm
                           AO global exploration → hand-off → PSO local
                           exploitation in a two-phase cascade.

Search Space
------------
The optimizers tune five hyperparameters of BaseTFT:

    Dimension  Parameter            Type        Range
    ─────────  ─────────            ────        ─────
    0          learning_rate        continuous   [1e-5, 1e-2]   (log-scale)
    1          d_model (hidden)     discrete     {32, 64, 128, 256}
    2          num_encoder_layers   discrete     {1, 2, 3, 4}
    3          n_heads              discrete     {1, 2, 4, 8}
    4          dropout              continuous   [0.05, 0.50]

Constraint: d_model must be divisible by n_heads.  This is enforced
            after every position update via ``_enforce_constraints()``.

References
----------
[1] Abualigah, L., Yousri, D., Abd Elaziz, M., Ewees, A.A.,
    Al-qaness, M.A.A. & Gandomi, A.H. (2021). "Aquila Optimizer:
    A novel meta-heuristic optimization Algorithm." Computers &
    Industrial Engineering, 157, 107250.
[2] Kennedy, J. & Eberhart, R. (1995). "Particle Swarm Optimization."
    Proceedings of ICNN'95, pp. 1942--1948.
[3] Lim, B. et al. (2021). "Temporal Fusion Transformers for
    Interpretable Multi-horizon Time Series Forecasting."
    International Journal of Forecasting, 37(4), pp. 1748--1764.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.models import BaseTFT

__all__ = [
    "SearchSpace",
    "ObjectiveFunction",
    "AquilaOptimizer",
    "ParticleSwarm",
    "Hybrid_AO_PSO",
]

# Module-level logger -- convergence data is emitted here so callers
# can capture it via standard Python logging handlers.
logger = logging.getLogger(__name__)


# =====================================================================
# SECTION 1: SEARCH SPACE DEFINITION
# =====================================================================

# --- Allowed discrete values for each discrete hyperparameter ---------
D_MODEL_OPTIONS = np.array([32, 64, 128, 256])
NUM_LAYERS_OPTIONS = np.array([1, 2, 3, 4])
N_HEADS_OPTIONS = np.array([1, 2, 4, 8])

# --- Continuous bounds (min, max) for each dimension ------------------
#   dim 0: learning_rate  (optimised in log10 space internally)
#   dim 1: d_model        (snapped to nearest allowed value)
#   dim 2: num_encoder_layers (snapped to nearest allowed value)
#   dim 3: n_heads        (snapped; then validated against d_model)
#   dim 4: dropout        (continuous, clamped to bounds)
LOWER_BOUNDS = np.array([
    np.log10(1e-5),              # dim 0: log10(lr) = -5.0
    float(D_MODEL_OPTIONS.min()),  # dim 1: 32
    float(NUM_LAYERS_OPTIONS.min()),  # dim 2: 1
    float(N_HEADS_OPTIONS.min()),  # dim 3: 1
    0.05,                        # dim 4: dropout
])

UPPER_BOUNDS = np.array([
    np.log10(1e-2),              # dim 0: log10(lr) = -2.0
    float(D_MODEL_OPTIONS.max()),  # dim 1: 256
    float(NUM_LAYERS_OPTIONS.max()),  # dim 2: 4
    float(N_HEADS_OPTIONS.max()),  # dim 3: 8
    0.50,                        # dim 4: dropout
])

N_DIMS = len(LOWER_BOUNDS)  # 5


def _snap_to_nearest(value: float, options: np.ndarray) -> float:
    """Snap a continuous value to its nearest discrete option."""
    idx = np.argmin(np.abs(options - value))
    return float(options[idx])


def _enforce_constraints(position: np.ndarray) -> np.ndarray:
    """
    Enforce search-space boundaries and the d_model % n_heads == 0
    divisibility constraint on a single position vector.

    Steps:
        1. Clamp all dimensions to [LOWER_BOUNDS, UPPER_BOUNDS].
        2. Snap discrete dimensions (d_model, num_layers, n_heads)
           to their nearest allowed values.
        3. If d_model is not divisible by n_heads, find the largest
           valid n_heads value that divides d_model.

    Parameters
    ----------
    position : np.ndarray, shape (N_DIMS,)
        Raw position vector (possibly out of bounds after an update).

    Returns
    -------
    np.ndarray, shape (N_DIMS,)
        Constraint-satisfying position vector.
    """
    pos = position.copy()

    # Step 1: Clamp to hard bounds
    pos = np.clip(pos, LOWER_BOUNDS, UPPER_BOUNDS)

    # Step 2: Snap discrete variables to nearest allowed values
    pos[1] = _snap_to_nearest(pos[1], D_MODEL_OPTIONS)
    pos[2] = _snap_to_nearest(pos[2], NUM_LAYERS_OPTIONS)
    pos[3] = _snap_to_nearest(pos[3], N_HEADS_OPTIONS)

    # Step 3: Enforce d_model % n_heads == 0
    # If the snapped n_heads doesn't divide d_model, choose the
    # largest n_heads option that does divide d_model.
    d_model = int(pos[1])
    n_heads = int(pos[3])
    if d_model % n_heads != 0:
        valid_heads = N_HEADS_OPTIONS[d_model % N_HEADS_OPTIONS == 0]
        if len(valid_heads) > 0:
            # Pick the valid n_heads closest to the original value
            pos[3] = float(valid_heads[np.argmin(np.abs(valid_heads - n_heads))])
        else:
            # Fallback: 1 head always divides any d_model
            pos[3] = 1.0

    return pos


def decode_position(position: np.ndarray) -> Dict[str, Any]:
    """
    Decode a raw position vector into a human-readable hyperparameter
    dictionary suitable for passing to ``BaseTFT.__init__``.

    Parameters
    ----------
    position : np.ndarray, shape (N_DIMS,)
        Constrained position vector.

    Returns
    -------
    dict
        Keys: learning_rate, d_model, num_encoder_layers, n_heads,
              dropout.
    """
    return {
        "learning_rate": 10 ** position[0],     # undo log10
        "d_model": int(position[1]),
        "num_encoder_layers": int(position[2]),
        "n_heads": int(position[3]),
        "dropout": float(position[4]),
    }


# =====================================================================
# SECTION 2: CONVERGENCE LOGGER
# =====================================================================

@dataclass
class ConvergenceRecord:
    """
    Structured log of optimiser convergence data for later plotting
    (Group A: Optimizer Overhead & Convergence).

    Attributes
    ----------
    optimizer_name : str
        Identifier string (e.g. "AO", "PSO", "Hybrid_AO_PSO").
    history : list[dict]
        Per-iteration records containing at minimum:
        ``iteration``, ``best_fitness``, ``mean_fitness``,
        ``wall_time_s``, ``best_params``.
    """
    optimizer_name: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)

    def log_iteration(
        self,
        iteration: int,
        best_fitness: float,
        mean_fitness: float,
        wall_time_s: float,
        best_params: Dict[str, Any],
    ) -> None:
        """Append one iteration record and emit a log message."""
        record = {
            "iteration": iteration,
            "best_fitness": best_fitness,
            "mean_fitness": mean_fitness,
            "wall_time_s": round(wall_time_s, 2),
            "best_params": best_params,
        }
        self.history.append(record)
        logger.info(
            "[%s] iter %3d | best_RMSE=%.6f | mean_RMSE=%.6f | "
            "elapsed=%.1fs | lr=%.2e d=%d L=%d H=%d do=%.3f",
            self.optimizer_name,
            iteration,
            best_fitness,
            mean_fitness,
            wall_time_s,
            best_params["learning_rate"],
            best_params["d_model"],
            best_params["num_encoder_layers"],
            best_params["n_heads"],
            best_params["dropout"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full record for JSON checkpointing."""
        return {
            "optimizer_name": self.optimizer_name,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConvergenceRecord":
        """Reconstruct from a serialised dict."""
        return cls(
            optimizer_name=data["optimizer_name"],
            history=data["history"],
        )


# =====================================================================
# SECTION 3: OBJECTIVE FUNCTION (FITNESS EVALUATOR)
# =====================================================================

class ObjectiveFunction:
    """
    Fitness evaluator for meta-heuristic optimizers.

    Given a position vector encoding hyperparameters, this callable:
        1. Decodes the position to a hyperparameter dict.
        2. Instantiates a ``BaseTFT`` with those hyperparameters.
        3. Trains it for a small number of proxy epochs on a random
           subset of the training data (fast proxy evaluation).
        4. Evaluates validation RMSE as the fitness score to minimise.

    Using a subset + few epochs provides a noisy but informative
    signal of hyperparameter quality at ~1/50th the cost of a full
    training run.  The Aquila Optimizer's stochastic search is robust
    to this noise (Abualigah et al., 2021, Section 4.2).

    Parameters
    ----------
    train_loader : DataLoader
        Full training DataLoader (batches are sub-sampled internally).
    val_loader : DataLoader
        Full validation DataLoader (sub-sampled via ``val_subset_fraction``).
    n_encoder_features : int
        Number of continuous input features (F in ``(B, T, F)``).
    window_size : int
        Encoder lookback window length (default 168).
    horizon : int
        Forecast horizon (default 24).
    proxy_epochs : int
        Number of training epochs per fitness evaluation (default 2).
    subset_fraction : float
        Fraction of training batches used per evaluation (default 0.3).
    val_subset_fraction : float
        Fraction of validation batches used per evaluation (default 1.0).
        Set lower for smoke tests to reduce evaluation time.
    device : str
        PyTorch device string (default "cuda" if available).
    """

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_encoder_features: int,
        window_size: int = 168,
        horizon: int = 24,
        proxy_epochs: int = 2,
        subset_fraction: float = 0.3,
        val_subset_fraction: float = 1.0,
        device: Optional[str] = None,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_encoder_features = n_encoder_features
        self.window_size = window_size
        self.horizon = horizon
        self.proxy_epochs = proxy_epochs
        self.subset_fraction = subset_fraction
        self.val_subset_fraction = val_subset_fraction
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Pre-compute which training batch indices to use.
        # Fixed across evaluations within one optimizer run for
        # fairness (all candidates see the same data).
        n_batches = len(train_loader)
        n_subset = max(1, int(n_batches * subset_fraction))
        self._rng = np.random.default_rng(42)
        self._subset_indices = sorted(
            self._rng.choice(n_batches, size=n_subset, replace=False).tolist()
        )

        # Pre-compute which validation batch indices to use.
        n_val_batches = len(val_loader)
        if val_subset_fraction < 1.0:
            n_val_subset = max(1, int(n_val_batches * val_subset_fraction))
            self._val_subset_indices: Optional[set] = set(sorted(
                self._rng.choice(
                    n_val_batches, size=n_val_subset, replace=False
                ).tolist()
            ))
        else:
            n_val_subset = n_val_batches
            self._val_subset_indices = None  # Use all

        logger.info(
            "[ObjectiveFunction] Using %d/%d training batches (%.0f%%) "
            "x %d proxy epochs on device='%s'",
            n_subset, n_batches, subset_fraction * 100,
            proxy_epochs, self.device,
        )
        logger.info(
            "[ObjectiveFunction] Using %d/%d validation batches (%.0f%%)",
            n_val_subset, n_val_batches, val_subset_fraction * 100,
        )

    def __call__(self, position: np.ndarray) -> float:
        """
        Evaluate fitness of a single hyperparameter position vector.

        Parameters
        ----------
        position : np.ndarray, shape (N_DIMS,)
            Constrained position vector from an optimizer.

        Returns
        -------
        float
            Validation RMSE (lower is better).  Returns ``float('inf')``
            on any training/evaluation error (e.g. NaN loss, OOM).
        """
        params = decode_position(position)

        try:
            # ── 1. Instantiate model with candidate hyperparameters ──
            model = BaseTFT(
                n_encoder_features=self.n_encoder_features,
                n_decoder_features=0,          # encoder-only mode
                n_static_categoricals=0,       # no static categoricals
                d_model=params["d_model"],
                n_heads=params["n_heads"],
                num_encoder_layers=params["num_encoder_layers"],
                dropout=params["dropout"],
                horizon=self.horizon,
                window_size=self.window_size,
            ).to(self.device)

            # ── 2. Quick proxy training ──────────────────────────────
            optimizer = torch.optim.Adam(
                model.parameters(), lr=params["learning_rate"]
            )
            criterion = nn.MSELoss()

            model.train()
            for _epoch in range(self.proxy_epochs):
                for batch_idx, (x_cont, x_cat, y) in enumerate(self.train_loader):
                    if batch_idx not in self._subset_indices:
                        continue

                    x_cont = x_cont.to(self.device)
                    y = y.to(self.device)

                    pred = model(x_cont)
                    loss = criterion(pred, y)

                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(
                            "[ObjectiveFunction] NaN/Inf loss for params %s",
                            params,
                        )
                        return float("inf")

                    optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping to prevent exploding gradients
                    # during short proxy runs with potentially extreme lr
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            # ── 3. Validation RMSE ───────────────────────────────────
            model.eval()
            val_mse_sum = 0.0
            val_count = 0

            with torch.no_grad():
                for val_idx, (x_cont, x_cat, y) in enumerate(self.val_loader):
                    if (self._val_subset_indices is not None
                            and val_idx not in self._val_subset_indices):
                        continue

                    x_cont = x_cont.to(self.device)
                    y = y.to(self.device)

                    pred = model(x_cont)
                    # Accumulate sum-of-squared-errors over all samples
                    val_mse_sum += ((pred - y) ** 2).sum().item()
                    val_count += y.numel()

            val_rmse = math.sqrt(val_mse_sum / val_count)

            if math.isnan(val_rmse) or math.isinf(val_rmse):
                return float("inf")

            return val_rmse

        except Exception as e:
            logger.warning(
                "[ObjectiveFunction] Exception for params %s: %s",
                params, e,
            )
            return float("inf")

        finally:
            # Aggressively free GPU memory between evaluations
            if "model" in dir():
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# =====================================================================
# SECTION 4: AQUILA OPTIMIZER
# =====================================================================

class AquilaOptimizer:
    """
    Aquila Optimizer (AO) -- Abualigah et al. (2021).

    The AO is a population-based metaheuristic inspired by the hunting
    behaviour of Aquila eagles.  It balances exploration and exploitation
    through four mathematically distinct phases:

    Phase 1 -- Expanded Exploration  (high-altitude soaring)
        The eagle uses thermals to soar high and select a broad region.
        Position update: X_new = X_best * (1 - t/T) + X_mean - X_rand
        This phase provides maximal search-space coverage early on.

    Phase 2 -- Narrowed Exploration  (contour flight with Lévy spiral)
        The eagle circles prey using Lévy flight patterns.
        Position update: X_new = X_best * Levy(D) + X_rand +
                          (y - x) * rand
        Lévy flights inject rare long jumps, preventing premature
        convergence while still exploring near promising regions.

    Phase 3 -- Expanded Exploitation (slow descent approach)
        The eagle descends towards prey with controlled gliding.
        Position update: X_new = (X_best - X_mean) * alpha - rand +
                          ((UB - LB) * rand + LB) * delta
        alpha and delta decay with iteration, shrinking the step size
        as the algorithm converges.

    Phase 4 -- Narrowed Exploitation (walk & grab with Lévy flight)
        Final close-quarters precision approach using quality function.
        Position update: X_new = QF * X_best - (G1 * X_new1 * rand)
                          - G2 * Levy(D) + rand * G1
        QF (Quality Function) = t^{2*rand-1} * (1 - t/T)^2
        provides an adaptive step control that diminishes to near-zero.

    Phase selection is determined by ``t/T`` (iteration progress) and
    a random threshold ``r``:
        - t/T <= 2/3 AND r <= 0.5  →  Phase 1 (Expanded Exploration)
        - t/T <= 2/3 AND r >  0.5  →  Phase 2 (Narrowed Exploration)
        - t/T >  2/3 AND r <= 0.5  →  Phase 3 (Expanded Exploitation)
        - t/T >  2/3 AND r >  0.5  →  Phase 4 (Narrowed Exploitation)

    Parameters
    ----------
    objective_fn : ObjectiveFunction
        Callable returning fitness (validation RMSE) for a position.
    pop_size : int
        Population (eagle flock) size.
    max_iter : int
        Maximum number of iterations.
    seed : int
        Random seed for reproducibility.
    checkpoint_dir : str or None
        Directory for JSON checkpoints.  None = no checkpointing.
    """

    def __init__(
        self,
        objective_fn: ObjectiveFunction,
        pop_size: int = 20,
        max_iter: int = 50,
        seed: int = 42,
        checkpoint_dir: Optional[str] = None,
    ):
        self.objective_fn = objective_fn
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.checkpoint_dir = checkpoint_dir

        self.rng = np.random.default_rng(seed)

        # Convergence log
        self.convergence = ConvergenceRecord(optimizer_name="AO")

        # Population initialisation: Latin Hypercube-style stratified
        # sampling across the search space for better initial coverage.
        self.population = np.zeros((pop_size, N_DIMS))
        for i in range(pop_size):
            for d in range(N_DIMS):
                self.population[i, d] = LOWER_BOUNDS[d] + (
                    self.rng.random() * (UPPER_BOUNDS[d] - LOWER_BOUNDS[d])
                )
            self.population[i] = _enforce_constraints(self.population[i])

        # Fitness tracking
        self.fitness = np.full(pop_size, float("inf"))
        self.global_best_pos = self.population[0].copy()
        self.global_best_fit = float("inf")

    # -----------------------------------------------------------------
    # Lévy flight (used in Phases 2 and 4)
    # -----------------------------------------------------------------
    def _levy_flight(self, size: int, beta: float = 1.5) -> np.ndarray:
        """
        Generate a Lévy flight step vector using Mantegna's algorithm.

        The Lévy distribution has infinite variance, producing
        occasional long jumps that allow the algorithm to escape
        local optima.  The step length follows:

            s = u / |v|^{1/beta}

        where u ~ N(0, sigma_u^2) and v ~ N(0, 1), with
        sigma_u = (Gamma(1+beta) * sin(pi*beta/2) /
                   (Gamma((1+beta)/2) * beta * 2^{(beta-1)/2}))^{1/beta}

        Parameters
        ----------
        size : int
            Dimensionality of the step vector.
        beta : float
            Lévy exponent (1 < beta <= 2).  beta=1.5 is the standard
            choice providing a good mix of short and long steps.

        Returns
        -------
        np.ndarray, shape (size,)
            Lévy flight step vector.
        """
        # Mantegna's formula for sigma_u
        numerator = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
        )
        denominator = (
            math.gamma((1 + beta) / 2)
            * beta
            * 2 ** ((beta - 1) / 2)
        )
        sigma_u = (numerator / denominator) ** (1 / beta)

        u = self.rng.normal(0, sigma_u, size=size)
        v = self.rng.normal(0, 1, size=size)

        step = u / (np.abs(v) ** (1 / beta))
        return step

    # -----------------------------------------------------------------
    # Core optimisation loop
    # -----------------------------------------------------------------
    def optimize(self) -> Tuple[Dict[str, Any], float, ConvergenceRecord]:
        """
        Run the Aquila Optimizer for ``max_iter`` iterations.

        Returns
        -------
        best_params : dict
            Best hyperparameters found (decoded).
        best_fitness : float
            Best validation RMSE achieved.
        convergence : ConvergenceRecord
            Full iteration-by-iteration convergence log.
        """
        logger.info(
            "[AO] Starting optimization: pop=%d, max_iter=%d",
            self.pop_size, self.max_iter,
        )

        # --- Initial population evaluation ---
        for i in range(self.pop_size):
            self.fitness[i] = self.objective_fn(self.population[i])
            if self.fitness[i] < self.global_best_fit:
                self.global_best_fit = self.fitness[i]
                self.global_best_pos = self.population[i].copy()

        start_time = time.time()

        for t in range(1, self.max_iter + 1):
            iter_start = time.time()

            # Iteration progress ratio (controls exploration/exploitation balance)
            progress = t / self.max_iter   # ∈ (0, 1]

            # Mean position of the population (centroid)
            X_mean = self.population.mean(axis=0)

            for i in range(self.pop_size):
                r = self.rng.random()  # Phase selection threshold

                # ── Phase 1: Expanded Exploration ────────────────
                # Condition: early stage AND r favours exploration
                # Mimics high-altitude soaring using thermals.
                # The eagle leverages the global best and population
                # centroid for broad spatial coverage.
                #
                #   X_new = X_best * (1 - t/T) + X_mean - X_rand_j
                #
                # As t/T increases, the X_best contribution shrinks,
                # shifting weight towards the centroid (exploitation).
                if progress <= (2 / 3) and r <= 0.5:
                    j = self.rng.integers(0, self.pop_size)
                    X_new = (
                        self.global_best_pos * (1 - progress)
                        + X_mean
                        - self.population[j]
                    )

                # ── Phase 2: Narrowed Exploration ────────────────
                # Condition: early stage AND r favours targeted search
                # Mimics contour flight -- circling prey at medium
                # altitude using a Lévy-modulated spiral.
                #
                #   X_new = X_best * Levy(D) + X_rand_j +
                #           (y - x) * rand
                #
                # where:
                #   x = r1 * sin(theta), y = r1 * cos(theta)
                #   r1 = 2 * pi * rand,  theta = -pi + 2*pi*rand
                # The spiral trajectory + Lévy flight creates a rich
                # search pattern around the best-known region.
                elif progress <= (2 / 3) and r > 0.5:
                    j = self.rng.integers(0, self.pop_size)
                    levy = self._levy_flight(N_DIMS)
                    theta = -math.pi + 2 * math.pi * self.rng.random()
                    r1 = 2 * math.pi * self.rng.random()
                    x_spiral = r1 * math.sin(theta)
                    y_spiral = r1 * math.cos(theta)

                    X_new = (
                        self.global_best_pos * levy
                        + self.population[j]
                        + (y_spiral - x_spiral) * self.rng.random(N_DIMS)
                    )

                # ── Phase 3: Expanded Exploitation ───────────────
                # Condition: late stage AND r favours broad refinement
                # Mimics the eagle's slow controlled descent towards
                # the prey, approaching from a computed angle.
                #
                #   alpha = delta = 0.1 (small perturbation)
                #   X_new = (X_best - X_mean) * alpha
                #           - rand(D)
                #           + ((UB - LB) * rand + LB) * delta
                #
                # The (X_best - X_mean) term orients movement towards
                # the best, while alpha/delta constrain step magnitude.
                elif progress > (2 / 3) and r <= 0.5:
                    alpha = 0.1
                    delta = 0.1
                    X_new = (
                        (self.global_best_pos - X_mean) * alpha
                        - self.rng.random(N_DIMS)
                        + (
                            (UPPER_BOUNDS - LOWER_BOUNDS)
                            * self.rng.random(N_DIMS)
                            + LOWER_BOUNDS
                        ) * delta
                    )

                # ── Phase 4: Narrowed Exploitation ───────────────
                # Condition: late stage AND r favours precision search
                # Mimics the final "walk and grab" attack pattern.
                #
                #   QF = t^{(2*rand - 1)} * (1 - t/T)^2
                #   G1 = 2 * rand - 1
                #   G2 = 2 * (1 - t/T)
                #   X_new = QF * X_best
                #           - (G1 * population[i] * rand)
                #           - G2 * Levy(D) + rand(D) * G1
                #
                # QF (Quality Function) decays quadratically with
                # progress, ensuring micro-scale adjustments in the
                # final iterations.  G2 also decays, further
                # reducing Lévy jump magnitude.
                else:  # progress > 2/3 and r > 0.5
                    QF = t ** (2 * self.rng.random() - 1) * (1 - progress) ** 2
                    G1 = 2 * self.rng.random() - 1  # ∈ [-1, 1]
                    G2 = 2 * (1 - progress)          # decays 2 → 0
                    levy = self._levy_flight(N_DIMS)

                    X_new = (
                        QF * self.global_best_pos
                        - G1 * self.population[i] * self.rng.random(N_DIMS)
                        - G2 * levy
                        + self.rng.random(N_DIMS) * G1
                    )

                # ── Enforce constraints and evaluate ─────────────
                X_new = _enforce_constraints(X_new)
                new_fit = self.objective_fn(X_new)

                # Greedy selection: keep new position only if better
                if new_fit < self.fitness[i]:
                    self.population[i] = X_new
                    self.fitness[i] = new_fit

                    if new_fit < self.global_best_fit:
                        self.global_best_fit = new_fit
                        self.global_best_pos = X_new.copy()

            # ── Iteration logging ────────────────────────────────
            elapsed = time.time() - start_time
            self.convergence.log_iteration(
                iteration=t,
                best_fitness=self.global_best_fit,
                mean_fitness=float(self.fitness.mean()),
                wall_time_s=elapsed,
                best_params=decode_position(self.global_best_pos),
            )

            # ── Checkpoint ───────────────────────────────────────
            if self.checkpoint_dir is not None:
                self._save_checkpoint(t)

        best_params = decode_position(self.global_best_pos)
        logger.info(
            "[AO] Optimization complete. Best RMSE=%.6f | Params: %s",
            self.global_best_fit, best_params,
        )

        return best_params, self.global_best_fit, self.convergence

    # -----------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------
    def _save_checkpoint(self, iteration: int) -> None:
        """Save optimizer state to JSON for crash recovery."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, "ao_checkpoint.json")
        state = {
            "iteration": iteration,
            "population": self.population.tolist(),
            "fitness": self.fitness.tolist(),
            "global_best_pos": self.global_best_pos.tolist(),
            "global_best_fit": self.global_best_fit,
            "convergence": self.convergence.to_dict(),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_checkpoint(self, path: str) -> int:
        """
        Restore optimizer state from a JSON checkpoint.

        Parameters
        ----------
        path : str
            Path to the checkpoint JSON file.

        Returns
        -------
        int
            Iteration number from which to resume.
        """
        with open(path, "r") as f:
            state = json.load(f)

        self.population = np.array(state["population"])
        self.fitness = np.array(state["fitness"])
        self.global_best_pos = np.array(state["global_best_pos"])
        self.global_best_fit = state["global_best_fit"]
        self.convergence = ConvergenceRecord.from_dict(state["convergence"])

        logger.info(
            "[AO] Resumed from checkpoint at iteration %d (best=%.6f)",
            state["iteration"], self.global_best_fit,
        )
        return state["iteration"]


# =====================================================================
# SECTION 5: PARTICLE SWARM OPTIMIZATION
# =====================================================================

class ParticleSwarm:
    """
    Particle Swarm Optimization (PSO) -- Kennedy & Eberhart (1995).

    PSO models a population of particles flying through the search
    space, accelerating towards their personal best (pBest) and the
    global best (gBest) discovered by the swarm.

    Velocity-position update equations:

        v_i(t+1) = w * v_i(t)
                   + c1 * r1 * (pBest_i - x_i(t))    [cognitive]
                   + c2 * r2 * (gBest   - x_i(t))    [social]

        x_i(t+1) = x_i(t) + v_i(t+1)

    Inertia weight ``w`` linearly decays from ``w_max`` to ``w_min``:
        w(t) = w_max - (w_max - w_min) * t / T

    This balances:
        - High inertia early → large velocity → exploration
        - Low inertia late  → small velocity → exploitation

    Velocity clamping: |v_d| <= V_max_d = 0.2 * (UB_d - LB_d)
    prevents particles from overshooting the search space.

    Parameters
    ----------
    objective_fn : ObjectiveFunction
        Callable returning fitness (validation RMSE) for a position.
    pop_size : int
        Swarm size (number of particles).
    max_iter : int
        Maximum iterations for PSO.
    c1 : float
        Cognitive acceleration coefficient (personal best pull).
    c2 : float
        Social acceleration coefficient (global best pull).
    w_max : float
        Initial (maximum) inertia weight.
    w_min : float
        Final (minimum) inertia weight.
    seed : int
        Random seed for reproducibility.
    checkpoint_dir : str or None
        Directory for JSON checkpoints.
    """

    def __init__(
        self,
        objective_fn: ObjectiveFunction,
        pop_size: int = 20,
        max_iter: int = 50,
        c1: float = 2.0,
        c2: float = 2.0,
        w_max: float = 0.9,
        w_min: float = 0.4,
        seed: int = 42,
        checkpoint_dir: Optional[str] = None,
    ):
        self.objective_fn = objective_fn
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.checkpoint_dir = checkpoint_dir

        self.rng = np.random.default_rng(seed)

        # Convergence log
        self.convergence = ConvergenceRecord(optimizer_name="PSO")

        # Velocity clamping limits: 20% of search range per dimension
        self.v_max = 0.2 * (UPPER_BOUNDS - LOWER_BOUNDS)

        # Particle initialisation
        self.positions = np.zeros((pop_size, N_DIMS))
        self.velocities = np.zeros((pop_size, N_DIMS))
        for i in range(pop_size):
            for d in range(N_DIMS):
                self.positions[i, d] = LOWER_BOUNDS[d] + (
                    self.rng.random() * (UPPER_BOUNDS[d] - LOWER_BOUNDS[d])
                )
                # Initialise velocities to small random values
                self.velocities[i, d] = (
                    self.rng.uniform(-1, 1) * self.v_max[d] * 0.1
                )
            self.positions[i] = _enforce_constraints(self.positions[i])

        # Personal bests (each particle remembers its best position)
        self.p_best_pos = self.positions.copy()
        self.p_best_fit = np.full(pop_size, float("inf"))

        # Global best (best across the entire swarm)
        self.g_best_pos = self.positions[0].copy()
        self.g_best_fit = float("inf")

    def _inject_swarm_state(
        self,
        positions: np.ndarray,
        fitness: np.ndarray,
        g_best_pos: np.ndarray,
        g_best_fit: float,
    ) -> None:
        """
        Inject external state into the PSO swarm (used for AO→PSO
        hand-off in the Hybrid algorithm).

        This method directly sets:
            - Particle positions and personal bests from AO's final
              population.
            - The global best from AO's discovered optimum.
            - Velocities to zero (PSO will develop its own momentum
              from the injected positions).

        Parameters
        ----------
        positions : np.ndarray, shape (pop_size, N_DIMS)
            Population positions from the AO phase.
        fitness : np.ndarray, shape (pop_size,)
            Fitness values corresponding to each position.
        g_best_pos : np.ndarray, shape (N_DIMS,)
            Global best position from AO.
        g_best_fit : float
            Global best fitness from AO.
        """
        self.positions = positions.copy()
        self.p_best_pos = positions.copy()
        self.p_best_fit = fitness.copy()
        self.g_best_pos = g_best_pos.copy()
        self.g_best_fit = g_best_fit

        # Reset velocities to zero -- PSO rebuilds momentum from
        # the AO-discovered landscape, preventing the swarm from
        # immediately scattering away from good regions.
        self.velocities = np.zeros_like(self.positions)

        logger.info(
            "[PSO] Swarm state injected from AO: gBest_fit=%.6f, "
            "pop_size=%d",
            g_best_fit, len(positions),
        )

    def optimize(self) -> Tuple[Dict[str, Any], float, ConvergenceRecord]:
        """
        Run PSO for ``max_iter`` iterations.

        Returns
        -------
        best_params : dict
            Best hyperparameters found (decoded).
        best_fitness : float
            Best validation RMSE achieved.
        convergence : ConvergenceRecord
            Full iteration-by-iteration convergence log.
        """
        logger.info(
            "[PSO] Starting optimization: pop=%d, max_iter=%d, "
            "c1=%.1f, c2=%.1f, w=%.2f→%.2f",
            self.pop_size, self.max_iter, self.c1, self.c2,
            self.w_max, self.w_min,
        )

        # --- Initial population evaluation ---
        # Skip if state was already injected (Hybrid hand-off)
        if self.g_best_fit == float("inf"):
            for i in range(self.pop_size):
                fit = self.objective_fn(self.positions[i])
                self.p_best_fit[i] = fit
                self.p_best_pos[i] = self.positions[i].copy()
                if fit < self.g_best_fit:
                    self.g_best_fit = fit
                    self.g_best_pos = self.positions[i].copy()

        start_time = time.time()

        for t in range(1, self.max_iter + 1):
            # ── Linearly decaying inertia weight ─────────────────
            # w(t) = w_max - (w_max - w_min) * t / T
            #
            # High w early: particles maintain momentum → exploration
            # Low w late:   particles decelerate → exploitation
            w = self.w_max - (self.w_max - self.w_min) * (t / self.max_iter)

            for i in range(self.pop_size):
                r1 = self.rng.random(N_DIMS)  # per-dimension random
                r2 = self.rng.random(N_DIMS)

                # ── Velocity update ──────────────────────────────
                #
                # v_i = w * v_i                              [inertia]
                #     + c1 * r1 * (pBest_i - x_i)           [cognitive]
                #     + c2 * r2 * (gBest   - x_i)           [social]
                #
                # Cognitive term: pulls particle towards its own
                # best-known position (individual memory).
                # Social term: pulls particle towards the swarm's
                # best-known position (collective intelligence).
                cognitive = self.c1 * r1 * (self.p_best_pos[i] - self.positions[i])
                social = self.c2 * r2 * (self.g_best_pos - self.positions[i])

                self.velocities[i] = w * self.velocities[i] + cognitive + social

                # ── Velocity clamping ────────────────────────────
                # Prevents particles from gaining excessive speed
                # and overshooting the search space.  Clamped to
                # [-V_max, +V_max] per dimension.
                self.velocities[i] = np.clip(
                    self.velocities[i], -self.v_max, self.v_max
                )

                # ── Position update ──────────────────────────────
                # x_i(t+1) = x_i(t) + v_i(t+1)
                self.positions[i] = self.positions[i] + self.velocities[i]

                # ── Enforce boundary & divisibility constraints ──
                self.positions[i] = _enforce_constraints(self.positions[i])

                # ── Evaluate fitness ─────────────────────────────
                fit = self.objective_fn(self.positions[i])

                # Update personal best
                if fit < self.p_best_fit[i]:
                    self.p_best_fit[i] = fit
                    self.p_best_pos[i] = self.positions[i].copy()

                    # Update global best
                    if fit < self.g_best_fit:
                        self.g_best_fit = fit
                        self.g_best_pos = self.positions[i].copy()

            # ── Iteration logging ────────────────────────────────
            elapsed = time.time() - start_time
            self.convergence.log_iteration(
                iteration=t,
                best_fitness=self.g_best_fit,
                mean_fitness=float(self.p_best_fit.mean()),
                wall_time_s=elapsed,
                best_params=decode_position(self.g_best_pos),
            )

            # ── Checkpoint ───────────────────────────────────────
            if self.checkpoint_dir is not None:
                self._save_checkpoint(t)

        best_params = decode_position(self.g_best_pos)
        logger.info(
            "[PSO] Optimization complete. Best RMSE=%.6f | Params: %s",
            self.g_best_fit, best_params,
        )

        return best_params, self.g_best_fit, self.convergence

    # -----------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------
    def _save_checkpoint(self, iteration: int) -> None:
        """Save PSO state to JSON."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, "pso_checkpoint.json")
        state = {
            "iteration": iteration,
            "positions": self.positions.tolist(),
            "velocities": self.velocities.tolist(),
            "p_best_pos": self.p_best_pos.tolist(),
            "p_best_fit": self.p_best_fit.tolist(),
            "g_best_pos": self.g_best_pos.tolist(),
            "g_best_fit": self.g_best_fit,
            "convergence": self.convergence.to_dict(),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_checkpoint(self, path: str) -> int:
        """Restore PSO state from a JSON checkpoint."""
        with open(path, "r") as f:
            state = json.load(f)

        self.positions = np.array(state["positions"])
        self.velocities = np.array(state["velocities"])
        self.p_best_pos = np.array(state["p_best_pos"])
        self.p_best_fit = np.array(state["p_best_fit"])
        self.g_best_pos = np.array(state["g_best_pos"])
        self.g_best_fit = state["g_best_fit"]
        self.convergence = ConvergenceRecord.from_dict(state["convergence"])

        logger.info(
            "[PSO] Resumed from checkpoint at iteration %d (best=%.6f)",
            state["iteration"], self.g_best_fit,
        )
        return state["iteration"]


# =====================================================================
# SECTION 6: HYBRID AO-PSO (THE CHAMPION ALGORITHM)
# =====================================================================

class Hybrid_AO_PSO:
    """
    Hybrid AO-PSO: two-phase meta-heuristic cascade.

    This is the core algorithmic innovation of the thesis.  The hybrid
    combines the complementary strengths of two optimizers:

        AO  → excels at global exploration (avoiding local minima)
        PSO → excels at local exploitation (precise convergence)

    Pipeline:

        Phase 1: GLOBAL MAP (AO)
        ────────────────────────────────────────────────────────────
        The Aquila Optimizer runs for the first ``ao_fraction`` of
        total iterations.  Its 4-phase hunting strategy aggressively
        explores the hyperparameter landscape, identifying the basin
        of attraction containing the global optimum.

                     ┌─────────────────────────────┐
                     │  AO explores search space    │
                     │  using 4 hunting phases       │
                     │  → identifies global basin    │
                     └──────────┬──────────────────┘
                                │
                     THE HAND-OFF
                                │
                     ┌──────────▼──────────────────┐
                     │  PSO receives:               │
                     │   • AO's final population    │
                     │   • AO's global best         │
                     │   • velocities zeroed        │
                     └──────────┬──────────────────┘
                                │
        Phase 2: LOCAL FINE-TUNING (PSO)
        ────────────────────────────────────────────────────────────
        PSO takes AO's final population as its starting swarm.
        Its velocity-position updates rapidly converge the already
        well-positioned particles onto the exact optimum within
        the basin that AO identified.

    The hand-off is the critical innovation:
        - AO's population IS the PSO starting positions.
        - AO's fitness values ARE the PSO pBest initialisations.
        - AO's global best IS the PSO gBest seed.
        - Velocities are zeroed so PSO builds fresh momentum
          from the AO-discovered landscape geometry.

    This avoids both:
        - AO's weakness: slow fine-grained convergence
        - PSO's weakness: premature convergence to local optima

    Parameters
    ----------
    objective_fn : ObjectiveFunction
        Callable returning fitness (validation RMSE).
    pop_size : int
        Population size shared between AO and PSO phases.
    total_iter : int
        Total optimisation budget (AO + PSO iterations combined).
    ao_fraction : float
        Fraction of total_iter allocated to Phase 1 (AO).
        Remaining ``1 - ao_fraction`` goes to Phase 2 (PSO).
    seed : int
        Random seed for reproducibility.
    checkpoint_dir : str or None
        Directory for JSON checkpoints.
    pso_c1 : float
        PSO cognitive coefficient.
    pso_c2 : float
        PSO social coefficient.
    pso_w_max : float
        PSO maximum inertia weight.
    pso_w_min : float
        PSO minimum inertia weight.
    """

    def __init__(
        self,
        objective_fn: ObjectiveFunction,
        pop_size: int = 20,
        total_iter: int = 50,
        ao_fraction: float = 0.5,
        seed: int = 42,
        checkpoint_dir: Optional[str] = None,
        pso_c1: float = 2.0,
        pso_c2: float = 2.0,
        pso_w_max: float = 0.9,
        pso_w_min: float = 0.4,
    ):
        self.objective_fn = objective_fn
        self.pop_size = pop_size
        self.total_iter = total_iter
        self.ao_fraction = ao_fraction
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir

        # Iteration budget allocation
        self.ao_iter = max(1, int(total_iter * ao_fraction))
        self.pso_iter = total_iter - self.ao_iter

        # PSO hyperparameters (stored for Phase 2 construction)
        self.pso_c1 = pso_c1
        self.pso_c2 = pso_c2
        self.pso_w_max = pso_w_max
        self.pso_w_min = pso_w_min

        # Unified convergence log spanning both phases
        self.convergence = ConvergenceRecord(optimizer_name="Hybrid_AO_PSO")

        logger.info(
            "[Hybrid] Budget: %d total = %d AO (%.0f%%) + %d PSO (%.0f%%)",
            total_iter, self.ao_iter, ao_fraction * 100,
            self.pso_iter, (1 - ao_fraction) * 100,
        )

    def optimize(self) -> Tuple[Dict[str, Any], float, ConvergenceRecord]:
        """
        Run the full Hybrid AO-PSO two-phase optimisation.

        Returns
        -------
        best_params : dict
            Best hyperparameters found (decoded).
        best_fitness : float
            Best validation RMSE achieved.
        convergence : ConvergenceRecord
            Unified convergence log spanning both AO and PSO phases.
        """
        overall_start = time.time()

        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: GLOBAL MAP (Aquila Optimizer)
        # ═══════════════════════════════════════════════════════════════
        logger.info("[Hybrid] ══ PHASE 1: AO Global Exploration ══")

        ao = AquilaOptimizer(
            objective_fn=self.objective_fn,
            pop_size=self.pop_size,
            max_iter=self.ao_iter,
            seed=self.seed,
            checkpoint_dir=self.checkpoint_dir,
        )

        ao_best_params, ao_best_fit, ao_convergence = ao.optimize()

        # Copy AO's convergence history into our unified log
        for record in ao_convergence.history:
            self.convergence.history.append({
                **record,
                "phase": "AO",
            })

        logger.info(
            "[Hybrid] AO phase complete. Best RMSE=%.6f after %d iters",
            ao_best_fit, self.ao_iter,
        )

        # ═══════════════════════════════════════════════════════════════
        # THE HAND-OFF: AO population → PSO swarm
        # ═══════════════════════════════════════════════════════════════
        logger.info(
            "[Hybrid] ══ HAND-OFF: Injecting AO population into PSO ══"
        )

        pso = ParticleSwarm(
            objective_fn=self.objective_fn,
            pop_size=self.pop_size,
            max_iter=self.pso_iter,
            c1=self.pso_c1,
            c2=self.pso_c2,
            w_max=self.pso_w_max,
            w_min=self.pso_w_min,
            seed=self.seed + 1000,  # Different seed for PSO phase
            checkpoint_dir=self.checkpoint_dir,
        )

        # Inject AO's discovered landscape into PSO
        pso._inject_swarm_state(
            positions=ao.population,
            fitness=ao.fitness,
            g_best_pos=ao.global_best_pos,
            g_best_fit=ao.global_best_fit,
        )

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: LOCAL FINE-TUNING (PSO)
        # ═══════════════════════════════════════════════════════════════
        logger.info("[Hybrid] ══ PHASE 2: PSO Local Exploitation ══")

        pso_best_params, pso_best_fit, pso_convergence = pso.optimize()

        # Copy PSO's convergence history into our unified log,
        # adjusting iteration numbers to be sequential after AO.
        for record in pso_convergence.history:
            self.convergence.history.append({
                **record,
                "iteration": record["iteration"] + self.ao_iter,
                "phase": "PSO",
            })

        total_time = time.time() - overall_start

        # ═══════════════════════════════════════════════════════════════
        # RESULTS SUMMARY
        # ═══════════════════════════════════════════════════════════════
        logger.info(
            "[Hybrid] ══ OPTIMIZATION COMPLETE ══\n"
            "  AO  phase: %d iters → best RMSE = %.6f\n"
            "  PSO phase: %d iters → best RMSE = %.6f\n"
            "  Total wall time: %.1f s\n"
            "  Champion params: %s",
            self.ao_iter, ao_best_fit,
            self.pso_iter, pso_best_fit,
            total_time, pso_best_params,
        )

        # ── Checkpoint final state ───────────────────────────────
        if self.checkpoint_dir is not None:
            self._save_final_checkpoint(
                pso_best_params, pso_best_fit, total_time,
            )

        return pso_best_params, pso_best_fit, self.convergence

    def _save_final_checkpoint(
        self,
        best_params: Dict[str, Any],
        best_fitness: float,
        total_time: float,
    ) -> None:
        """Save final Hybrid results to JSON."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(
            self.checkpoint_dir, "hybrid_ao_pso_result.json"
        )
        state = {
            "best_params": best_params,
            "best_fitness": best_fitness,
            "total_time_s": round(total_time, 2),
            "ao_iterations": self.ao_iter,
            "pso_iterations": self.pso_iter,
            "convergence": self.convergence.to_dict(),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("[Hybrid] Final checkpoint saved to %s", path)
