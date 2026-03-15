"""
models.py -- Core Neural Architectures for Energy Load Forecasting
==================================================================

Research: Hybrid AO-PSO Optimised Temporal Fusion Transformer
           for Smart Grid Multi-Horizon Energy Forecasting

This module implements the full model roster required by the testing matrix
defined in MASTER_BLUEPRINT.md:

    Group B (Forecasting Accuracy):
        StandardLSTM      -- Vanilla LSTM sequence-to-sequence baseline
        BiLSTM            -- Bidirectional LSTM with fwd/bwd state fusion
        XGBoostBaseline   -- Gradient-boosted tree baseline (tabular adapter)
        BaseTFT           -- Temporal Fusion Transformer (Lim et al., 2021)

All deep learning models inherit from ``torch.nn.Module``, accept
sliding-window tensors of shape ``(B, T, F)`` as encoder input, and
produce multi-step point forecasts of shape ``(B, H)`` where H is the
prediction horizon.

References
----------
[1] Lim, B., Arik, S.O., Loeff, N. & Pfister, T. (2021). "Temporal
    Fusion Transformers for Interpretable Multi-horizon Time Series
    Forecasting." International Journal of Forecasting, 37(4),
    pp. 1748--1764.
[2] Dauphin, Y.N., Fan, A., Auli, M. & Grangier, D. (2017).
    "Language Modeling with Gated Convolutional Networks." ICML.
[3] Abualigah, L. et al. (2021). "Aquila Optimizer: A novel
    meta-heuristic optimization Algorithm." Computers & Industrial
    Engineering, 157, 107250.
"""

import math
import numpy as np

import torch
import torch.nn as nn

try:
    import xgboost as xgb
    from sklearn.multioutput import MultiOutputRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

__all__ = [
    "StandardLSTM",
    "BiLSTM",
    "XGBoostBaseline",
    "GatedLinearUnit",
    "GatedResidualNetwork",
    "VariableSelectionNetwork",
    "BaseTFT",
]


# =====================================================================
# SECTION 1: LSTM BASELINES
# =====================================================================

class StandardLSTM(nn.Module):
    """
    Vanilla unidirectional LSTM for multi-step time-series forecasting.

    Architecture
    ------------
    Input (B, T, F)
      -> Stacked LSTM (num_layers deep)
      -> Take last time-step output h_T
      -> Dropout
      -> Fully-connected projection
      -> Output (B, H)

    The stacked LSTM encodes the input window into a fixed-size hidden
    representation h_T at the final time step.  A fully-connected layer
    then maps h_T to the full forecast horizon in a single shot (direct
    multi-step forecasting strategy).

    The LSTM cell at each layer l and time step t computes:

        f_t = sigma(W_f * [h_{t-1}, x_t] + b_f)      (forget gate)
        i_t = sigma(W_i * [h_{t-1}, x_t] + b_i)      (input gate)
        c~_t = tanh(W_c * [h_{t-1}, x_t] + b_c)      (candidate)
        c_t = f_t . c_{t-1} + i_t . c~_t              (cell state)
        o_t = sigma(W_o * [h_{t-1}, x_t] + b_o)       (output gate)
        h_t = o_t . tanh(c_t)                          (hidden state)

    Parameters
    ----------
    n_features : int
        Number of input features per time step (F).
    hidden_size : int
        Dimensionality of each LSTM layer's hidden and cell states.
    num_layers : int
        Depth of the LSTM stack.  Deeper stacks learn increasingly
        abstract temporal representations at the cost of training
        difficulty.
    dropout : float
        Dropout probability applied between LSTM layers (only active
        when ``num_layers > 1``, per PyTorch convention) and before
        the output projection head.
    horizon : int
        Number of future time steps to predict (H).
    """

    def __init__(self, n_features, hidden_size=128, num_layers=2,
                 dropout=0.2, horizon=24):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, T, F)
            Sliding-window input where B = batch size, T = window
            length, F = number of features.

        Returns
        -------
        torch.Tensor, shape (B, H)
            Multi-step point forecast over the prediction horizon.
        """
        # lstm_out: (B, T, hidden_size) -- output at every time step
        lstm_out, _ = self.lstm(x)

        # Extract the representation at the final time step only.
        # This is equivalent to h_n[-1] (last layer's final hidden
        # state) but makes the temporal data flow explicit.
        last_step = lstm_out[:, -1, :]          # (B, hidden_size)

        return self.fc(self.dropout(last_step))  # (B, H)


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM for multi-step time-series forecasting.

    Architecture
    ------------
    Input (B, T, F)
      -> Stacked Bi-LSTM (num_layers deep, bidirectional=True)
      -> Concatenate final forward and backward hidden states
      -> Dropout
      -> Fully-connected projection
      -> Output (B, H)

    The bidirectional LSTM processes the input sequence in both temporal
    directions.  The forward pass captures causal dependencies
    (past -> future) while the backward pass captures anti-causal
    patterns (future -> past), providing a richer contextual
    representation of the input window.

    For each direction d in {fwd, bwd}, the LSTM equations apply
    independently.  The final hidden states from both directions of
    the last layer are concatenated:

        h_combined = [h_T^{fwd} ; h_0^{bwd}]   in R^{2 * hidden_size}

    This combined vector is projected to the forecast horizon.

    Parameters
    ----------
    n_features : int
        Number of input features per time step (F).
    hidden_size : int
        Hidden state dimensionality *per direction*.  The concatenated
        representation has dimensionality ``2 * hidden_size``.
    num_layers : int
        Depth of the stacked Bi-LSTM.
    dropout : float
        Dropout between LSTM layers and before the output projection.
    horizon : int
        Forecast horizon length (H).
    """

    def __init__(self, n_features, hidden_size=128, num_layers=2,
                 dropout=0.2, horizon=24):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        # The input to the FC layer is 2 * hidden_size because we
        # concatenate the forward and backward final hidden states.
        self.fc = nn.Linear(hidden_size * 2, horizon)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, T, F)

        Returns
        -------
        torch.Tensor, shape (B, H)
        """
        # h_n shape for bidirectional:
        #   (num_layers * 2, B, hidden_size)
        # The stacking order is:
        #   [layer_0_fwd, layer_0_bwd, layer_1_fwd, layer_1_bwd, ...]
        _, (h_n, _) = self.lstm(x)

        # Extract the final layer's hidden states for both directions:
        #   h_n[-2] = last layer, forward  (processes t=0 -> t=T)
        #   h_n[-1] = last layer, backward (processes t=T -> t=0)
        h_forward = h_n[-2]                                    # (B, hidden_size)
        h_backward = h_n[-1]                                   # (B, hidden_size)

        # Concatenate both directional representations
        combined = torch.cat([h_forward, h_backward], dim=-1)  # (B, 2*hidden_size)

        return self.fc(self.dropout(combined))                  # (B, H)


# =====================================================================
# SECTION 2: XGBOOST BASELINE
# =====================================================================

class XGBoostBaseline:
    """
    Gradient-boosted tree baseline for time-series forecasting.

    This class wraps ``xgboost.XGBRegressor`` inside a scikit-learn
    ``MultiOutputRegressor`` to produce multi-step forecasts, providing
    a fair (identical input data) comparison against the deep learning
    models in the testing matrix.

    XGBoost operates on 2D tabular data, so 3D sliding-window tensors
    ``(B, T, F)`` are flattened to ``(B, T*F)`` via
    :meth:`flatten_windows`.  This preserves the full temporal context
    while conforming to the tabular input requirement.

    The ``MultiOutputRegressor`` trains one independent ``XGBRegressor``
    per horizon step, allowing each step to learn its own feature
    importances and split thresholds.

    Parameters
    ----------
    horizon : int
        Forecast horizon H.  One ``XGBRegressor`` is fitted per step.
    xgb_params : dict
        Additional keyword arguments forwarded to
        ``xgboost.XGBRegressor``.  Common keys include
        ``n_estimators``, ``max_depth``, ``learning_rate``,
        ``subsample``, ``device='cpu'`` to force CPU, etc.
    """

    def __init__(self, horizon=24, **xgb_params):
        if not _HAS_XGB:
            raise ImportError(
                "XGBoost is required for XGBoostBaseline.  "
                "Install via:  pip install xgboost"
            )

        self.horizon = horizon

        # Sensible defaults for hourly energy forecasting regression.
        # Users may override any of these via **xgb_params.
        defaults = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'device': 'cuda',
            'random_state': 42,
        }
        defaults.update(xgb_params)

        self.model = MultiOutputRegressor(
            xgb.XGBRegressor(**defaults), n_jobs=-1
        )
        self._is_fitted = False

    @staticmethod
    def flatten_windows(X):
        """
        Reshape 3D sliding-window arrays into 2D tabular format.

        This is necessary because tree-based models (XGBoost, LightGBM,
        Random Forest) require a flat feature vector per sample, whereas
        deep learning models operate on sequential 3D tensors.

        The flattening concatenates all time steps' features into a
        single row, so a window of shape ``(T, F)`` becomes a vector
        of length ``T * F``.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor, shape (B, T, F)
            Batch of sliding-window inputs.

        Returns
        -------
        np.ndarray, shape (B, T * F)
            Flattened feature matrix.
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if X.ndim == 3:
            B, T, F = X.shape
            return X.reshape(B, T * F)
        return X

    def fit(self, X, y):
        """
        Fit the multi-output XGBoost model.

        Parameters
        ----------
        X : array-like, shape (B, T, F) or (B, T*F)
            Input windows (3D) or pre-flattened features (2D).
        y : array-like, shape (B, H)
            Target values for each horizon step.
        """
        X_flat = self.flatten_windows(X)
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        self.model.fit(X_flat, y)
        self._is_fitted = True

    def predict(self, X):
        """
        Generate multi-step forecasts.

        Parameters
        ----------
        X : array-like, shape (B, T, F) or (B, T*F)

        Returns
        -------
        np.ndarray, shape (B, H)
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.  Call fit() first.")
        X_flat = self.flatten_windows(X)
        return self.model.predict(X_flat)


# =====================================================================
# SECTION 3: TFT BUILDING BLOCKS
# =====================================================================

class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (Dauphin et al., 2017).

    Implements element-wise sigmoid gating:

        GLU(x) = W_1 x  .  sigma(W_2 x)

    where ``.`` denotes the Hadamard (element-wise) product and sigma
    is the sigmoid function.  The first linear projection produces the
    candidate activation and the second produces the gate signal that
    controls information flow.

    This provides a learnable, data-dependent skip mechanism that the
    TFT uses throughout its architecture for adaptive depth.

    Parameters
    ----------
    d_model : int
        Input and output dimensionality.
    """

    def __init__(self, d_model):
        super().__init__()
        self.fc_value = nn.Linear(d_model, d_model)
        self.fc_gate = nn.Linear(d_model, d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (..., d_model)

        Returns
        -------
        Tensor, shape (..., d_model)
        """
        return self.fc_value(x) * self.sigmoid(self.fc_gate(x))


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (Lim et al., 2021, Section 3.3, Eq. 3--4).

    The GRN applies non-linear processing with ELU activation, followed
    by a GLU gate and a residual connection with layer normalisation:

        eta_1 = ELU(W_1 a + W_3 c + b)    if context c is available
        eta_1 = ELU(W_1 a + b)            otherwise
        eta_2 = W_2 eta_1
        GRN(a, c) = LayerNorm(a + GLU(eta_2))

    The optional context vector ``c`` allows external information
    (e.g., static covariates) to modulate the processing of the
    primary input ``a``.

    Parameters
    ----------
    d_model : int
        Primary input and output dimensionality.
    d_context : int or None
        Dimensionality of the optional context vector.  If None,
        no context projection is created.
    dropout : float
        Dropout rate applied after the second linear layer.
    """

    def __init__(self, d_model, d_context=None, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(d_model, d_model)
        self.glu = GatedLinearUnit(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Optional context projection (no bias -- additive modulation)
        self.fc_context = (
            nn.Linear(d_context, d_model, bias=False)
            if d_context is not None else None
        )

    def forward(self, x, context=None):
        """
        Parameters
        ----------
        x : Tensor, shape (..., d_model)
            Primary input.
        context : Tensor or None, shape (..., d_context)
            Optional context vector.  Must be provided if the GRN
            was initialised with ``d_context is not None``.

        Returns
        -------
        Tensor, shape (..., d_model)
        """
        residual = x
        hidden = self.fc1(x)
        if context is not None and self.fc_context is not None:
            hidden = hidden + self.fc_context(context)
        hidden = self.elu(hidden)
        hidden = self.dropout(self.fc2(hidden))
        hidden = self.glu(hidden)
        return self.layer_norm(residual + hidden)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (Lim et al., 2021, Section 3.4, Eq. 5--7).

    The VSN learns per-variable importance weights via a softmax gating
    mechanism, then aggregates individually-processed variable
    representations:

        xi_j^{(l)} = Linear_j(x_j)        (per-variable embedding)
        v_j        = GRN_j(xi_j^{(l)})    (per-variable non-linear processing)
        alpha      = Softmax(GRN_v(Xi))    (variable selection weights)
        output     = sum_j  alpha_j * v_j  (weighted aggregation)

    where ``Xi`` is the concatenation of all variable embeddings and
    ``GRN_v`` may receive a static context vector for dataset-level
    gating.

    Parameters
    ----------
    n_features : int
        Number of input variables to select from.
    d_model : int
        Embedding dimension per variable.
    d_context : int or None
        Static context dimension for the gating GRN.  None = no context.
    dropout : float
        Dropout rate for all internal GRNs.
    """

    def __init__(self, n_features, d_model, d_context=None, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Per-variable projection: scalar -> d_model
        self.var_transforms = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(n_features)]
        )

        # Flattened GRN for computing gating logits (with optional context)
        self.grn_flat = GatedResidualNetwork(
            n_features * d_model, d_context=d_context, dropout=dropout
        )
        self.gate_proj = nn.Linear(n_features * d_model, n_features)

        # Per-variable GRN (no context -- context influences gating only)
        self.var_grns = nn.ModuleList(
            [GatedResidualNetwork(d_model, dropout=dropout)
             for _ in range(n_features)]
        )

    def forward(self, x, context=None):
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, n_features)
            Raw input with one scalar per variable per time step.
        context : Tensor or None, shape (B, d_context)
            Static context vector.  Broadcast across T internally.

        Returns
        -------
        output : Tensor, shape (B, T, d_model)
            Weighted aggregation of per-variable representations.
        weights : Tensor, shape (B, T, n_features)
            Learned variable importance weights (for interpretability).
        """
        B, T, _ = x.shape

        # Step 1: Per-variable embedding  (B, T, 1) -> (B, T, d_model) each
        var_embeds = [
            self.var_transforms[i](x[:, :, i:i + 1])
            for i in range(self.n_features)
        ]

        # Step 2: Concatenate all embeddings for gating
        #   shape: (B, T, n_features * d_model)
        concat = torch.cat(var_embeds, dim=-1)

        # Step 3: Expand static context to temporal dimension
        ctx = None
        if context is not None:
            ctx = context.unsqueeze(1).expand(-1, T, -1)  # (B, T, d_context)

        # Step 4: Compute gating weights via GRN + softmax
        gate_input = self.grn_flat(concat, ctx)
        weights = torch.softmax(
            self.gate_proj(gate_input), dim=-1
        )  # (B, T, n_features)

        # Step 5: Weighted aggregation of per-variable GRN outputs
        weighted_sum = torch.zeros(B, T, self.d_model, device=x.device)
        for i in range(self.n_features):
            processed = self.var_grns[i](var_embeds[i])      # (B, T, d_model)
            weighted_sum += weights[:, :, i:i + 1] * processed

        return weighted_sum, weights


# =====================================================================
# SECTION 4: TEMPORAL FUSION TRANSFORMER
# =====================================================================

class BaseTFT(nn.Module):
    """
    Temporal Fusion Transformer (Lim et al., 2021).

    A full custom PyTorch implementation of the TFT with explicit
    support for three categories of input variables:

        1. **Observed (encoder-only):**  Dynamic features available only
           in the past window (e.g., ``System_Load``,
           ``weighted_temperature``, ``weighted_solar``).
        2. **Known future (encoder + decoder):**  Covariates available
           for both past and future time steps (e.g., ``hour_sin``,
           ``hour_cos``, ``is_holiday``, ``day_of_week``).
        3. **Static categorical:**  Time-invariant discrete features
           (e.g., ``grid_id``) embedded via learnable look-up tables.

    Architecture Overview
    ---------------------
    ::

        Static categoricals (optional)
          -> Embedding -> GRN x4 -> context vectors (c_s, c_e, c_h, c_c)

        Encoder input (past observed + known)
          -> Variable Selection Network (context = c_s)
          -> LSTM Encoder (init = c_h, c_c)
          -> Gated skip + LayerNorm

        Decoder input (future known, optional)
          -> Variable Selection Network (context = c_s)
          -> LSTM Decoder (init = encoder final state)
          -> Gated skip + LayerNorm

        Concatenate [encoder ; decoder] temporal representations
          -> Static Enrichment GRN (context = c_e)
          -> Positional Encoding (sinusoidal)
          -> Multi-Head Self-Attention (causal mask in decoder mode)
          -> Gated skip + LayerNorm
          -> Position-wise GRN (feed-forward)
          -> Gated skip + LayerNorm
          -> Output Projection -> (B, H)

    Operating Modes
    ---------------
    **Encoder-only mode** (``n_decoder_features=0``):
        All features are passed as ``x_encoder``.  The output is
        produced by flattening the encoder representations and
        projecting to the horizon.  This is equivalent to the
        simplified TFT used in the V5 notebook and is compatible
        with the Aquila Optimizer's fitness evaluation.

    **Encoder-decoder mode** (``n_decoder_features > 0``):
        Known future covariates are passed separately as ``x_decoder``
        with shape ``(B, H, n_decoder_features)`` where H = horizon.
        The decoder LSTM is initialised from the encoder's final
        hidden state, and a causal attention mask prevents decoder
        positions from attending to future decoder positions.

    Parameters
    ----------
    n_encoder_features : int
        Total features in the encoder input (observed + known past).
    n_decoder_features : int
        Features in the decoder input (known future covariates).
        Set to 0 for encoder-only mode.
    n_static_categoricals : int
        Number of static categorical variables.  Set to 0 if none.
    static_vocab_sizes : list[int] or None
        Vocabulary size for each static categorical embedding.
        Required when ``n_static_categoricals > 0``.
    static_embedding_dim : int
        Embedding dimension per static categorical variable.
    d_model : int
        Model-wide hidden dimensionality.
    n_heads : int
        Number of attention heads.  Must divide ``d_model`` evenly.
    num_encoder_layers : int
        Number of stacked Transformer encoder layers (self-attention
        + feed-forward blocks).
    dropout : float
        Dropout probability used throughout the architecture.
    horizon : int
        Forecast horizon H (= decoder sequence length in decoder mode).
    window_size : int
        Encoder window length T_enc (e.g., 168 for 7 days of hourly data).
    """

    def __init__(
        self,
        n_encoder_features,
        n_decoder_features=0,
        n_static_categoricals=0,
        static_vocab_sizes=None,
        static_embedding_dim=8,
        d_model=64,
        n_heads=4,
        num_encoder_layers=2,
        dropout=0.1,
        horizon=24,
        window_size=168,
    ):
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon
        self.window_size = window_size
        self.has_decoder = n_decoder_features > 0
        self.has_static = n_static_categoricals > 0

        # ── 1. Static Covariate Processing ──────────────────────────
        # Each static categorical is embedded and the embeddings are
        # concatenated then projected to d_model.  Four separate GRNs
        # produce context vectors for different parts of the network:
        #   c_selection   -> VSN gating
        #   c_enrichment  -> static enrichment layer
        #   c_state_h     -> LSTM hidden state initialisation
        #   c_state_c     -> LSTM cell state initialisation
        if self.has_static:
            assert static_vocab_sizes is not None, (
                "static_vocab_sizes required when n_static_categoricals > 0"
            )
            assert len(static_vocab_sizes) == n_static_categoricals

            self.static_embeddings = nn.ModuleList([
                nn.Embedding(vocab_size, static_embedding_dim)
                for vocab_size in static_vocab_sizes
            ])
            total_static_dim = n_static_categoricals * static_embedding_dim
            self.static_proj = nn.Linear(total_static_dim, d_model)

            # Four context-generating GRNs (Lim et al. 2021, Eq. 5)
            self.ctx_grn_selection = GatedResidualNetwork(d_model, dropout=dropout)
            self.ctx_grn_enrichment = GatedResidualNetwork(d_model, dropout=dropout)
            self.ctx_grn_state_h = GatedResidualNetwork(d_model, dropout=dropout)
            self.ctx_grn_state_c = GatedResidualNetwork(d_model, dropout=dropout)

            static_ctx_dim = d_model
        else:
            static_ctx_dim = None

        # ── 2. Encoder Variable Selection ───────────────────────────
        self.encoder_vsn = VariableSelectionNetwork(
            n_encoder_features, d_model,
            d_context=static_ctx_dim, dropout=dropout,
        )

        # ── 3. Decoder Variable Selection (if applicable) ──────────
        if self.has_decoder:
            self.decoder_vsn = VariableSelectionNetwork(
                n_decoder_features, d_model,
                d_context=static_ctx_dim, dropout=dropout,
            )

        # ── 4. LSTM Encoder ─────────────────────────────────────────
        # Single-layer LSTM followed by a gated skip connection from
        # the VSN output (pre-LSTM representation).
        self.encoder_lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_model,
            num_layers=1, batch_first=True,
        )
        self.post_enc_gate = GatedLinearUnit(d_model)
        self.post_enc_norm = nn.LayerNorm(d_model)

        # ── 5. LSTM Decoder (if applicable) ─────────────────────────
        # Decoder LSTM is initialised with the encoder's final hidden
        # and cell states, providing temporal continuity.
        if self.has_decoder:
            self.decoder_lstm = nn.LSTM(
                input_size=d_model, hidden_size=d_model,
                num_layers=1, batch_first=True,
            )
            self.post_dec_gate = GatedLinearUnit(d_model)
            self.post_dec_norm = nn.LayerNorm(d_model)

        # ── 6. Static Enrichment ────────────────────────────────────
        # GRN that enriches temporal features with static context.
        # When no statics are available, this acts as an additional
        # non-linear transformation (GRN without context).
        self.enrichment_grn = GatedResidualNetwork(
            d_model,
            d_context=d_model if self.has_static else None,
            dropout=dropout,
        )

        # ── 7. Positional Encoding ──────────────────────────────────
        # Sinusoidal encoding (Vaswani et al., 2017) added to temporal
        # representations before self-attention.
        max_seq_len = window_size + (horizon if self.has_decoder else 0)
        self.register_buffer(
            'pos_encoding',
            self._sinusoidal_encoding(max_seq_len, d_model),
        )

        # ── 8. Temporal Self-Attention ──────────────────────────────
        # Multi-layer Transformer encoder with multi-head attention
        # and position-wise feed-forward networks.
        # enable_nested_tensor was removed in PyTorch 2.10+; only pass
        # it on older versions where it defaults to True and can cause
        # issues with custom attention masks.
        _enc_kwargs = dict(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        import inspect as _inspect
        if "enable_nested_tensor" in _inspect.signature(
            nn.TransformerEncoderLayer.__init__
        ).parameters:
            _enc_kwargs["enable_nested_tensor"] = False
        encoder_layer = nn.TransformerEncoderLayer(**_enc_kwargs)
        _te_kwargs = dict(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
        )
        if "enable_nested_tensor" in _inspect.signature(
            nn.TransformerEncoder.__init__
        ).parameters:
            _te_kwargs["enable_nested_tensor"] = False
        self.transformer = nn.TransformerEncoder(**_te_kwargs)

        # Post-attention GRN with gated skip from pre-transformer
        self.post_attn_grn = GatedResidualNetwork(d_model, dropout=dropout)
        self.post_attn_norm = nn.LayerNorm(d_model)

        # ── 9. Output Projection ────────────────────────────────────
        if self.has_decoder:
            # In decoder mode each of the H decoder positions produces
            # one forecast value via a shared linear layer.
            self.output_proj = nn.Linear(d_model, 1)
        else:
            # In encoder-only mode the full temporal representation is
            # flattened and projected to the horizon.
            self.output_proj = nn.Linear(d_model * window_size, horizon)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _sinusoidal_encoding(max_len, d_model):
        """
        Generate fixed sinusoidal positional encodings
        (Vaswani et al., 2017).

        PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
        PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

        Returns
        -------
        Tensor, shape (1, max_len, d_model)
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    @staticmethod
    def _make_decoder_mask(n_encoder, n_decoder, device):
        """
        Create an additive attention mask for encoder-decoder mode.

        The mask allows:
          - Encoder positions to attend to ALL encoder positions
            (no restriction -- all past observations are known).
          - Decoder positions to attend to ALL encoder positions
            and to preceding (or same) decoder positions only.

        Blocked positions are filled with ``-inf`` (additive mask
        convention used by ``nn.TransformerEncoder``).

        Parameters
        ----------
        n_encoder : int
            Number of encoder (past) time steps.
        n_decoder : int
            Number of decoder (future) time steps.
        device : torch.device

        Returns
        -------
        Tensor, shape (n_encoder + n_decoder, n_encoder + n_decoder)
        """
        total = n_encoder + n_decoder
        # Start with no masking (all attend to all)
        mask = torch.zeros(total, total, device=device)

        # Apply causal (upper-triangular) masking to the
        # decoder-decoder block only.  This prevents decoder
        # position k from attending to decoder positions > k,
        # while leaving encoder-encoder and decoder-encoder
        # attention unrestricted.
        if n_decoder > 0:
            causal_block = torch.triu(
                torch.full((n_decoder, n_decoder), float('-inf'),
                           device=device),
                diagonal=1,
            )
            mask[n_encoder:, n_encoder:] = causal_block

        return mask

    def _get_static_contexts(self, s_cat):
        """
        Process static categorical inputs into four context vectors.

        Parameters
        ----------
        s_cat : LongTensor, shape (B, n_static_categoricals)
            Integer indices for each categorical variable.

        Returns
        -------
        c_selection, c_enrichment, c_h, c_c : each Tensor (B, d_model)
        """
        # Embed each categorical and concatenate
        embedded = [
            self.static_embeddings[i](s_cat[:, i])
            for i in range(s_cat.size(1))
        ]
        s = self.static_proj(torch.cat(embedded, dim=-1))  # (B, d_model)

        # Generate four independent context vectors via GRNs
        c_selection = self.ctx_grn_selection(s)
        c_enrichment = self.ctx_grn_enrichment(s)
        c_h = self.ctx_grn_state_h(s)
        c_c = self.ctx_grn_state_c(s)

        return c_selection, c_enrichment, c_h, c_c

    # -----------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------

    def forward(self, x_encoder, x_decoder=None, s_cat=None):
        """
        Forward pass of the Temporal Fusion Transformer.

        Parameters
        ----------
        x_encoder : Tensor, shape (B, T_enc, n_encoder_features)
            Past window containing all observed and known-past features.
        x_decoder : Tensor or None, shape (B, H, n_decoder_features)
            Known future covariates for each forecast horizon step.
            Required when the model was initialised with
            ``n_decoder_features > 0``.  Pass ``None`` for
            encoder-only mode.
        s_cat : LongTensor or None, shape (B, n_static_categoricals)
            Static categorical variable indices.  Required when the
            model was initialised with ``n_static_categoricals > 0``.

        Returns
        -------
        forecast : Tensor, shape (B, H)
            Point forecast for each horizon step.
        """
        B = x_encoder.size(0)

        # ── 1. Static context vectors ───────────────────────────────
        c_sel = c_enr = c_h = c_c = None
        if self.has_static and s_cat is not None:
            c_sel, c_enr, c_h, c_c = self._get_static_contexts(s_cat)

        # ── 2. Encoder Variable Selection ───────────────────────────
        enc_selected, _enc_weights = self.encoder_vsn(
            x_encoder, context=c_sel,
        )  # (B, T_enc, d_model)

        # ── 3. Encoder LSTM ─────────────────────────────────────────
        if c_h is not None:
            # Initialise LSTM hidden/cell states from static context
            h0 = c_h.unsqueeze(0)  # (1, B, d_model)
            c0 = c_c.unsqueeze(0)  # (1, B, d_model)
            enc_lstm_out, enc_state = self.encoder_lstm(
                enc_selected, (h0, c0),
            )
        else:
            enc_lstm_out, enc_state = self.encoder_lstm(enc_selected)

        # Gated skip connection: skip from VSN output (pre-LSTM)
        enc_temporal = self.post_enc_norm(
            enc_selected + self.post_enc_gate(enc_lstm_out)
        )  # (B, T_enc, d_model)

        # ── 4. Decoder path (if applicable) ─────────────────────────
        if self.has_decoder and x_decoder is not None:
            dec_selected, _ = self.decoder_vsn(
                x_decoder, context=c_sel,
            )  # (B, H, d_model)

            # Decoder LSTM initialised with encoder's final state
            dec_lstm_out, _ = self.decoder_lstm(dec_selected, enc_state)

            # Gated skip connection
            dec_temporal = self.post_dec_norm(
                dec_selected + self.post_dec_gate(dec_lstm_out)
            )  # (B, H, d_model)

            # Concatenate encoder and decoder temporal representations
            temporal = torch.cat(
                [enc_temporal, dec_temporal], dim=1,
            )  # (B, T_enc + H, d_model)
        else:
            temporal = enc_temporal  # (B, T_enc, d_model)

        # ── 5. Static Enrichment ────────────────────────────────────
        if c_enr is not None:
            c_enr_expanded = c_enr.unsqueeze(1).expand(
                -1, temporal.size(1), -1,
            )  # (B, T, d_model)
            temporal = self.enrichment_grn(temporal, context=c_enr_expanded)
        else:
            temporal = self.enrichment_grn(temporal)

        # ── 6. Positional Encoding ──────────────────────────────────
        temporal = temporal + self.pos_encoding[:, :temporal.size(1), :]

        # Save pre-transformer representation for gated skip connection
        pre_attn = temporal

        # ── 7. Temporal Self-Attention ──────────────────────────────
        if self.has_decoder and x_decoder is not None:
            # Causal mask: decoder positions cannot attend to future
            # decoder positions, but can attend to all encoder positions.
            mask = self._make_decoder_mask(
                self.window_size, self.horizon, temporal.device,
            )
            temporal = self.transformer(temporal, mask=mask)
        else:
            # Encoder-only: no masking needed (all positions are past)
            temporal = self.transformer(temporal)

        # Post-attention GRN with skip from pre-transformer input.
        # This provides a coarse-grained residual around the entire
        # Transformer stack, complementing the fine-grained residuals
        # within each TransformerEncoderLayer.
        temporal = self.post_attn_norm(
            pre_attn + self.post_attn_grn(temporal)
        )  # (B, T, d_model)

        # ── 8. Output Projection ────────────────────────────────────
        if self.has_decoder and x_decoder is not None:
            # Take only the decoder positions (last H steps)
            dec_output = temporal[:, -self.horizon:, :]  # (B, H, d_model)
            forecast = self.output_proj(dec_output).squeeze(-1)  # (B, H)
        else:
            # Encoder-only: flatten and project
            flat = temporal.reshape(B, -1)  # (B, T_enc * d_model)
            forecast = self.output_proj(flat)  # (B, H)

        return forecast
