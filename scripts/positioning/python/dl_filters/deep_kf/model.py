# -*- coding: utf-8 -*-
"""
Deep Kalman Filter — State Predictor LSTM + State Decoder.

Architecture
------------
From Hosseinyalamdary, MDPI Sensors 18(5):1316, 2018 — Eqs. 18-21, Figure 3-4.

The LSTM receives the posterior navigation state x_t^+ at each step (Eq. 20):
    h_t = φ(x_{t-1}^+, h_{t-1})

and predicts the full 15D state at the next step (Eq. 21):
    x_t^{+-} = decoder(h_t) + x_{t-1}^+     (residual prediction)

State vector: x = [p(3), v(3), θ(3), b_acc(3), b_gyr(3)]  — 15D
    p      : position (ENU)
    v      : velocity (ENU)
    θ      : orientation as Euler angles [roll, pitch, yaw]
    b_acc  : accelerometer bias (FLU body frame)
    b_gyr  : gyroscope bias (FLU body frame)

The DNN output x_t^{+-} replaces the strapdown-predicted prior x_t^- in the
ESKF.  Covariance propagation still uses the linearized Solà F matrix.
"""

import torch
import torch.nn as nn


class StatePredictorLSTM(nn.Module):
    """
    LSTM that models the navigation system dynamics from state history.

    Receives the posterior state x_t^+ (15D) at each step and produces a
    latent vector h_t.  The LSTM hidden state carries temporal context from
    all previous posterior states (Eq. 18-20 in the paper).

    Parameters
    ----------
    nav_state_dim : int — dimension of navigation state (15)
    hidden_dim    : int — LSTM hidden size
    num_layers    : int — LSTM depth
    dropout       : float — dropout between LSTM layers (0 if num_layers == 1)
    """

    def __init__(
        self,
        nav_state_dim: int = 15,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=nav_state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.,
        )

    def forward(self, nav_state: torch.Tensor, hidden=None):
        """
        Parameters
        ----------
        nav_state : (B, nav_state_dim) — posterior navigation state x_t^+
        hidden    : optional tuple (h_0, c_0) LSTM hidden state

        Returns
        -------
        h_out  : (B, hidden_dim) — latent output (last time step)
        hidden : (h_n, c_n)     — new LSTM hidden state (pass to next step)
        """
        x = nav_state.unsqueeze(1)                  # (B, 1, 15) — single step
        out, hidden_new = self.lstm(x, hidden)      # out: (B, 1, hidden_dim)
        h_out = out[:, 0, :]                        # (B, hidden_dim)
        return h_out, hidden_new


class StateDecoder(nn.Module):
    """
    Decode LSTM latent vector → 15D state increment (residual prediction).

    The decoder outputs δx = x_t^{+-} - x_{t-1}^+, i.e. the predicted change
    in state.  The caller adds this residual to x_{t-1}^+ to get x_t^{+-}.
    This matches the paper's Eq. 21:  x_t^{+-} = σ(W_xx · h_t) + μ_t
    where μ_t = x_{t-1}^+ (the previous posterior).

    A two-layer MLP with wider intermediate layer to handle the 15D output.
    Final layer is linear (no activation) for unbounded state predictions.
    """

    def __init__(self, hidden_dim: int = 128, out_dim: int = 15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : (B, hidden_dim)

        Returns
        -------
        delta_state : (B, 15) — predicted state increment [δp, δv, δθ, δb_a, δb_g]
        """
        return self.net(h)


class DeepKFNet(nn.Module):
    """
    Combined LSTM + State Decoder module (Eqs. 18-21, Figure 3).

    Predicts the full 15D navigation state using residual prediction:
        x_t^{+-} = decoder(LSTM(x_{t-1}^+)) + x_{t-1}^+
    """

    def __init__(
        self,
        nav_state_dim: int = 15,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm    = StatePredictorLSTM(nav_state_dim, hidden_dim,
                                          num_layers, dropout)
        self.decoder = StateDecoder(hidden_dim, out_dim=nav_state_dim)
        self.hidden_dim = hidden_dim

    def forward(self, nav_state: torch.Tensor, hidden=None):
        """
        Parameters
        ----------
        nav_state : (B, 15) — posterior state x_{t-1}^+
        hidden    : optional LSTM hidden state

        Returns
        -------
        state_pred : (B, 15) — predicted state x_t^{+-} (residual added)
        hidden     : new LSTM hidden state
        """
        h, hidden_new = self.lstm(nav_state, hidden)
        delta = self.decoder(h)
        state_pred = nav_state + delta    # residual connection (Eq. 21, μ_t = x_{t-1}^+)
        return state_pred, hidden_new

    def init_hidden(self, batch_size: int = 1, device: str = 'cpu'):
        """Return zero initial hidden state (h_0, c_0)."""
        h0 = torch.zeros(self.lstm.num_layers, batch_size,
                         self.hidden_dim, device=device)
        c0 = torch.zeros_like(h0)
        return h0, c0
