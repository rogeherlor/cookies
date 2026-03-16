# -*- coding: utf-8 -*-
"""
Deep KF IMU Error LSTM and Bias Decoder.

Architecture
------------
From Hosseinyalamdary, MDPI Sensors 18(5):1316, 2018 — Figure 3 / Section 3.

The LSTM receives at each step:
  - the current navigation state x_post (15-vector: [p,v,φ,b_a,b_g] errors)
  - the raw FLU IMU window u_t  (T × 6: [accel_flu | gyro_flu])

It outputs a latent vector h_t whose decoder produces IMU bias corrections
[δb_acc(3), δb_gyr(3)] in the FLU body frame. These corrections are applied
to the raw IMU measurements before strapdown propagation.

Channel convention
------------------
IMU input channels: [accel_flu(3) | gyro_flu(3)]   shape (T, 6)
The LSTM input at each step: flatten([x_post, u_t_mean]) → (15 + 6,) = (21,)
(mean over the IMU window so the LSTM step size stays constant regardless of T)
"""

import torch
import torch.nn as nn


class IMUErrorLSTM(nn.Module):
    """
    LSTM that models time-varying IMU bias errors from navigation state history.

    Parameters
    ----------
    nav_state_dim : int — dimension of navigation state (15)
    imu_dim       : int — IMU channels (6: accel_flu + gyro_flu)
    hidden_dim    : int — LSTM hidden size
    num_layers    : int — LSTM depth
    dropout       : float — dropout between LSTM layers (0 if num_layers == 1)
    """

    def __init__(
        self,
        nav_state_dim: int = 15,
        imu_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        input_dim = nav_state_dim + imu_dim   # 21
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.,
        )

    def forward(self, nav_state: torch.Tensor, imu_mean: torch.Tensor,
                hidden=None):
        """
        Parameters
        ----------
        nav_state : (B, nav_state_dim) — current navigation error state
        imu_mean  : (B, imu_dim)       — mean of IMU window (agnostic to window size)
        hidden    : optional tuple (h_0, c_0) LSTM hidden state

        Returns
        -------
        h_out  : (B, hidden_dim) — latent output (last time step)
        hidden : (h_n, c_n)     — new LSTM hidden state (pass to next step)
        """
        x = torch.cat([nav_state, imu_mean], dim=-1)   # (B, 21)
        x = x.unsqueeze(1)                              # (B, 1, 21) — single step
        out, hidden_new = self.lstm(x, hidden)          # out: (B, 1, hidden_dim)
        h_out = out[:, 0, :]                            # (B, hidden_dim)
        return h_out, hidden_new


class BiasDecoder(nn.Module):
    """
    Decode LSTM latent vector → [δb_acc(3), δb_gyr(3)] in FLU body frame.

    A small two-layer MLP keeps the decoder lightweight while allowing
    non-linear mapping from latent space to physical bias units.
    """

    def __init__(self, hidden_dim: int = 128, out_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : (B, hidden_dim)

        Returns
        -------
        bias : (B, 6) — [δb_acc(3), δb_gyr(3)]
        """
        return self.net(h)


class DeepKFNet(nn.Module):
    """
    Combined LSTM + Bias Decoder module.

    Convenience wrapper for saving / loading the full network as one unit.
    """

    def __init__(
        self,
        nav_state_dim: int = 15,
        imu_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm    = IMUErrorLSTM(nav_state_dim, imu_dim, hidden_dim,
                                    num_layers, dropout)
        self.decoder = BiasDecoder(hidden_dim, out_dim=6)
        self.hidden_dim = hidden_dim

    def forward(self, nav_state: torch.Tensor, imu_mean: torch.Tensor,
                hidden=None):
        """
        Returns
        -------
        bias   : (B, 6) — [δb_acc(3), δb_gyr(3)] bias corrections in FLU
        hidden : new LSTM hidden state
        """
        h, hidden_new = self.lstm(nav_state, imu_mean, hidden)
        bias = self.decoder(h)
        return bias, hidden_new

    def init_hidden(self, batch_size: int = 1, device: str = 'cpu'):
        """Return zero initial hidden state (h_0, c_0)."""
        h0 = torch.zeros(self.lstm.num_layers, batch_size,
                         self.hidden_dim, device=device)
        c0 = torch.zeros_like(h0)
        return h0, c0
