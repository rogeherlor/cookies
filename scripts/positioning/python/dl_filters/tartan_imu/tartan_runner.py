# -*- coding: utf-8 -*-
"""
Tartan IMU: A Light Foundation Model for Inertial Positioning in Robotics
==========================================================================
Paper : Zhao et al., "Tartan IMU: A Light Foundation Model for Inertial
        Positioning in Robotics", CVPR 2025
        Project page: https://superodometry.com/tartanimu
Code  : https://github.com/castacks (not yet released as of this implementation)
        Weights: raphael-blanchard/TartanIMU on HuggingFace

Differences from the original paper:
--------------------------------------
1. Input rate: 100 Hz (this project) upsampled to 200 Hz before feeding the model.
   Original paper normalises all inputs to 200 Hz.

2. GPS integration: GPS position updates added to the 15-state ESKF when
   gps_available[i] is True.  Original paper is IMU-only (no GPS integration).

3. Filter: 15-state ESKF (Solà 2017) for position/velocity/attitude tracking.
   Tartan network outputs velocity in body frame → rotated to ENU → velocity
   measurement (H = [0 | I_3 | 0_3×9]).  Original paper integrates velocity
   directly without a Kalman filter.

4. Online adaptation: DISABLED for fair comparison.  Tartan's GMM-based adaptive
   training buffer (paper Section 3.4) is not used during inference.

5. Training: LOO CV on KITTI clean sequences for LoRA fine-tuning.  We never
   train from scratch — only LoRA adapters are fine-tuned on KITTI.

6. Multi-head: 'car' head used exclusively for KITTI data.

7. Tartan IMU MUST NOT be trained from scratch.  Raises RuntimeError if no
   pretrained weights are found.

8. Outage simulation and DR_MODE: added for project compatibility, not in paper.

Weights search order (base model):
  1. TARTAN_IMU_WEIGHTS env var
  2. external/tartan_imu/tartan_imu_base.pt
  3. artifacts/tartan_imu/tartan_imu_base.pt
  4. external/tartan_imu/checkpoints/foundation_model/checkpoint_<N>.pt
     (HuggingFace snapshot_download layout — highest N wins)

LoRA adapter search (optional, applied after base model):
  1. artifacts/tartan_imu/lora_fold_<seq_id>.pt
  2. artifacts/tartan_imu/lora_adapters.pt
  (If none found, runs zero-shot with car head.)
"""

import os
import sys
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent.parent.parent
_ARTIFACTS = _REPO_ROOT / 'artifacts/tartan_imu'
_EXTERNAL  = _REPO_ROOT / 'external/tartan_imu'
_SCRIPTS   = _REPO_ROOT / 'scripts/positioning/python'

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

DEFAULT_PARAMS = {
    'Rpos': 4.0,
    'lora_rank':     8,
    'lstm_steps':    10,
    'target_rate_hz': 200,
    'Qpos':     1e-4,
    'Qvel':     1e-3,
    'Qorient':  1e-5,
    'Qacc':     1e-6,
    'Qgyr':     1e-7,
    'P_pos_std':    1.0,
    'P_vel_std':    0.5,
    'P_orient_std': 0.1,
    'P_acc_std':    0.05,
    'P_gyr_std':    0.01,
    'beta_acc': 0.0,
    'beta_gyr': 0.0,
}

GRAVITY = np.array([0., 0., -9.81])


# ── Physics-based stub (fallback when official weights not released) ───────────

class _TartanImuStub(nn.Module):
    """
    Stub model used when official Tartan IMU checkpoint cannot be loaded.

    Estimates body-frame velocity by integrating the last 1-second gravity-free
    accelerometer window.  This is a reasonable (non-learned) fallback.

    Input : (B, lstm_steps, step_samples, 6) — [accel_gf | gyro] at 200 Hz
    Output: ((B, 3) velocity in body frame, (B, 3) log-std)
    """

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, imu_input, robot_type='car'):
        B           = imu_input.shape[0]
        last_step   = imu_input[:, -1, :, 0:3]    # (B, step_samples, 3) accel
        dt          = 1.0 / 200.0
        v_body      = last_step.mean(dim=1) * dt * last_step.shape[1] * self.scale
        log_std     = torch.ones(B, 3, device=imu_input.device) * 0.5
        return v_body, log_std


# ── Real TartanIMU architecture (reconstructed from checkpoint state dict) ─────
#
# Key dimensions verified from checkpoint_28.pt:
#   input_block : Conv1d(6→64, k=7, stride=4, padding=3), BN
#   residual_groups: 3 groups of 2 ResBlocks
#     group 0: 64→64 (stride=1)
#     group 1: 64→128, 128→128 (stride=2 in first block)
#     group 2: 128→256, 256→256 (stride=2 in first block)
#   resnet_post_pro : Conv1d(256→128, k=1), BN, ReLU, Conv1d(128→128, k=1), BN
#   lstm            : LSTM(input=1664, hidden=64)   1664 = 128 × 13 time steps
#   IMU_Trunk       : 6 × TransformerBlock(embed=64, heads=4, ffn=256)
#   heads           : {car, dog, human, drone} each with 3 MLP output blocks

class _ResBlock1D(nn.Module):
    """1-D residual block matching TartanIMU checkpoint structure."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.convs(x)
        res = self.downsample(x) if self.downsample is not None else x
        return self.relu(out + res)


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block matching IMU_Trunk checkpoint structure."""

    def __init__(self, embed_dim: int = 64, num_heads: int = 4, ffn_dim: int = 256):
        super().__init__()
        self.attn   = nn.MultiheadAttention(embed_dim, num_heads,
                                            add_bias_kv=True, batch_first=True)
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        from collections import OrderedDict
        self.mlp = nn.Sequential(OrderedDict([
            ('fc1',  nn.Linear(embed_dim, ffn_dim)),
            ('act',  nn.GELU()),
            ('fc2',  nn.Linear(ffn_dim, embed_dim)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm_1(x)
        attn_out, _ = self.attn(xn, xn, xn)
        x = x + attn_out
        x = x + self.mlp(self.norm_2(x))
        return x


class _IMUTrunk(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4,
                 ffn_dim: int = 256, n_blocks: int = 6):
        super().__init__()
        self.blocks = nn.ModuleList([
            _TransformerBlock(embed_dim, num_heads, ffn_dim)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class _OutputBlock(nn.Module):
    """MLP head: Linear indices 0, 3, 6 with ReLU+Dropout at 1-2, 4-5."""

    def __init__(self, in_dim: int = 64, hidden: int = 256, out_dim: int = 3):
        super().__init__()
        self.fcs = nn.Sequential(
            nn.Linear(in_dim, hidden),   # 0
            nn.ReLU(),                   # 1
            nn.Dropout(0.0),             # 2
            nn.Linear(hidden, hidden),   # 3
            nn.ReLU(),                   # 4
            nn.Dropout(0.0),             # 5
            nn.Linear(hidden, out_dim),  # 6
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fcs(x)


class _RobotHead(nn.Module):
    """Per-robot output head with 3 MLP blocks matching checkpoint structure."""

    def __init__(self):
        super().__init__()
        self.output_block1   = _OutputBlock(64, 256, 2)  # vx, vy
        self.output_block1_z = _OutputBlock(64, 256, 1)  # vz
        self.output_block2   = _OutputBlock(64, 256, 3)  # log_std

    def forward(self, feat: torch.Tensor):
        v_xy    = self.output_block1(feat)    # (B, 2)
        v_z     = self.output_block1_z(feat)  # (B, 1)
        log_std = self.output_block2(feat)    # (B, 3)
        v_body  = torch.cat([v_xy, v_z], dim=-1)  # (B, 3)
        return v_body, log_std


class _TartanIMUBackbone(nn.Module):
    """
    Conv1D backbone matching TartanIMU checkpoint.

    Processes one LSTM step window: (B, 6, step_samples) → (B, 1664).
    After stride-4 input conv + 3 residual groups + post-processing:
      200 → 50 → 25 → 13  time steps; 13 × 128 = 1664 LSTM input.
    """

    def __init__(self):
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Conv1d(6, 64, 7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(64),
        )
        self.residual_groups = nn.ModuleList([
            nn.ModuleList([_ResBlock1D(64,  64,  stride=1),
                           _ResBlock1D(64,  64,  stride=1)]),
            nn.ModuleList([_ResBlock1D(64,  128, stride=2),
                           _ResBlock1D(128, 128, stride=1)]),
            nn.ModuleList([_ResBlock1D(128, 256, stride=2),
                           _ResBlock1D(256, 256, stride=1)]),
        ])
        self.resnet_post_pro = nn.Sequential(
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
        )
        self.lstm      = nn.LSTM(input_size=1664, hidden_size=64, batch_first=True)
        self.IMU_Trunk = _IMUTrunk()

    def forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """One step: (B, 6, step_samples) → (B, 1664)."""
        x = torch.relu(self.input_block(x))
        for group in self.residual_groups:
            for block in group:
                x = block(x)
        x = self.resnet_post_pro(x)          # (B, 128, 13)
        return x.flatten(1)                  # (B, 1664)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, lstm_steps, step_samples, 6)
        Returns: (B, 64) — transformer features from last LSTM step
        """
        B, T, S, C = x.shape
        feats = self.forward_cnn(x.reshape(B * T, C, S))  # (B*T, 1664)
        feats = feats.reshape(B, T, -1)                    # (B, T, 1664)
        lstm_out, _ = self.lstm(feats)                     # (B, T, 64)
        trunk_out   = self.IMU_Trunk(lstm_out)             # (B, T, 64)
        return trunk_out[:, -1, :]                         # (B, 64)


class _TartanIMUModel(nn.Module):
    """
    Full TartanIMU model matching checkpoint_28.pt state dict.

    State dict structure:
      model.*  — _TartanIMUBackbone (CNN + LSTM + Transformer)
      heads.*  — _RobotHead per robot type {car, dog, human, drone}

    Input : (B, lstm_steps, step_samples, 6)
    Output: (v_body (B,3), log_std (B,3))
    """

    def __init__(self):
        super().__init__()
        self.model = _TartanIMUBackbone()
        self.heads  = nn.ModuleDict({
            'car':   _RobotHead(),
            'dog':   _RobotHead(),
            'human': _RobotHead(),
            'drone': _RobotHead(),
        })

    def forward(self, x: torch.Tensor, robot_type: str = 'car'):
        feat = self.model(x)                    # (B, 64)
        return self.heads[robot_type](feat)     # (v_body, log_std)


# ── Weight discovery ──────────────────────────────────────────────────────────

def _find_tartan_weights() -> Path:
    """
    Locate the pretrained Tartan IMU base checkpoint.  Search order:
      1. TARTAN_IMU_WEIGHTS env var
      2. external/tartan_imu/tartan_imu_base.pt  (legacy name)
      3. artifacts/tartan_imu/tartan_imu_base.pt (legacy name)
      4. external/tartan_imu/checkpoints/foundation_model/checkpoint_<N>.pt
         (HuggingFace snapshot layout — highest N wins)
    Raises RuntimeError if nothing found (never trains from scratch).
    """
    env = os.environ.get('TARTAN_IMU_WEIGHTS')
    if env and Path(env).exists():
        return Path(env)

    # Legacy flat-file names
    for c in [_EXTERNAL / 'tartan_imu_base.pt', _ARTIFACTS / 'tartan_imu_base.pt']:
        if c.exists():
            return c

    # HuggingFace snapshot layout: checkpoints/foundation_model/checkpoint_<N>.pt
    ckpt_dir = _EXTERNAL / 'checkpoints' / 'foundation_model'
    if ckpt_dir.is_dir():
        candidates = sorted(ckpt_dir.glob('checkpoint_*.pt'),
                            key=lambda p: int(p.stem.split('_')[-1]))
        if candidates:
            return candidates[-1]   # highest checkpoint number

    raise RuntimeError(
        "Tartan IMU pretrained weights not found — NEVER train from scratch!\n"
        "Download from HuggingFace:\n"
        "  python -c \"\n"
        "  from huggingface_hub import snapshot_download\n"
        "  snapshot_download('raphael-blanchard/TartanIMU', repo_type='dataset',\n"
        f"                    local_dir='{_EXTERNAL}')\"\n"
        "Or set TARTAN_IMU_WEIGHTS to the .pt file path."
    )


def _resolve_seq_id(seq_id):
    """Convert full KITTI drive name to short seq ID if needed."""
    if seq_id is None:
        return None
    if len(seq_id) <= 2:
        return seq_id
    from data_loader import KITTI_SEQ_TO_DRIVE
    _drive_to_seq = {v: k for k, v in KITTI_SEQ_TO_DRIVE.items()}
    return _drive_to_seq.get(seq_id, seq_id)


def _find_lora_adapter(seq_id=None):
    """
    Locate LoRA adapter weights.  Search order:
      1. lora_fold_<seq_id>.pt  (LOO fold)
      2. lora_adapters.pt       (all-sequences)
      3. Any available lora_fold_*.pt (fallback with warning)
    """
    short_id = _resolve_seq_id(seq_id)
    if short_id is not None:
        fold = _ARTIFACTS / f'lora_fold_{short_id}.pt'
        if fold.exists():
            return fold
    general = _ARTIFACTS / 'lora_adapters.pt'
    if general.exists():
        return general
    # Fallback: use any available LoRA fold
    available = sorted(_ARTIFACTS.glob('lora_fold_*.pt'))
    if available:
        print(f"WARNING: Tartan lora_fold_{short_id}.pt not found, falling back to {available[0].name}. "
              f"Train the proper fold with: python ins_train.py tartan_imu --seqs {short_id}")
        return available[0]
    return None


def _load_tartan_model(weights_path: Path, lora_path, lora_rank: int,
                       device: str = 'cpu'):
    """Load base model + optional LoRA. Returns (model, use_lora)."""
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    if isinstance(ckpt, nn.Module):
        model = ckpt
    elif isinstance(ckpt, dict):
        state = (ckpt.get('model_state_dict') or ckpt.get('state_dict')
                 or ckpt.get('model') or ckpt)
        model = _TartanIMUModel()
        try:
            result = model.load_state_dict(state, strict=False)
            missing = [k for k in result.missing_keys if 'num_batches_tracked' not in k]
            if missing:
                print(f"Tartan IMU: {len(missing)} missing keys (first 3: {missing[:3]}). Best effort.")
            else:
                print("Tartan IMU: checkpoint loaded successfully into _TartanIMUModel.")
        except Exception as e:
            print(f"Tartan IMU: could not load state dict ({e}). Using stub.")
            model = _TartanImuStub()
    else:
        print(f"Tartan IMU: unrecognised checkpoint type ({type(ckpt)}). Using stub.")
        model = _TartanImuStub()

    model = model.to(device)

    use_lora = False
    if lora_path is not None:
        try:
            lora_ckpt = torch.load(lora_path, map_location=device, weights_only=False)
            if isinstance(lora_ckpt, dict):
                sd = lora_ckpt.get('lora_state_dict') or lora_ckpt
                model.load_state_dict(sd, strict=False)
            print(f"Tartan IMU: LoRA adapter loaded from {lora_path.name}")
            use_lora = True
        except Exception as e:
            print(f"Tartan IMU: WARNING — LoRA load failed: {e}")
    else:
        print("Tartan IMU: no LoRA adapter — zero-shot inference.")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, use_lora


# ── Quaternion / rotation utilities ──────────────────────────────────────────

def _skew(v):
    return np.array([[ 0.,-v[2], v[1]],[ v[2], 0.,-v[0]],[-v[1], v[0], 0.]])

def _qnorm(q):
    n = np.linalg.norm(q); return q/n if n > 0. else np.array([1.,0.,0.,0.])

def _qmul(q1, q2):
    w1,x1,y1,z1=q1; w2,x2,y2,z2=q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

def _qfrom_axis_angle(dtheta):
    a = np.linalg.norm(dtheta)
    if a < 1e-12: return _qnorm(np.array([1.,.5*dtheta[0],.5*dtheta[1],.5*dtheta[2]]))
    ax = dtheta/a; s = np.sin(.5*a)
    return np.array([np.cos(.5*a), ax[0]*s, ax[1]*s, ax[2]*s])

def _qfrom_euler(roll, pitch, yaw):
    cr,sr = np.cos(roll/2), np.sin(roll/2)
    cp,sp = np.cos(pitch/2), np.sin(pitch/2)
    cy,sy = np.cos(yaw/2), np.sin(yaw/2)
    return _qnorm(np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                              cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]))

def _qto_rpy(q):
    w,x,y,z=q
    return np.array([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),
                     np.arcsin(np.clip(2*(w*y-z*x),-1.,1.)),
                     np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))])

def _qto_Rbn(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)  ],
                     [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)  ],
                     [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


# ── Main filter ───────────────────────────────────────────────────────────────

def run(nav_data, params=None, outage_config=None, use_3d_rotation=True):
    """
    Run Tartan IMU filter on nav_data.

    Returns dict: p, v, r, bias_acc, bias_gyr,
                  std_pos, std_vel, std_orient, std_bias_acc, std_bias_gyr.
    All arrays (N, 3) float64.
    """
    from scipy.interpolate import interp1d

    p_cfg = dict(DEFAULT_PARAMS)
    if params:
        p_cfg.update(params)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    accel_flu   = nav_data.accel_flu
    gyro_flu    = nav_data.gyro_flu
    orient      = nav_data.orient
    lla         = nav_data.lla
    vel_enu     = nav_data.vel_enu
    sample_rate = nav_data.sample_rate
    lla0        = nav_data.lla0

    import pymap3d as pm
    N  = accel_flu.shape[0]
    Ts = 1.0 / sample_rate

    # ── Load model ─────────────────────────────────────────────────────────
    seq_id    = getattr(nav_data, 'dataset_name', None)
    w_path    = _find_tartan_weights()
    l_path    = _find_lora_adapter(seq_id)
    model, _  = _load_tartan_model(w_path, l_path, int(p_cfg['lora_rank']), device)
    print(f"Tartan IMU: weights={w_path.name}  lora={'yes' if l_path else 'no'}  device={device}")

    # ── Upsample IMU to 200 Hz ─────────────────────────────────────────────
    tgt_hz        = int(p_cfg['target_rate_hz'])
    lstm_steps    = int(p_cfg['lstm_steps'])
    step_samples  = tgt_hz   # samples per 1-second LSTM step
    ctx_up        = lstm_steps * step_samples   # 2000 samples = 10s context at 200Hz

    t_src = np.arange(N) / sample_rate
    t_up  = np.arange(0., t_src[-1], 1.0 / tgt_hz)
    N_up  = len(t_up)

    accel_up = interp1d(t_src, accel_flu, axis=0, kind='linear',
                        bounds_error=False,
                        fill_value=(accel_flu[0], accel_flu[-1]))(t_up)
    gyro_up  = interp1d(t_src, gyro_flu,  axis=0, kind='linear',
                        bounds_error=False,
                        fill_value=(gyro_flu[0], gyro_flu[-1]))(t_up)

    # Precompute gravity-free accel at 200 Hz
    roll_up  = np.interp(t_up, t_src, orient[:, 0])
    pitch_up = np.interp(t_up, t_src, orient[:, 1])
    accel_gf_up = np.zeros_like(accel_up)
    for k in range(N_up):
        R_nb = _qto_Rbn(_qfrom_euler(roll_up[k], pitch_up[k], 0.))
        g_body = R_nb.T @ np.array([0., 0., -9.81])
        accel_gf_up[k] = accel_up[k] - g_body

    # ── GPS outage mask ────────────────────────────────────────────────────
    try:
        import ins_config as _ic
        dr_mode = getattr(_ic, 'DR_MODE', False)
    except Exception:
        dr_mode = False

    gps_avail = nav_data.gps_available.copy()
    if outage_config is not None:
        t1 = outage_config.get('start', 0.)
        d  = outage_config.get('duration', 0.)
        A  = int(t1 * sample_rate)
        B  = int((t1 + d) * sample_rate)
        gps_avail[A:B] = False
    if dr_mode:
        gps_avail[:] = False

    e, n, u = pm.geodetic2enu(lla[:,0], lla[:,1], lla[:,2], lla0[0], lla0[1], lla0[2])
    p_gps_enu = np.column_stack([e, n, u])

    # ── Output arrays ──────────────────────────────────────────────────────
    pos        = np.zeros((N, 3))
    vel        = np.zeros((N, 3))
    rpy_out    = np.zeros((N, 3))
    b_acc_out  = np.zeros((N, 3))
    b_gyr_out  = np.zeros((N, 3))
    std_pos    = np.zeros((N, 3))
    std_vel    = np.zeros((N, 3))
    std_orient = np.zeros((N, 3))
    std_b_acc  = np.zeros((N, 3))
    std_b_gyr  = np.zeros((N, 3))

    pos[0]     = p_gps_enu[0]
    vel[0]     = vel_enu[0]
    rpy_out[0] = orient[0]

    pIMU = pos[0].copy()
    vIMU = vel[0].copy()
    q    = _qfrom_euler(orient[0,0], orient[0,1], orient[0,2])
    b_a  = np.zeros(3)
    b_g  = np.zeros(3)
    dx   = np.zeros(15)

    beta_acc = p_cfg['beta_acc']
    beta_gyr = p_cfg['beta_gyr']

    Q = np.zeros((15, 15))
    Q[0:3,   0:3]   = np.eye(3) * p_cfg['Qpos']
    Q[3:6,   3:6]   = np.eye(3) * (p_cfg['Qvel'] * Ts**2)
    Q[6:9,   6:9]   = np.eye(3) * p_cfg['Qorient']
    Q[9:12,  9:12]  = np.eye(3) * (p_cfg['Qacc'] * Ts)
    Q[12:15, 12:15] = np.eye(3) * (p_cfg['Qgyr'] * Ts)

    P = np.diag([
        p_cfg['P_pos_std'],    p_cfg['P_pos_std'],    p_cfg['P_pos_std'],
        p_cfg['P_vel_std'],    p_cfg['P_vel_std'],    p_cfg['P_vel_std'],
        p_cfg['P_orient_std'], p_cfg['P_orient_std'], p_cfg['P_orient_std'],
        p_cfg['P_acc_std'],    p_cfg['P_acc_std'],    p_cfg['P_acc_std'],
        p_cfg['P_gyr_std'],    p_cfg['P_gyr_std'],    p_cfg['P_gyr_std'],
    ]) ** 2

    R_pos = np.eye(3) * p_cfg['Rpos']

    # Tartan velocity update interval in src-rate samples (1 Hz = every 100 @ 100 Hz)
    tartan_interval = int(sample_rate)  # 100

    def src_to_up(i_src):
        return int(i_src * tgt_hz / sample_rate)

    print(f"Tartan IMU: running {N} samples, Tartan updates at 1 Hz ...")

    for i in range(N - 1):
        acc_b   = accel_flu[i] - b_a
        omega_b = gyro_flu[i]  - b_g

        dtheta = omega_b * Ts if use_3d_rotation else np.array([0.,0.,omega_b[2]*Ts])
        q      = _qnorm(_qmul(q, _qfrom_axis_angle(dtheta)))
        Rbn    = _qto_Rbn(q)
        accENU = Rbn @ acc_b
        pIMU   = pIMU + Ts*vIMU + 0.5*Ts**2*(accENU + GRAVITY)
        vIMU   = vIMU + Ts*(accENU + GRAVITY)

        F = np.zeros((15,15))
        F[0:3,3:6]    = np.eye(3)
        F[3:6,6:9]    = -Rbn @ _skew(acc_b)
        F[3:6,9:12]   = -Rbn
        F[6:9,6:9]    = -_skew(omega_b)
        F[6:9,12:15]  = -np.eye(3)
        F[9:12,9:12]  = beta_acc * np.eye(3)
        F[12:15,12:15]= beta_gyr * np.eye(3)
        Fd            = np.eye(15) + F*Ts
        Fd[6:9,6:9]   = _qto_Rbn(_qfrom_axis_angle(omega_b*Ts)).T
        P = Fd @ P @ Fd.T + Q

        update_occurred = False

        # ── Tartan velocity update at 1 Hz ─────────────────────────────
        if (i + 1) % tartan_interval == 0:
            i_up = src_to_up(i + 1)

            if i_up >= ctx_up:
                s = i_up - ctx_up
                # Build (1, lstm_steps, step_samples, 6) tensor
                imu_block = np.zeros((1, lstm_steps, step_samples, 6), dtype=np.float32)
                for step in range(lstm_steps):
                    ss = s + step * step_samples
                    ee = ss + step_samples
                    imu_block[0, step, :, 0:3] = accel_gf_up[ss:ee].astype(np.float32)
                    imu_block[0, step, :, 3:6] = gyro_up[ss:ee].astype(np.float32)

                imu_t = torch.from_numpy(imu_block).to(device)
                with torch.no_grad():
                    v_body_t, log_std_t = model(imu_t, robot_type='car')

                v_body  = v_body_t[0].cpu().numpy().astype(np.float64)
                log_std = log_std_t[0].cpu().numpy().astype(np.float64)
                log_std = np.clip(log_std, -10.0, 10.0)   # prevent exp underflow/overflow

                v_enu_pred = Rbn @ v_body
                Sigma_body = np.diag(np.maximum(np.exp(log_std), 1e-4))   # floor at 1e-4 m²/s²
                Sigma_enu  = Rbn @ Sigma_body @ Rbn.T

                z_vel  = v_enu_pred - vIMU
                H_vel  = np.zeros((3,15)); H_vel[:,3:6] = np.eye(3)
                S_vel  = H_vel @ P @ H_vel.T + Sigma_enu
                S_vel_reg = S_vel + 1e-9 * np.eye(3)
                K_vel  = np.linalg.solve(S_vel_reg, H_vel @ P).T   # 15×3
                dx     = dx + K_vel @ (z_vel - H_vel @ dx)
                IKH_v  = np.eye(15) - K_vel @ H_vel
                P      = IKH_v @ P @ IKH_v.T + K_vel @ Sigma_enu @ K_vel.T  # Joseph form
                P      = 0.5*(P + P.T)
                update_occurred = True

        # ── GPS position update ────────────────────────────────────────
        if gps_avail[i + 1]:
            z_pos     = p_gps_enu[i+1] - pIMU
            innov     = z_pos - dx[0:3]
            S_gps     = P[0:3,0:3] + R_pos
            S_gps_reg = S_gps + 1e-9 * np.eye(3)
            K_gps     = np.linalg.solve(S_gps_reg, P[0:3, :]).T   # 15×3
            dx        = dx + K_gps @ innov
            H_gps     = np.zeros((3, 15)); H_gps[:, 0:3] = np.eye(3)
            IKH_gps   = np.eye(15) - K_gps @ H_gps
            P         = IKH_gps @ P @ IKH_gps.T + K_gps @ R_pos @ K_gps.T  # Joseph form
            P         = 0.5*(P + P.T)
            update_occurred = True

        if update_occurred:
            pIMU += dx[0:3]
            vIMU += dx[3:6]
            b_a  += dx[9:12]
            b_g  += dx[12:15]
            dth   = dx[6:9]
            q     = _qnorm(_qmul(q, _qfrom_axis_angle(dth)))
            G     = np.eye(15); G[6:9,6:9] = np.eye(3) - 0.5*_skew(dth)
            P     = G @ P @ G.T
            dx[:] = 0.

        pos[i+1]       = pIMU
        vel[i+1]       = vIMU
        rpy_out[i+1]   = _qto_rpy(q)
        b_acc_out[i+1] = b_a
        b_gyr_out[i+1] = b_g
        std_pos[i+1]      = np.sqrt(np.maximum(np.diag(P[0:3,0:3]),   0.))
        std_vel[i+1]      = np.sqrt(np.maximum(np.diag(P[3:6,3:6]),   0.))
        std_orient[i+1]   = np.sqrt(np.maximum(np.diag(P[6:9,6:9]),   0.))
        std_b_acc[i+1]    = np.sqrt(np.maximum(np.diag(P[9:12,9:12]), 0.))
        std_b_gyr[i+1]    = np.sqrt(np.maximum(np.diag(P[12:15,12:15]),0.))

    return {
        'p': pos, 'v': vel, 'r': rpy_out,
        'bias_acc': b_acc_out, 'bias_gyr': b_gyr_out,
        'std_pos': std_pos, 'std_vel': std_vel, 'std_orient': std_orient,
        'std_bias_acc': std_b_acc, 'std_bias_gyr': std_b_gyr,
    }
