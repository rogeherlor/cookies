"""
causal_mesnet.py — strictly causal drop-in replacement for AI-IMU's MesNet.

The upstream MesNet (external/ai-imu-dr/src/utils_torch_filter.py) pads with
symmetric ReplicationPad1d, so each covariance output depends on ~8 FUTURE IMU
samples (receptive field ≈ 17, centred).  That cannot run online without latency.

CausalMesNet is the same network (same conv shapes, same head, same submodule
names `cov_net` / `cov_lin` so the AI-IMU optimiser/loop accept it unchanged) but
uses LEFT-only padding placed BEFORE each conv — a standard causal TCN.  Therefore
output[i] depends only on inputs ≤ i.  A key consequence: a causal network can be
evaluated in ONE batch pass over the whole sequence and still be strictly online —
no sliding window, no current-sample-hold approximation.

Train it with `train_ai_imu.py --causal`; run it with `iekf_ai_imu_online.py`.
"""

import torch
import torch.nn as nn


class CausalMesNet(nn.Module):
    """Causal variant of MesNet. Same I/O and submodule names as the original."""

    def __init__(self):
        super().__init__()
        self.beta_measurement = 3 * torch.ones(2).double()
        self.tanh = nn.Tanh()

        # Left-pad BEFORE each conv so no future sample is read.
        #   conv1: kernel 5            → left pad 4
        #   conv2: kernel 5, dilation 3 → left pad (5-1)*3 = 12
        self.cov_net = nn.Sequential(
            nn.ReplicationPad1d((4, 0)),
            nn.Conv1d(6, 32, 5),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.ReplicationPad1d((12, 0)),
            nn.Conv1d(32, 32, 5, dilation=3),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        ).double()

        self.cov_lin = nn.Sequential(
            nn.Linear(32, 2),
            nn.Tanh(),
        ).double()
        self.cov_lin[0].bias.data[:] /= 100
        self.cov_lin[0].weight.data[:] /= 100

    def forward(self, u, iekf):
        # Identical to MesNet.forward; only the padding inside cov_net differs.
        y_cov = self.cov_net(u).transpose(0, 2).squeeze()
        z_cov = self.cov_lin(y_cov)
        z_cov_net = self.beta_measurement.unsqueeze(0) * z_cov
        measurements_covs = (iekf.cov0_measurement.unsqueeze(0) * (10 ** z_cov_net))
        return measurements_covs


# Conv-layer index map: upstream MesNet has the convs at cov_net.0 / cov_net.4
# (conv-then-pad); CausalMesNet has them at cov_net.1 / cov_net.5 (pad-then-conv).
_REMAP = {
    'cov_net.0.weight': 'cov_net.1.weight',
    'cov_net.0.bias':   'cov_net.1.bias',
    'cov_net.4.weight': 'cov_net.5.weight',
    'cov_net.4.bias':   'cov_net.5.bias',
}


def remap_acausal_state_dict(state_dict, prefix='mes_net.'):
    """
    Translate an acausal-MesNet state_dict so it loads into CausalMesNet,
    for WARM-STARTING causal training from the existing trained weights.
    Conv weights are shape-identical; only their index within cov_net moves.
    `prefix` is the mes_net prefix when the dict is a full TORCHIEKF state_dict
    (use '' for a bare MesNet state_dict). Non-mesnet keys pass through unchanged.
    """
    out = {}
    for k, v in state_dict.items():
        sub = k[len(prefix):] if k.startswith(prefix) else None
        if sub is not None and sub in _REMAP:
            out[prefix + _REMAP[sub]] = v
        else:
            out[k] = v
    return out


def attach_causal_mesnet(torch_iekf, warm_start_state_dict=None):
    """
    Replace torch_iekf.mes_net with a CausalMesNet (in place).
    If warm_start_state_dict (a full acausal TORCHIEKF state_dict) is given, the
    conv/linear weights are remapped and loaded so causal training starts from the
    trained acausal weights instead of from scratch.
    Returns torch_iekf.
    """
    causal = CausalMesNet()
    if warm_start_state_dict is not None:
        remapped = remap_acausal_state_dict(warm_start_state_dict, prefix='mes_net.')
        mes_sd = {k[len('mes_net.'):]: v for k, v in remapped.items()
                  if k.startswith('mes_net.')}
        missing, unexpected = causal.load_state_dict(mes_sd, strict=False)
        if missing or unexpected:
            print(f"[causal_mesnet] warm-start partial load "
                  f"(missing={list(missing)}, unexpected={list(unexpected)})")
    torch_iekf.mes_net = causal
    return torch_iekf
