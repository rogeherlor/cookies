import os, sys, socket, json, hashlib
sys.path.insert(0, '.')
sys.path.insert(0, 'dl_filters/deep_iekf')
import numpy as np
import torch
import data_loader
from iekf_ai_imu_online import _find_online_weights, _load_causal_torch_iekf, _build_inputs

SEQ = '01'
os.environ['AI_IMU_ONLINE_WEIGHTS'] = os.path.abspath(f'../../../artifacts/deep_iekf_online/fold_{SEQ}.p')
nav = data_loader.get_kitti_dataset(SEQ)
N, sr, t, u_np, ang0, v0 = _build_inputs(nav)
weights = _find_online_weights()
ti = _load_causal_torch_iekf(u_np, weights)

def h(x):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(x, dtype=np.float64)).tobytes()).hexdigest()[:12]

rep = {'host': socket.gethostname(), 'torch': torch.__version__, 'numpy': np.__version__,
       'weights_path_resolved': str(weights)}
rep['u_np_sha'] = h(u_np); rep['u_np_0'] = u_np[0].tolist(); rep['u_np_26'] = u_np[26].tolist()
rep['u_loc'] = ti.u_loc.detach().cpu().numpy().tolist()
rep['u_std'] = ti.u_std.detach().cpu().numpy().tolist()
rep['cov0_measurement'] = ti.cov0_measurement.detach().cpu().numpy().tolist()
with torch.no_grad():
    u_t = torch.from_numpy(u_np).double()
    u_n = ti.normalize_u(u_t).t().unsqueeze(0)[:, :6]
    rep['u_n_sha'] = h(u_n.cpu().numpy())
    x = u_n; mags = []
    for idx, layer in enumerate(ti.mes_net.cov_net):
        x = layer(x)
        mags.append({'idx': idx, 'type': type(layer).__name__, 'absmean': float(x.abs().mean())})
    rep['cov_net_layer_absmean'] = mags
    y_cov = x.transpose(1, 2).squeeze(0)
    rep['y_cov_absmean'] = float(y_cov.abs().mean())
    z_cov = ti.mes_net.cov_lin(y_cov)
    rep['z_cov_0'] = z_cov[0].cpu().numpy().tolist(); rep['z_cov_26'] = z_cov[26].cpu().numpy().tolist()
    rep['final_cov_26'] = ti.forward_nets(u_t).cpu().numpy()[26].tolist()
rep['w'] = {p: h(tt.detach().cpu().numpy()) for p, tt in ti.mes_net.named_parameters()}
rep['w_absmean'] = {p: float(tt.detach().abs().mean()) for p, tt in ti.mes_net.named_parameters()}
print(json.dumps(rep, indent=2))

# conv-vs-numpy sanity on THIS machine
with torch.no_grad():
    pad0 = ti.mes_net.cov_net[0](u_n); conv1 = ti.mes_net.cov_net[1]
    to = conv1(pad0).cpu().numpy()[0]; xin = pad0.cpu().numpy()[0]
    W = conv1.weight.detach().cpu().numpy(); b = conv1.bias.detach().cpu().numpy()
Nout = to.shape[1]; man = np.zeros((32, Nout))
for k in range(5): man += np.einsum('oc,cn->on', W[:, :, k], xin[:, k:k+Nout])
man += b[:, None]
print("conv1 torch-vs-numpy max abs diff:", np.abs(to - man).max())
