import json, os, socket, sys
sys.path.insert(0, '.')
sys.path.insert(0, 'dl_filters/deep_iekf')
import numpy as np
import torch
import data_loader
import ins_compare
import iekf_ai_imu as iai
import iekf_ai_imu_online as monl
from iekf_ai_imu_online import _find_online_weights, _load_causal_torch_iekf, _build_inputs

SEQ = '01'
os.environ['AI_IMU_ONLINE_WEIGHTS'] = os.path.abspath(f'../../../artifacts/deep_iekf_online/fold_{SEQ}.p')
nav = data_loader.get_kitti_dataset(SEQ)
tuned = ins_compare._load_tuned_params(nav, True)
params = tuned.get('iekf_ai_imu_online')

# ── Part 1: raw covariance network output, first 50 samples ────────────────
N, sr, t, u_np, ang0, v0 = _build_inputs(nav)
weights = _find_online_weights()
torch_iekf = _load_causal_torch_iekf(u_np, weights)
with torch.no_grad():
    covs = torch_iekf.forward_nets(torch.from_numpy(u_np).double()).cpu().numpy()

print("=== covariance network output, samples 0-49 ===")
for i in range(50):
    print(f"{i:3d} {covs[i,0]:.10f} {covs[i,1]:.10f}")

# ── Part 2: GPS update log (innovation + condition number + covariance) ────
_orig_gps = iai._gps_update_step
gps_log = []
def probe_gps(Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, z_gps, R_gps):
    H = np.zeros((3, 21)); H[:, 6:9] = np.eye(3)
    S = H @ P @ H.T + R_gps
    gps_log.append({'cond': float(np.linalg.cond(S)),
                    'std_pos': np.sqrt(np.diag(P)[6:9]).tolist(),
                    'z_gps': z_gps.tolist(), 'p': p.tolist()})
    return _orig_gps(Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, z_gps, R_gps)
iai._gps_update_step = probe_gps
monl._run_filter_loop.__globals__['_gps_update_step'] = probe_gps

result = monl.run(nav, params=params, outage_config={'start': 0.0, 'duration': 0.0},
                  use_3d_rotation=True)

print(f"\n=== {len(gps_log)} GPS updates, first 30 ===")
for i, e in enumerate(gps_log[:30]):
    print(f"gps#{i:3d} cond(S)={e['cond']:10.3f}  std_pos={np.array(e['std_pos'])}  "
         f"|z_gps|={np.linalg.norm(e['z_gps']):9.3f}  p={np.array(e['p'])}")

out = {
    'host': socket.gethostname(), 'seq': SEQ,
    'weights_path': os.environ['AI_IMU_ONLINE_WEIGHTS'],
    'Rpos': (params or {}).get('Rpos'),
    'cov_net_first50': covs[:50].tolist(),
    'gps_log_first30': gps_log[:30],
    'final_p': result['p'][-1].tolist(),
}
out_path = f'diag_covnet_gps_{socket.gethostname()}.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nWrote {out_path}")
