#!/usr/bin/env python3
"""
Deep IEKF (online) cross-machine diagnostic — dumps intermediate filter
state (p, v, r) at fixed checkpoints for KITTI seq01, no outage, so two
machines running the SAME inputs (identical nav_data, identical
fold_01.p weights, identical tuned params — all confirmed byte-identical
between the two machines this is being compared across) can be diffed
step-by-step to find exactly where the trajectories first diverge,
instead of only comparing final ATE.

Run from scripts/positioning/python/:
    python3 diag_deep_iekf_seq01.py

Writes deep_iekf_seq01_diag_<hostname>.json to the current directory and
prints the same checkpoints to stdout.
"""
import json
import os
import socket
import sys

sys.path.insert(0, '.')
sys.path.insert(0, 'dl_filters/deep_iekf')

import numpy as np
import data_loader
import ins_compare
import iekf_ai_imu_online as m

SEQ = '01'
os.environ['AI_IMU_ONLINE_WEIGHTS'] = os.path.abspath(
    f'../../../artifacts/deep_iekf_online/fold_{SEQ}.p')

nav = data_loader.get_kitti_dataset(SEQ)
tuned = ins_compare._load_tuned_params(nav, True)
params = tuned.get('iekf_ai_imu_online')
print('Weights path:', os.environ['AI_IMU_ONLINE_WEIGHTS'])
print('Rpos used:', (params or {}).get('Rpos'))

result = m.run(nav, params=params, outage_config={'start': 0.0, 'duration': 0.0},
               use_3d_rotation=True)

N = result['p'].shape[0]
checkpoints = sorted(set(c for c in [0, 100, 500, 1000, 3000, 6000, 9000, N - 1] if c < N))

dump = {
    'host': socket.gethostname(),
    'seq': SEQ,
    'N': int(N),
    'weights_path': os.environ['AI_IMU_ONLINE_WEIGHTS'],
    'params': params,
    'checkpoints': {
        str(i): {
            'p': result['p'][i].tolist(),
            'v': result['v'][i].tolist(),
            'r': result['r'][i].tolist(),
        } for i in checkpoints
    },
}

out_path = f'deep_iekf_seq01_diag_{socket.gethostname()}.json'
with open(out_path, 'w') as f:
    json.dump(dump, f, indent=2)
print(f'Wrote {out_path}')
for i in checkpoints:
    print(f"  step {i:6d}  p={result['p'][i]}  v={result['v'][i]}  r={result['r'][i]}")
