"""
Independent verification of the claimed Deep-KF Hailo calibration mismatch.

Captures the REAL e_norm_in tensors the CPU production loop
(deep_kf_runner.py:454, `e_norm_in = (e - norm_mean) / norm_std`) actually
feeds to the LSTM during a real KITTI run (via a forward hook — no filter
logic duplicated, so this is the true production distribution), then runs
those exact samples through:

  (a) PyTorch DeepKFNetONNX wrapper (reference)
  (b) the EXISTING deep_kf_quantized_model.har (SDK_QUANTIZED emulator) —
      the HEF as currently calibrated (on raw absolute nav states, per
      2_optimisation.py's calib_nav construction)

and reports MAE / worst-case / position-channel error, to check whether the
existing HEF is really badly out-of-distribution for the signal the CPU path
actually uses.
"""
import sys
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent.parent
_DKF_DIR = _REPO / "scripts/positioning/python/dl_filters/deep_kf"
_PY_DIR = _REPO / "scripts/positioning/python"
for _p in [str(_PY_DIR), str(_DKF_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_loader
import deep_kf_runner as dkr
from model import DeepKFNet

FOLD = _REPO / "artifacts/deep_kf/fold_01.pt"
HAR = _HERE / "deep_kf_quantized_model.har"
SEQ = "01"

STATE_LABELS = "p_e p_n p_u v_e v_n v_u roll pitch yaw ba_x ba_y ba_z bg_x bg_y bg_z".split()


def capture_real_inputs(nav):
    """Run the real CPU loop, hook DeepKFNet.forward to record every e_norm_in
    tensor it actually receives (the true production distribution)."""
    captured = []
    orig_forward = DeepKFNet.forward

    def hooked(self, nav_state, hidden=None):
        captured.append(nav_state.detach().cpu().numpy().copy())
        return orig_forward(self, nav_state, hidden)

    DeepKFNet.forward = hooked
    try:
        dkr.run(nav, backend='cpu', outage_config={'start': 0., 'duration': 0.})
    finally:
        DeepKFNet.forward = orig_forward
    arr = np.concatenate(captured, axis=0)
    return arr[:, 0, :] if arr.ndim == 3 else arr  # -> (N, 15)


def main():
    nav = data_loader.get_kitti_dataset(SEQ)
    print(f"Capturing real e_norm_in from CPU production loop on seq{SEQ} ...")
    e_norm_in = capture_real_inputs(nav)
    print(f"Captured {len(e_norm_in)} real production samples, shape {e_norm_in.shape}")
    np.savez(str(_HERE / "e_norm_in_capture.npz"), e_norm_in=e_norm_in)
    print(f"  range per channel (min..max) and percentiles (p1/p50/p99):")
    for i, lab in enumerate(STATE_LABELS):
        col = e_norm_in[:, i]
        p1, p50, p99 = np.percentile(col, [1, 50, 99])
        print(f"    {lab:6s} min={col.min():9.3f} max={col.max():9.3f}  "
              f"p1={p1:8.3f} p50={p50:8.3f} p99={p99:8.3f}")
    # when (if ever) does yaw first exceed a sane bound (say |val|>20)?
    yaw = e_norm_in[:, 8]
    bad = np.where(np.abs(yaw) > 20)[0]
    print(f"  yaw: {len(bad)}/{len(yaw)} samples have |yaw_norm|>20 "
          f"(first at step {bad[0] if len(bad) else 'never'}, "
          f"last at step {bad[-1] if len(bad) else 'never'})")

    # subsample for the comparison (deterministic, evenly spaced)
    N_TEST = min(500, len(e_norm_in))
    idx = np.linspace(0, len(e_norm_in) - 1, N_TEST, dtype=int)
    x_test = e_norm_in[idx].astype(np.float32)

    # ── PyTorch reference ──────────────────────────────────────────────────
    ckpt = torch.load(str(FOLD), map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    pt_model = DeepKFNet(nav_state_dim=15,
                        hidden_dim=cfg.get("latent_dim", 128),
                        num_layers=cfg.get("num_layers", 2))
    pt_model.load_state_dict(sd)
    pt_model.eval()

    # import the sibling script directly by path (filenames starting with a
    # digit aren't valid module names for a plain `import`)
    import importlib.util
    spec = importlib.util.spec_from_file_location("dkf_opt", str(_HERE / "2_optimisation.py"))
    dkf_opt = importlib.util.module_from_spec(spec)
    sys.modules["dkf_opt"] = dkf_opt
    spec.loader.exec_module(dkf_opt)

    wrapped = dkf_opt.DeepKFNetONNX(pt_model)
    wrapped.eval()
    x_seq = x_test[:, np.newaxis, :]
    with torch.no_grad():
        delta_pt = wrapped(torch.from_numpy(x_seq)).numpy()

    # ── Existing HAR, SDK_QUANTIZED (the CURRENT wrongly-calibrated HEF) ────
    from hailo_sdk_client import ClientRunner, InferenceContext
    runner = ClientRunner(har=str(HAR))
    names = [l.name for l in runner._hn.get_input_layers()]
    x_nhwc = x_seq[:, :, np.newaxis, :]  # (N,1,1,15) NHWC
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        res = runner.infer(ctx, {names[0]: x_nhwc})
    delta_hef = np.concatenate(res, axis=0).reshape(-1, 15)

    d = np.abs(delta_hef - delta_pt)
    print(f"\n=== Existing HEF (calibrated on raw nav states) vs PyTorch, "
          f"fed the REAL e_norm_in signal ===")
    print(f"mean abs error (delta): {d.mean():.4f}  (signal |delta| mean: {np.abs(delta_pt).mean():.4f})")
    print(f"worst case:             {d.max():.4f}")
    print(f"per-channel MAE:")
    for i, lab in enumerate(STATE_LABELS):
        extra = ""
        if i < 3:
            denorm = d[:, i].mean() * 577.9638 if i == 0 else d[:, i].mean()
            extra = f"   (denorm by p_e std=578m: {d[:,i].mean()*577.9638:.2f} m)" if i == 0 else ""
        print(f"  {lab:6s} {d[:, i].mean():8.5f}{extra}")


if __name__ == "__main__":
    main()
