# -*- coding: utf-8 -*-
"""
Hailo-8L step-wise inference backends for the causal online CPU-vs-Hailo
evaluation (run_online_eval.py).

Each class here wraps ONE .hef with a persistent VDevice/configure/activate
lifecycle (opened once, kept alive for an entire causal simulation run) and
exposes a per-tick `.step(...)` call (or `.run_whole_sequence(...)` for
deep_iekf, which is architecturally a single whole-sequence call — see
hailo/deep_iekf/4_inference.py). This is deliberately NOT the batch-loop
shape of the offline hailo/<approach>/4_inference.py scripts (which open one
VDevice and loop over N samples internally, purely to iterate HEF calls with
no real-time pacing) — here each call is invoked from *inside* the relevant
dl_filters/<approach>/*_runner.py causal loop, interleaved with the same
outage-gated filter math the CPU backend uses, so per-call wall-clock time
reflects genuine per-tick cost.

All shape/dtype/pre-post-processing conventions below are ported directly
from the already-validated hailo/<approach>/4_inference.py `infer_hef()`
functions — not re-derived — to avoid subtly diverging from what those
scripts already proved works against real hardware.
"""

import importlib.util
import time
from pathlib import Path

import numpy as np

_HAILO_DIR = Path(__file__).resolve().parent


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _open_vdevice():
    import hailo_platform as hp
    params = hp.VDevice.create_params()
    return hp, hp.VDevice(params)


class _BaseHailoNet:
    """Common VDevice → create_infer_model → configure → activate lifecycle.

    Kept open for the lifetime of the `with` block so that HailoRT persists
    any on-chip recurrent state across `.step()` calls within one activation
    (relevant for deep_kf's LSTM — see hailo/deep_kf/0_onxx_converter.py).
    """

    def __init__(self, hef_path):
        self.hef_path = str(hef_path)
        self.hp = None
        self._vdevice_cm = None
        self.vdevice = None
        self._configured_cm = None
        self.configured_model = None
        self.infer_model = None
        self.n_calls = 0
        self.total_s = 0.0

    def __enter__(self):
        self.hp, self._vdevice_cm = _open_vdevice()
        self.vdevice = self._vdevice_cm.__enter__()
        self.infer_model = self.vdevice.create_infer_model(self.hef_path)
        self.infer_model.set_batch_size(1)
        self._setup_io()
        self._configured_cm = self.infer_model.configure()
        self.configured_model = self._configured_cm.__enter__()
        self.configured_model.activate()
        return self

    def _setup_io(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc, tb):
        try:
            self.configured_model.deactivate()
        finally:
            self._configured_cm.__exit__(exc_type, exc, tb)
            self._vdevice_cm.__exit__(exc_type, exc, tb)

    def _timed_run(self, bindings, timeout_ms=1000):
        t0 = time.perf_counter()
        self.configured_model.run([bindings], timeout_ms)
        dt = time.perf_counter() - t0
        self.n_calls += 1
        self.total_s += dt
        return dt


# ── deep_kf ─────────────────────────────────────────────────────────────────

class HailoDeepKF(_BaseHailoNet):
    """Per-tick causal LSTM state predictor (raw full 15D nav state I/O).

    IMPORTANT — calibration compatibility: this HEF was quantisation-
    calibrated (hailo/deep_kf/2_optimisation.py) on REAL FULL NAV STATES
    (position in metres, velocity in m/s, etc — large, heterogeneous-scale
    values), matching exactly what DeepKFNetONNX's documented interface
    expects. The true CPU production filter (deep_kf_runner.py) instead
    feeds the LSTM a NORMALISED ERROR STATE (small, ~zero-centred residuals)
    inside an EKF wrapper — a different signal the HEF was never calibrated
    for. Feeding error-state values into this HEF would be badly
    out-of-distribution for its INT8 quantisation range and is not a valid
    substitution. This class therefore runs the network in the ONLY role
    it's actually calibrated/compatible for: a standalone autoregressive
    full-state predictor (Eq. 21 of Hosseinyalamdary 2018, matching
    model.py's documented formula) — same trained weights as the CPU LSTM,
    reused causally tick-by-tick via HailoRT's persistent LSTM state across
    `.step()` calls, GPS-reset when a fix is available (see
    deep_kf_runner.py's `run(..., backend='hailo')` branch for how this is
    wired into the outage-gated simulation loop).
    """

    STATE_DIM = 15

    def _setup_io(self):
        self.infer_model.input().set_format_type(self.hp.FormatType.FLOAT32)
        self.infer_model.output().set_format_type(self.hp.FormatType.FLOAT32)
        self._input_name = self.infer_model.input().name
        self._output_name = self.infer_model.output().name

    def step(self, full_state_15):
        """full_state_15: (15,) float — returns (predicted_next_state (15,), elapsed_s)."""
        bindings = self.configured_model.create_bindings()
        sample = np.asarray(full_state_15, dtype=np.float32).reshape(1, self.STATE_DIM)
        bindings.input(self._input_name).set_buffer(sample)
        out_buf = np.empty((1, self.STATE_DIM), dtype=np.float32)
        bindings.output(self._output_name).set_buffer(out_buf)
        dt = self._timed_run(bindings)
        delta = out_buf[0].astype(np.float64)
        return np.asarray(full_state_15, dtype=np.float64) + delta, dt


# ── tlio ────────────────────────────────────────────────────────────────────

class HailoTLIO(_BaseHailoNet):
    """Per-window TLIO displacement predictor.

    Hailo runs the CNN backbone only (two output heads, pre-bn1); the final
    bn1+flatten+fc1/2/3 head runs on the host from tlio_postproc.pt weights
    (see hailo/tlio/4_inference.py / 0_onnx_converter.py) — this is a
    genuine hybrid accelerator+host deployment shape, not a shortcut, and
    `.step()` times the full backbone-call + host-head latency together
    since that is the real per-update cost a live deployment would see.
    """

    def __init__(self, hef_path, postproc_path):
        super().__init__(hef_path)
        self._infer_mod = _load_module(_HAILO_DIR / "tlio" / "4_inference.py", "_tlio_hailo_infer")
        import torch
        postproc = torch.load(str(postproc_path), map_location="cpu")
        self._head1_state = postproc["head1"]
        self._head2_state = postproc["head2"]

    def _setup_io(self):
        self._input_name = self.infer_model.input_names[0]
        self._out_name1, self._out_name2 = self.infer_model.output_names
        self.infer_model.input().set_format_type(self.hp.FormatType.FLOAT32)
        self.infer_model.output(self._out_name1).set_format_type(self.hp.FormatType.FLOAT32)
        self.infer_model.output(self._out_name2).set_format_type(self.hp.FormatType.FLOAT32)
        self._out_shape = self.infer_model.output(self._out_name1).shape

    def step(self, window_6xW):
        """window_6xW: (6, W) float32 gravity-aligned IMU window.

        Returns ((mean(3), logstd(3)), elapsed_s) — matches
        tlio_runner.py::_predict_displacement's `mean, logstd = model(x)`.
        """
        bindings = self.configured_model.create_bindings()
        sample_nchw = np.asarray(window_6xW, dtype=np.float32)[:, np.newaxis, :]  # (6,1,W)
        sample = np.ascontiguousarray(sample_nchw.transpose(1, 2, 0))              # (1,W,6) NHWC
        bindings.input(self._input_name).set_buffer(sample)
        out_buf1 = np.empty(self._out_shape, dtype=np.float32)
        out_buf2 = np.empty(self._out_shape, dtype=np.float32)
        bindings.output(self._out_name1).set_buffer(out_buf1)
        bindings.output(self._out_name2).set_buffer(out_buf2)
        dt = self._timed_run(bindings)
        mean = self._infer_mod._apply_head(out_buf1, self._head1_state)[0]
        logstd = self._infer_mod._apply_head(out_buf2, self._head2_state)[0]
        return (mean.astype(np.float64), logstd.astype(np.float64)), dt


# ── tartan_imu ──────────────────────────────────────────────────────────────

class HailoTartanIMU(_BaseHailoNet):
    """Per-window Tartan IMU velocity predictor.

    Hailo runs the CNN backbone, one LSTM step at a time (LSTM_STEPS=10
    sequential Hailo calls per update); the LSTM + IMU_Trunk + robot head
    run on the host (see hailo/tartan_imu/4_inference.py). `.step()` times
    all 10 backbone calls plus the host postproc together, since that sum
    is the real per-update latency of this hybrid deployment.
    """

    def __init__(self, hef_path, postproc_path, robot_type="car"):
        super().__init__(hef_path)
        self._infer_mod = _load_module(_HAILO_DIR / "tartan_imu" / "4_inference.py", "_tartan_hailo_infer")
        self._postproc = self._infer_mod.TartanPostproc(postproc_path, robot_type=robot_type)
        self._lstm_steps = self._infer_mod.LSTM_STEPS

    def _setup_io(self):
        self.infer_model.input().set_format_type(self.hp.FormatType.FLOAT32)
        self.infer_model.output().set_format_type(self.hp.FormatType.FLOAT32)
        self._input_name = self.infer_model.input().name
        self._output_name = self.infer_model.output().name
        self._out_shape = self.infer_model.output().shape

    def step(self, window_10x200x6):
        """window_10x200x6: (LSTM_STEPS, 200, 6) float32 gravity-free IMU block.

        Returns ((v_body(3), log_std(3)), elapsed_s) — matches
        tartan_runner.py's `v_body_t, log_std_t = model(imu_t, robot_type='car')`.
        """
        import torch
        step_nhwc = self._infer_mod._step_to_nhwc(np.asarray(window_10x200x6, dtype=np.float32))
        feats = []
        dt_total = 0.0
        for t in range(self._lstm_steps):
            bindings = self.configured_model.create_bindings()
            bindings.input(self._input_name).set_buffer(step_nhwc[t])
            out_buf = np.empty(self._out_shape, dtype=np.float32)
            bindings.output(self._output_name).set_buffer(out_buf)
            dt_total += self._timed_run(bindings)
            feats.append(out_buf.squeeze(0).T.reshape(1, -1))
        t0 = time.perf_counter()
        feats_seq = np.concatenate(feats, axis=0)[np.newaxis]  # (1, LSTM_STEPS, 1664)
        with torch.no_grad():
            v, log_std = self._postproc(torch.from_numpy(feats_seq.astype(np.float32)))
        dt_total += time.perf_counter() - t0
        return (v[0].numpy().astype(np.float64), log_std[0].numpy().astype(np.float64)), dt_total


# ── deep_iekf ───────────────────────────────────────────────────────────────

class HailoDeepIEKF(_BaseHailoNet):
    """Whole-sequence causal MesNet covariance predictor.

    The compiled HEF has a FIXED input length (SEQ_LEN=4544 samples,
    ~45.44s @ 100Hz) — it processes one whole (left-padded, causal)
    sequence per call, not one IMU sample at a time (see
    hailo/deep_iekf/0_onnx_converter.py and 4_inference.py). This mirrors
    the CPU production runner exactly: iekf_ai_imu_online.py::run() also
    calls `torch_iekf.forward_nets(u_np)` ONCE for the whole sequence
    up-front (the causal, left-padded architecture makes this valid — see
    that module's own comment: "evaluated in a single batch pass (causal by
    construction -> strictly online, 0 ms latency, exact)"). There is
    therefore no per-tick distribution to report for this approach on
    either backend — `.run_whole_sequence()` returns one latency number.
    """

    SEQ_LEN = 4544

    def __init__(self, hef_path, postproc_npz_path):
        super().__init__(hef_path)
        self._infer_mod = _load_module(_HAILO_DIR / "deep_iekf" / "4_inference.py", "_deep_iekf_hailo_infer")
        self._pp = self._infer_mod.load_postproc(postproc_npz_path)

    def _setup_io(self):
        self.infer_model.input().set_format_type(self.hp.FormatType.FLOAT32)
        self.infer_model.output().set_format_type(self.hp.FormatType.FLOAT32)
        self._input_name = self.infer_model.input().name
        self._output_name = self.infer_model.output().name

    def run_whole_sequence(self, imu_np):
        """imu_np: (SEQ_LEN, 6) float32 raw IMU. Returns (covs (SEQ_LEN,2), elapsed_s)."""
        u_nchw = self._infer_mod.preprocess(imu_np, self._pp)   # (1,6,1,N)
        u_nhwc = u_nchw.transpose(0, 2, 3, 1)                    # (1,1,N,6)
        bindings = self.configured_model.create_bindings()
        out_info = self.infer_model.output()
        out_buf = np.empty(out_info.shape, dtype=np.float32)
        bindings.input(self._input_name).set_buffer(u_nhwc)
        bindings.output(self._output_name).set_buffer(out_buf)
        dt = self._timed_run(bindings, timeout_ms=5000)
        return self._infer_mod.postprocess(out_buf, self._pp), dt
