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
        """Device-only time. NOTE: this is NOT what `.step()` should return as
        its Event latency — see the class docstrings. The CPU backends time the
        whole `model(x)` forward pass, so a Hailo `.step()` that returned only
        this would compare a bare device call against a full forward pass,
        overstating the offload and silently pushing the excluded host work
        (input marshalling, and for some models a real slice of the network
        that stays on the host) into the Routine column instead. Every
        `.step()` therefore times from the first host op to the last, and uses
        this only for the device-side accounting in n_calls/total_s."""
        t0 = time.perf_counter()
        self.configured_model.run([bindings], timeout_ms)
        dt = time.perf_counter() - t0
        self.n_calls += 1
        self.total_s += dt
        return dt


# ── deep_kf ─────────────────────────────────────────────────────────────────

class HailoDeepKF(_BaseHailoNet):
    """Per-tick causal LSTM error-state predictor — STATEFUL.

    Runs the SAME error-state EKF loop as the CPU path (deep_kf_runner.py's
    `run()`), with only the LSTM forward pass swapped to this device. `.step()`
    is fed the normalised error state e_norm_in = (e - norm_mean)/norm_std the
    CPU LSTM is trained on, NOT a raw absolute nav state.

    Recurrence
    ----------
    The HEF is a single-step LSTM CELL: both layers' hidden and cell states are
    ordinary device inputs and outputs, and this class carries them across
    `.step()` calls, exactly as the CPU runner carries `hidden`. Sequences start
    from zeros, matching the CPU path.

    This replaces an earlier export in which h0 was baked into the graph as a
    constant, leaving one input and one output and therefore no way to carry
    state. That model was memoryless — verified by feeding the same input
    repeatedly and getting bit-identical outputs regardless of history — while
    the trained model is a stateful two-layer LSTM. The cost was not subtle:
    on the outage scenario it moved sequence 01 from 82 m to 272 m of ATE
    before quantisation was even applied, and the network-level error against
    PyTorch was 9.3e-2 where the stateful graph achieves 7.3e-4.

    `deep_kf_runner.py`'s `_run_hailo()` only invokes the LSTM prediction
    during a GENUINE GPS outage (matching the paper's intent and
    esekfs_enhanced.py's outage handling) — not on every routine tick.
    """

    STATE_DIM = 15

    def _setup_io(self):
        for n in self.infer_model.input_names:
            self.infer_model.input(n).set_format_type(self.hp.FormatType.FLOAT32)
        for n in self.infer_model.output_names:
            self.infer_model.output(n).set_format_type(self.hp.FormatType.FLOAT32)

        # Map by the input_layerN suffix, which follows PARSE order
        # (x, h_l0, c_l0, h_l1, c_l1 — see deep_kf/1_parsing.py). The order of
        # infer_model.input_names does NOT: it comes back as layer 1,5,3,4,2.
        # Binding positionally would swap c_l1 into h_l0 and silently produce a
        # plausible-looking but wrong trajectory.
        ins = sorted(self.infer_model.input_names,
                     key=lambda n: int(n.rsplit('input_layer', 1)[-1]))
        if len(ins) != 5:
            raise RuntimeError(
                f"deep_kf HEF has {len(ins)} inputs, expected 5 "
                f"(x, h_l0, c_l0, h_l1, c_l1): {ins}. This is the old "
                f"memoryless single-input export — rebuild it.")
        self._in_x, self._in_h0, self._in_c0, self._in_h1, self._in_c1 = ins

        # Outputs come back in end-node order: state, h_l0_o, c_l0_o, h_l1_o, c_l1_o.
        outs = list(self.infer_model.output_names)
        self._out_state = next(
            n for n in outs
            if int(np.prod(self.infer_model.output(n).shape)) == self.STATE_DIM)
        rest = [n for n in outs if n != self._out_state]
        self._out_h0, self._out_c0, self._out_h1, self._out_c1 = rest
        self._hidden_dim = int(np.prod(self.infer_model.output(self._out_h0).shape))
        self.reset()

    def reset(self):
        """Zero the recurrent state — start of a sequence, as the CPU path does."""
        z = lambda: np.zeros((1, 1, self._hidden_dim), dtype=np.float32)
        self._h0, self._c0, self._h1, self._c1 = z(), z(), z(), z()

    def __enter__(self):
        super().__enter__()
        self.reset()
        return self

    def step(self, e_norm_in_15):
        """e_norm_in_15: (15,) normalised error state (e-norm_mean)/norm_std.
        Returns (predicted next normalised error state (15,), elapsed_s); the
        caller denormalises. The recurrent state advances as a side effect.

        The returned time spans host marshalling + device call + the residual
        add and state write-back, matching what the CPU backend times."""
        _t0 = time.perf_counter()
        bindings = self.configured_model.create_bindings()
        sample = np.ascontiguousarray(
            np.asarray(e_norm_in_15, dtype=np.float32).reshape(1, 1, self.STATE_DIM))
        bindings.input(self._in_x).set_buffer(sample)
        for name, buf in ((self._in_h0, self._h0), (self._in_c0, self._c0),
                          (self._in_h1, self._h1), (self._in_c1, self._c1)):
            bindings.input(name).set_buffer(buf)

        out_state = np.empty(self.infer_model.output(self._out_state).shape,
                             dtype=np.float32)
        bindings.output(self._out_state).set_buffer(out_state)
        nxt = {}
        for name in (self._out_h0, self._out_c0, self._out_h1, self._out_c1):
            b = np.empty(self.infer_model.output(name).shape, dtype=np.float32)
            bindings.output(name).set_buffer(b)
            nxt[name] = b

        self._timed_run(bindings)

        self._h0 = np.ascontiguousarray(nxt[self._out_h0].reshape(1, 1, -1))
        self._c0 = np.ascontiguousarray(nxt[self._out_c0].reshape(1, 1, -1))
        self._h1 = np.ascontiguousarray(nxt[self._out_h1].reshape(1, 1, -1))
        self._c1 = np.ascontiguousarray(nxt[self._out_c1].reshape(1, 1, -1))

        delta = out_state.reshape(-1).astype(np.float64)
        out = np.asarray(e_norm_in_15, dtype=np.float64) + delta
        return out, time.perf_counter() - _t0


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

        The returned time spans the WHOLE hybrid update: input marshalling,
        the device call, and the host-side bn1+FC head (`_apply_head`). The
        head is a genuine part of this network that the partitioning leaves on
        the CPU, so excluding it would both overstate the offload and misplace
        its cost in the Routine column.
        """
        _t0 = time.perf_counter()
        bindings = self.configured_model.create_bindings()
        sample_nchw = np.asarray(window_6xW, dtype=np.float32)[:, np.newaxis, :]  # (6,1,W)
        sample = np.ascontiguousarray(sample_nchw.transpose(1, 2, 0))              # (1,W,6) NHWC
        bindings.input(self._input_name).set_buffer(sample)
        out_buf1 = np.empty(self._out_shape, dtype=np.float32)
        out_buf2 = np.empty(self._out_shape, dtype=np.float32)
        bindings.output(self._out_name1).set_buffer(out_buf1)
        bindings.output(self._out_name2).set_buffer(out_buf2)
        self._timed_run(bindings)
        mean = self._infer_mod._apply_head(out_buf1, self._head1_state)[0]
        logstd = self._infer_mod._apply_head(out_buf2, self._head2_state)[0]
        return (mean.astype(np.float64), logstd.astype(np.float64)), time.perf_counter() - _t0


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
        _t0 = time.perf_counter()
        step_nhwc = self._infer_mod._step_to_nhwc(np.asarray(window_10x200x6, dtype=np.float32))
        feats = []
        for t in range(self._lstm_steps):
            bindings = self.configured_model.create_bindings()
            bindings.input(self._input_name).set_buffer(step_nhwc[t])
            out_buf = np.empty(self._out_shape, dtype=np.float32)
            bindings.output(self._output_name).set_buffer(out_buf)
            self._timed_run(bindings)
            feats.append(out_buf.squeeze(0).T.reshape(1, -1))
        feats_seq = np.concatenate(feats, axis=0)[np.newaxis]  # (1, LSTM_STEPS, 1664)
        with torch.no_grad():
            v, log_std = self._postproc(torch.from_numpy(feats_seq.astype(np.float32)))
        return (v[0].numpy().astype(np.float64),
                log_std[0].numpy().astype(np.float64)), time.perf_counter() - _t0


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
        """imu_np: (SEQ_LEN, 6) float32 raw IMU. Returns (covs (SEQ_LEN,2), elapsed_s).

        Time spans host preprocess + device call + host postprocess, matching
        the span the CPU backend times."""
        _t0 = time.perf_counter()
        u_nchw = self._infer_mod.preprocess(imu_np, self._pp)   # (1,6,1,N)
        u_nhwc = u_nchw.transpose(0, 2, 3, 1)                    # (1,1,N,6)
        bindings = self.configured_model.create_bindings()
        out_info = self.infer_model.output()
        out_buf = np.empty(out_info.shape, dtype=np.float32)
        bindings.input(self._input_name).set_buffer(u_nhwc)
        bindings.output(self._output_name).set_buffer(out_buf)
        self._timed_run(bindings, timeout_ms=5000)
        covs = self._infer_mod.postprocess(out_buf, self._pp)
        return covs, time.perf_counter() - _t0


# ── deep_iekf (streaming, small window) ──────────────────────────────────────

class HailoDeepIEKFStream(_BaseHailoNet):
    """Streaming causal MesNet covariance predictor (fixed SMALL window).

    Companion to HailoDeepIEKF: instead of one fixed 4544-sample whole-sequence
    call (which can only cover min(N,4544) of a real sequence — 8..40% of most
    KITTI drives), this HEF has a fixed W=SEQ_LEN=32 window and is driven over
    the whole sequence in one of two modes (`.run_stream`):

      * 'per_tick' : slide by 1 each new IMU sample, keep the last output
                     (genuine online — one HEF call per sample, ~1-tick latency)
      * 'block'    : slide by FRESH=16, feed 32 (=16 carried context + 16 new),
                     keep the last 16 outputs (16 covariances per HEF call)

    Both are bit-faithful to the whole-sequence pass in steady state because the
    network is causal with receptive field 17: an output keeps only its own 16
    samples of real left context (proven in
    hailo/deep_iekf_stream/verify_chunk_math.py to <=2e-6).  Only the first ~16
    samples of the whole stream differ (ZeroPad warmup — same as HailoDeepIEKF).

    On-device output is the BOUNDED z = tanh(cov_lin(cov_net(u))) in [-1,1]; the
    covariance cov = cov0*10**(beta*z) is reconstructed here on the host.  This
    is deliberate: baking the exp scaling on-device (as HailoDeepIEKF does) makes
    the HEF emit a heavy-tailed 0..~10000 range that INT8 quantises coarsely —
    catastrophic for folds whose covariances spike (fold_06: 150 m drift).  A
    bounded z quantises uniformly well for every fold; see
    hailo/deep_iekf_stream/build_stream.py.

    postproc npz (deep_iekf_stream_fold_XX_postproc.npz) carries u_loc,u_std
    (per-fold input normalisation) plus beta,cov0 (the host scaling constants).
    """

    SEQ_LEN = 32                 # fixed HEF window W
    STREAMING = True             # runtime uses .run_stream(), not run_whole_sequence()
    RF = 17                      # receptive field
    CARRY = 16                   # left-context samples carried between blocks (RF-1)
    FRESH = SEQ_LEN - CARRY      # fresh outputs kept per block (=16)

    def __init__(self, hef_path, postproc_npz_path, mode="block"):
        super().__init__(hef_path)
        pp = np.load(str(postproc_npz_path))
        self._u_loc = pp["u_loc"].astype(np.float32)
        self._u_std = pp["u_std"].astype(np.float32)
        self._beta  = pp["beta"].astype(np.float64)
        self._cov0  = pp["cov0"].astype(np.float64)
        if mode not in ("block", "per_tick"):
            raise ValueError(f"mode must be 'block' or 'per_tick', got {mode!r}")
        self.mode = mode

    def _setup_io(self):
        self.infer_model.input().set_format_type(self.hp.FormatType.FLOAT32)
        self.infer_model.output().set_format_type(self.hp.FormatType.FLOAT32)
        self._input_name = self.infer_model.input().name
        self._output_name = self.infer_model.output().name
        self._out_shape = self.infer_model.output().shape

    def _to_cov(self, z):
        """z (...,2) -> covariance (...,2) via cov0*10**(beta*z)."""
        return self._cov0 * np.power(10.0, self._beta * np.asarray(z, np.float64))

    def _infer_window(self, seg_raw, pad):
        """seg_raw: (<=SEQ_LEN, 6) RAW IMU, the window's real samples.
        pad: how many left rows are outside the stream. Returns
        (cov (SEQ_LEN,2), elapsed_s, host_cpu_s).

        Padding is zeros in NORMALISED space, not raw space — that is what the
        HEF was built and verified against (build_stream.py / verify_chunk_math.py),
        so the normalisation is applied to the real samples only and the pad rows
        stay exactly zero.

        The returned time spans the whole host+device sequence, over the same
        operations the CPU backend's span covers. The CPU Event latency for this
        approach is `torch_iekf.forward_nets(win)`
        (iekf_ai_imu_online.py::causal_mes_inference), and forward_nets is
        `normalize_u` -> cov_net/cov_lin -> `cov0 * 10**(beta*z)`
        (utils_torch_filter.py:423, MesNet.forward). Input normalisation and the
        exponential covariance scaling are therefore both inside the CPU's
        measured span and must be inside this one; timing only the device call
        would understate Hailo's per-event cost and push the remainder into the
        Routine column — the same asymmetry already fixed in the four `.step()`
        methods. Window construction is outside the span on both backends."""
        _t0 = time.perf_counter()
        _c0 = time.thread_time()
        win = np.zeros((self.SEQ_LEN, 6), np.float32)
        seg = np.asarray(seg_raw, np.float32)
        win[pad:pad + len(seg)] = (seg - self._u_loc) / self._u_std
        nhwc = np.ascontiguousarray(win[np.newaxis, np.newaxis, :, :])
        bindings = self.configured_model.create_bindings()
        bindings.input(self._input_name).set_buffer(nhwc)
        out_buf = np.empty(self._out_shape, dtype=np.float32)
        bindings.output(self._output_name).set_buffer(out_buf)
        self._timed_run(bindings)
        cov = self._to_cov(out_buf.reshape(-1, 2))
        # host_cpu_s is the share of the span this thread actually computed on;
        # the device wait is excluded. Reported (not warned on) so a reader can
        # separate "blocked on the accelerator" from "preempted by other load" —
        # see _full_eval_worker.py's wall/cpu self-check.
        return cov, time.perf_counter() - _t0, time.thread_time() - _c0

    def step(self, window_raw):
        """Per-tick call. window_raw: (<=SEQ_LEN, 6) raw IMU ending at the current
        sample (caller keeps a rolling buffer). Returns (cov (2,), elapsed_s) for
        the current (last) sample. Left zero-padded at stream start.

        The returned time spans input normalisation, the device call and the
        host-side 10**(beta*z) covariance scaling — the same span the CPU backend
        times when it runs the whole MesNet forward pass."""
        seg = np.asarray(window_raw, np.float32)[-self.SEQ_LEN:]
        cov, dt, _ = self._infer_window(seg, self.SEQ_LEN - len(seg))
        return cov[-1], dt

    def run_stream(self, imu_np):
        """Whole-sequence driver. imu_np: (N,6) raw IMU. Returns (covs (N,2),
        per_call_latencies (num_calls,), per_call_host_cpu (num_calls,)).
        Reconstructs the full-sequence causal covariances; per-call latency is
        genuine per-tick (per_tick) or per-block (block) cost, host work included
        — see _infer_window."""
        u = np.asarray(imu_np, np.float32)
        N = len(u)
        covs = np.zeros((N, 2), np.float64)
        lat, cpu = [], []
        if self.mode == "per_tick":
            for i in range(N):
                lo = i - (self.SEQ_LEN - 1); src = max(lo, 0)
                cov, dt, ct = self._infer_window(u[src:i + 1], src - lo)
                lat.append(dt); cpu.append(ct)
                covs[i] = cov[-1]
        else:  # block
            for t in range(0, N, self.FRESH):
                lo = t - self.CARRY; src = max(lo, 0)
                cov, dt, ct = self._infer_window(u[src:lo + self.SEQ_LEN], src - lo)
                lat.append(dt); cpu.append(ct)
                n = min(self.FRESH, N - t)
                covs[t:t + n] = cov[self.CARRY:self.CARRY + n]
        return covs, np.asarray(lat), np.asarray(cpu)
