# Deep Learning Filters — Training Guide

Run these steps **before** calling `ins_compare.py` or `ins_runner.py` with any
DL filter. All commands are run from `scripts/positioning/python/`.

---

## Overview

| Filter | Paper | Script | Output artifact |
|--------|-------|--------|-----------------|
| Deep IEKF (AI-IMU) | Brossard et al., IEEE TIV 2020 | `dl_filters/deep_iekf/train_ai_imu.py --causal` | `artifacts/deep_iekf_online/fold_<SEQ>.p` |
| TLIO | Liu et al., IEEE RA-L 2020 | `dl_filters/tlio/train_tlio.py` | `artifacts/tlio/fold_<SEQ>.pt` |
| Deep KF | Hosseinyalamdary, MDPI Sensors 2018 | `dl_filters/deep_kf/train_deep_kf.py` | `artifacts/deep_kf/fold_<SEQ>.pt` |
| Tartan IMU | Zhao et al., CVPR 2025 | `dl_filters/tartan_imu/train_tartan.py` | `artifacts/tartan_imu/lora_fold_<SEQ>.pt` |

**LOO clean sequences:** `01  04  06  07  08  09  10`
Sequences `00 02 05` have ~2-second data gaps; `03` has no raw data.

---

## Python dependencies

```bash
pip install torch torchvision          # all DL filters
pip install peft>=0.6.0               # Tartan IMU LoRA fine-tuning
pip install huggingface_hub>=0.20     # Tartan IMU weight download
```

---

## 1. Deep IEKF / AI-IMU (Brossard et al. 2020)

The acausal checkpoint is tracked in this repo at `artifacts/deep_iekf/iekfnets.p`
(the `external/ai-imu-dr/` clone is source only), and is found automatically, so the
diagnostic batch filter runs out of the box. The filter that is actually evaluated does
need training.

`iekf_ai_imu.py` resolves it in this order: `AI_IMU_WEIGHTS`,
`artifacts/deep_iekf/iekfnets.p`, then `external/ai-imu-dr/src/iekfnets.p` — the last
being where an upstream training run would save one.

The evaluated filter is the causal variant: a left-padded CausalMesNet in place of
MesNet, so no sample sees its own future. Causal is the default, and `--held-out`
takes a full drive name rather than a sequence number. The simplest route is
`ins_train.py ai_imu`, which does the sequence-to-drive translation; directly:

```bash
# From scripts/positioning/python/
python dl_filters/deep_iekf/train_ai_imu.py \
    --mode loo --held-out 2011_09_30_drive_0028_extract \
    --epochs 400 --output ../../../artifacts/deep_iekf_online
```

Weights go to `artifacts/deep_iekf_online/fold_<SEQ>.p`, alongside a `_norm.p` holding
that fold's input normalisation. Passing `--no-causal` trains the acausal batch model
into `artifacts/deep_iekf/` instead; that one is a diagnostic reference and is off by
default in `ins_compare.py`.

Warm-starting the causal conv layers from acausal weights is available
(`--warm-start`) but off by default, so a causal-vs-acausal comparison is not
confounded.

The runner picks the correct fold automatically via `nav_data.dataset_name`.

---

## 2. TLIO (Liu et al. 2020)

Trained from scratch on KITTI vehicle data (pedestrian weights from the original
repo are not used — different motion profile).

**LOO training — one fold per held-out sequence:**

```bash
for SEQ in 01 04 06 07 08 09 10; do
    python dl_filters/tlio/train_tlio.py \
        --mode loo --val-seq $SEQ --epochs 200
done
```

**Or train on all sequences** (use for sequences outside the clean KITTI set):

```bash
python dl_filters/tlio/train_tlio.py --mode all --epochs 200
```

**Key options:**

| Flag | Default | Notes |
|------|---------|-------|
| `--epochs` | 200 | First 100 = MSE pre-training, last 100 = NLL fine-tuning |
| `--batch-size` | 64 | Reduce if GPU OOM |
| `--lr` | 1e-3 | Learning rate (CosineAnnealing schedule) |
| `--resume` | — | Path to `fold_<SEQ>_ckpt.pt` to continue interrupted training |
| `--output` | `artifacts/tlio/` | Where weights are saved |

**Outputs:** `artifacts/tlio/fold_<SEQ>.pt`

---

## 3. Deep KF (Hosseinyalamdary 2018)

LSTM trained to predict IMU bias corrections from navigation state history.
GPS measurements at 1 Hz provide the training supervision signal.

**LOO training:**

```bash
for SEQ in 01 04 06 07 08 09 10; do
    python dl_filters/deep_kf/train_deep_kf.py \
        --mode loo --val-seq $SEQ --epochs 150
done
```

**Or train on all sequences:**

```bash
python dl_filters/deep_kf/train_deep_kf.py --mode all --epochs 150
```

**Key options:**

| Flag | Default | Notes |
|------|---------|-------|
| `--epochs` | 150 | TBPTT over GPS-available segments |
| `--latent-dim` | 128 | LSTM hidden size |
| `--tbptt-len` | 20 | Backprop every N GPS updates (memory vs. gradient quality trade-off) |
| `--lambda-vel` | 0.5 | Weight of velocity loss vs. position loss |
| `--lr` | 1e-3 | Learning rate |
| `--output` | `artifacts/deep_kf/` | Where weights are saved |

**Outputs:** `artifacts/deep_kf/fold_<SEQ>.pt`

---

## 4. Tartan IMU (Zhao et al. CVPR 2025)

> **IMPORTANT: Tartan IMU must NEVER be trained from scratch.**
> The entire value of this filter comes from the pretrained foundation model.
> Only LoRA adapter layers are fine-tuned on KITTI.

### Step 1 — Download pretrained weights

```python
from huggingface_hub import snapshot_download
snapshot_download(
    'raphael-blanchard/TartanIMU',
    repo_type='dataset',
    local_dir='external/tartan_imu',   # relative to scripts/positioning/python/../../..
)
```

The runner searches for weights in this order
(`tartan_runner.py::_find_tartan_weights`):
1. `TARTAN_IMU_WEIGHTS` environment variable
2. `external/tartan_imu/tartan_imu_base.pt` (legacy name)
3. `artifacts/tartan_imu/tartan_imu_base.pt` (legacy name)
4. `external/tartan_imu/checkpoints/foundation_model/checkpoint_<N>.pt` — the layout
   `snapshot_download` above produces; the highest `N` wins

Note that the submodule stores these in Git LFS, so `git submodule update` alone
leaves pointer stubs behind. Use `snapshot_download` as above, or
`git -C external/tartan_imu lfs pull`.

Alternatively, set the env var directly:

```bash
export TARTAN_IMU_WEIGHTS=/path/to/tartan_imu_base.pt
```

**Zero-shot mode:** The filter works without LoRA fine-tuning (uses the `car` head
of the pretrained model). LoRA fine-tuning improves KITTI-specific accuracy.

### Step 2 — Fine-tune LoRA adapters (optional but recommended)

```bash
for SEQ in 01 04 06 07 08 09 10; do
    python dl_filters/tartan_imu/train_tartan.py \
        --mode loo --val-seq $SEQ --epochs 50
done
```

**Key options:**

| Flag | Default | Notes |
|------|---------|-------|
| `--epochs` | 50 | NLL velocity loss; fewer epochs than full training |
| `--lora-rank` | 8 | LoRA rank r; higher = more capacity but more params |
| `--batch-size` | 32 | |
| `--lr` | 1e-3 | AdamW learning rate |
| `--output` | `artifacts/tartan_imu/` | Where LoRA weights are saved |

**Outputs:** `artifacts/tartan_imu/lora_fold_<SEQ>.pt`

---

## LOO convenience loop — all DL filters at once

Run from `scripts/positioning/python/`:

```bash
SEQS="01 04 06 07 08 09 10"

# TLIO — 50 epochs is what ins_train.py uses and what the published results ran;
# the trainer's own default of 200 applies only to direct invocation.
for SEQ in $SEQS; do
    python dl_filters/tlio/train_tlio.py --mode loo --val-seq $SEQ --epochs 50
done

# Deep KF
for SEQ in $SEQS; do
    python dl_filters/deep_kf/train_deep_kf.py --mode loo --val-seq $SEQ --epochs 150
done

# Tartan IMU — download weights first (see Step 1 above), then:
for SEQ in $SEQS; do
    python dl_filters/tartan_imu/train_tartan.py --mode loo --val-seq $SEQ --epochs 50
done
```

---

## Verifying weights before running ins_compare.py

```python
from pathlib import Path

artifacts = Path('../../../../artifacts')   # adjust relative to where you run this
seqs = ['01', '04', '06', '07', '08', '09', '10']

checks = {
    'TLIO':      [artifacts / f'tlio/fold_{s}.pt'       for s in seqs],
    'Deep KF':   [artifacts / f'deep_kf/fold_{s}.pt'    for s in seqs],
    'Tartan':    [artifacts / f'tartan_imu/lora_fold_{s}.pt' for s in seqs],
}

for name, paths in checks.items():
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print(f'{name}: MISSING {missing}')
    else:
        print(f'{name}: OK ({len(paths)} folds)')
```

Once all folds are present, run:

```bash
python ins_compare.py
# or to test a specific sequence:
python ins_compare.py --test-seq 08
```

---

## Artifact directory layout (after full LOO training)

```
artifacts/
├── deep_iekf_online/
│   ├── fold_01.p  …  fold_10.p     # causal AI-IMU — the evaluated filter
│   └── fold_01_norm.p …            # per-fold input normalisation
├── deep_iekf/
│   ├── fold_01.p  …  fold_10.p     # acausal batch AI-IMU (diagnostic only)
├── tlio/
│   ├── fold_01.pt …  fold_10.pt
│   └── tlio_resnet.pt              # all-seqs checkpoint (--mode all)
├── deep_kf/
│   ├── fold_01.pt …  fold_10.pt
│   └── deep_kf.pt
└── tartan_imu/
    ├── lora_fold_01.pt … lora_fold_10.pt
    └── lora_adapters.pt            # all-seqs LoRA (--mode all)
```


## LOO protocol and checkpoint selection

Per fold, with the 7 clean KITTI drives `01 04 06 07 08 09 10`:

| | train | validation | test |
|---|---|---|---|
| TLIO / Deep KF / Tartan | 5 drives | 1 drive (inner) | held-out drive |
| deep_iekf | 6 drives | none (final epoch) | held-out drive |
| classical + smoothers | 5 drives | 1 drive (report-only) | held-out drive |

The held-out drive is **never loaded during training**. The inner validation
drive is chosen by `dl_filters._validation.inner_split`: the next drive after the
held-out one, cyclically, skipping seq 04 (29.7 s against 113-537 s for every
other drive — too few windows to select on).

### What selects the deployed epoch

The **journal metric**, closed-loop on the inner-validation drive with the
standard 40 s / 60 s outage:

    J = ATE_outage / 1 m  +  t_rel / 1 %  +  r_rel / 1 (deg/km)

This is the same objective `ins_genetic_cv.py` minimises for the seven classical
filters, which is what makes the DL and classical rows comparable rather than
merely adjacent.

It replaced one-step NLL validation loss, for two reasons:

1. **NLL cannot see what the tables report.** These models train OUTAGE-FREE by
   design — outages are simulated only at evaluation. A one-step density score
   on clean data therefore carries no information about dead-reckoning
   behaviour during a GNSS outage, which is the headline result.
2. **It measurably mattered.** Deep KF's leave-one-out outage mean landed at
   86 m or 557 m depending only on which epoch that blind criterion happened to
   pick — and the 86 m version was the one selected on the *test* drive.

`--val-metric-every K` sets how often J is scored, i.e. the granularity at which
the epoch can be chosen (not a logging interval). It is 2 for TLIO/Tartan and 5
for Deep KF. A fold where J is `inf` at every evaluated epoch **raises** rather
than deploying an unselected checkpoint.

The LR schedule still follows inner-val NLL: a plateau detector needs a smooth
every-epoch signal, and only *selection* moved.

