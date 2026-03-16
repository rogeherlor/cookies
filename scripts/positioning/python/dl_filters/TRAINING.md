# Deep Learning Filters — Training Guide

Run these steps **before** calling `ins_compare.py` or `ins_runner.py` with any
DL filter. All commands are run from `scripts/positioning/python/`.

---

## Overview

| Filter | Paper | Script | Output artifact |
|--------|-------|--------|-----------------|
| IEKF AI-IMU | Brossard et al., IEEE TIV 2020 | `dl_filters/deep_iekf/train_ai_imu.py` | `artifacts/deep_iekf/fold_<SEQ>.p` |
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

## 1. IEKF AI-IMU (Brossard et al. 2020)

Pre-trained weights ship with the `external/ai-imu-dr/` clone and are searched
automatically. No mandatory training step — the filter works out-of-the-box.

**Optional LOO re-training** (improves per-sequence accuracy):

```bash
# From scripts/positioning/python/
for SEQ in 01 04 06 07 08 09 10; do
    python dl_filters/deep_iekf/train_ai_imu.py --val-seq $SEQ
done
```

Weights are saved to `artifacts/deep_iekf/fold_<SEQ>.p`.
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

The runner searches for weights in this order:
1. `TARTAN_IMU_WEIGHTS` environment variable
2. `external/tartan_imu/tartan_imu_base.pt`
3. `artifacts/tartan_imu/tartan_imu_base.pt`

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

# TLIO
for SEQ in $SEQS; do
    python dl_filters/tlio/train_tlio.py --mode loo --val-seq $SEQ --epochs 200
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
├── deep_iekf/
│   ├── fold_01.p  …  fold_10.p     # AI-IMU (optional re-train)
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
