# Classical Filters — Parameter Tuning Guide

Run `ins_genetic_fast.py` **before** using `ins_compare.py` to obtain
optimised parameters for each classical filter. Without tuning, all filters
use generic defaults that are not suitable for performance comparisons.

All commands are run from `scripts/positioning/python/`.

---

## Why tuning matters

The six classical filters have 13–15 free noise parameters (process noise Q,
measurement noise R, initial covariance P, Gauss-Markov decay constants).
Default values are intentionally conservative and dataset-agnostic.

`ins_genetic_fast.py` runs differential evolution to minimise position RMSE
during GPS outage. Tuned parameters are saved to `filter_params.json` and
loaded automatically by `ins_compare.py` at startup.

> **Rule:** Never compare filter outputs using default parameters.
> Always tune first, then compare.

---

## Filters that need tuning

| Filter key | Algorithm | Tunable params |
|------------|-----------|----------------|
| `ekf_vanilla` | Euler EKF, GPS only (Groves 2013) | Q, R, P, β |
| `ekf_enhanced` | Euler EKF + NHC + ZUPT | Q, R, P, β, R_nhc, R_zupt |
| `eskf_vanilla` | Quaternion ESKF, GPS only (Solà 2017) | Q, R, P, β |
| `eskf_enhanced` | Quaternion ESKF + NHC + ZUPT | Q, R, P, β, R_nhc, R_zupt |
| `iekf_vanilla` | Left-invariant EKF, GPS only | Q, R, P, β |
| `iekf_enhanced` | Left-invariant EKF + NHC + ZUPT | Q, R, P, β, R_nhc, R_zupt |

**Filters that do NOT need genetic tuning:**
- `imu_only` — no filter, no parameters
- `iekf_ai_imu`, `tlio`, `deep_kf`, `tartan_imu` — DL filters (see `dl_filters/TRAINING.md`)

---

## Script: `ins_genetic_fast.py`

### Quick reference

```
usage: ins_genetic_fast.py [-h] [--seq SEQ_ID] [--3d] [--2d]
                           [--maxiter N] [--popsize N]
                           [filter_key ...]

Positional args:
  filter_key     Which filters to tune (default: all 6 classical filters).
                 Example: eskf_enhanced iekf_enhanced

Optional args:
  --seq SEQ_ID   KITTI held-out sequence ID for LOO split (e.g. 01, 08).
                 When omitted, uses the single dataset in ins_config.py.
  --3d           Tune 3D mode only  (MODE_3D=True)
  --2d           Tune 2D mode only  (MODE_3D=False)
  --maxiter N    DE generations [default: 10]
  --popsize N    DE population size [default: 8]
```

### Quality tiers

| Tier | Flags | Evals / filter | Wall time / filter |
|------|-------|----------------|--------------------|
| Fast (exploration) | *(defaults)* | ~1 200 | 1–3 min |
| Balanced | `--maxiter 20 --popsize 10` | ~3 000 | 5–10 min |
| Production | use `ins_genetic_cv.py` (40 gen, full LOO avg) | ~9 000 | 20–40 min |

---

## Recommended workflow

### Option A — Tune on a single sequence (quick start)

Edit `ins_config.py` to set the desired sequence, then:

```bash
python ins_genetic_fast.py
```

Results are saved under the `dataset_name` key in `filter_params.json`.

### Option B — Full LOO sweep (recommended for fair comparison)

Run for every clean held-out sequence. Each invocation trains on the other 6
clean sequences and saves results under the `__loo_held_<SEQ>__` key.

```bash
for SEQ in 01 04 06 07 08 09 10; do
    python ins_genetic_fast.py --seq $SEQ
done
```

`ins_compare.py --test-seq <SEQ>` then automatically loads the matching
`__loo_held_<SEQ>__` parameters.

### Option C — Tune a single filter quickly

```bash
python ins_genetic_fast.py --seq 08 eskf_enhanced
```

### Option D — Tune 2D and 3D modes separately

```bash
for SEQ in 01 04 06 07 08 09 10; do
    python ins_genetic_fast.py --seq $SEQ --2d
    python ins_genetic_fast.py --seq $SEQ --3d
done
```

The current `MODE_3D` setting in `ins_config.py` selects which set of params
`ins_compare.py` uses.

---

## Where results are stored

`filter_params.json` — same directory as the scripts. Structure:

```json
{
  "eskf_enhanced": {
    "2d": {
      "__loo_held_08__": { "Qpos": ..., "Rpos": ..., ... },
      "kitti_08":        { ... }
    },
    "3d": { ... }
  },
  ...
}
```

`ins_compare.py` calls `filter_params.get(key, mode_3d, dataset_name)` which
looks up params in priority order:
1. `__loo_held_<dataset_name>__` — LOO fold (best for fair comparison)
2. `<dataset_name>` — single-dataset tune
3. `__cv_kitti__` — full CV aggregate

---

## Complete setup checklist (fresh environment)

```bash
# Step 1 — tune all classical filters with LOO split
for SEQ in 01 04 06 07 08 09 10; do
    python ins_genetic_fast.py --seq $SEQ
done

# Step 2 — train / download DL filter weights
# (see dl_filters/TRAINING.md)

# Step 3 — run comparison
python ins_compare.py --test-seq 08
```

---

## Tips

- **Outage window matters:** Tuning optimises for the outage scenario defined in
  `ins_config.py` (`OUTAGE_START`, `OUTAGE_DURATION`). If you change the outage
  window, re-tune.

- **MODE_3D matters:** 2D and 3D filters need separate tuning runs. Use `--2d`
  and `--3d` flags to tune both, or tune only the mode you intend to evaluate.

- **Interrupted runs:** Results are saved after each filter completes, so a
  partial run can be resumed by listing only the remaining filter keys:

  ```bash
  python ins_genetic_fast.py --seq 08 iekf_vanilla iekf_enhanced
  ```

- **Faster iterations:** Reduce `MAXITER` and `POPSIZE` at the top of
  `ins_genetic_fast.py` for exploration, then run `ins_genetic_cv.py` for
  production-quality parameters.
