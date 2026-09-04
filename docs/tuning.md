# Tuning the classical filters

The six classical filters have 13–15 free noise parameters each (process noise Q,
measurement noise R, initial covariance P, Gauss-Markov decay constants). The defaults
in the filter modules are conservative and dataset-agnostic — good enough to get a
trajectory out, not good enough to compare filters against each other. Tune first,
then compare.

Tuning is a differential-evolution search driven by `ins_genetic_cv.py`. Results are
written to `filter_params.json` and picked up automatically by `ins_compare.py`.

All commands run from `scripts/positioning/python/`.

## Filters covered

| Key | Algorithm |
|-----|-----------|
| `esekfg_vanilla` | Error-state EKF, Groves formulation, GNSS only |
| `esekfg_enhanced` | Groves ES-EKF + NHC + ZUPT |
| `esekfs_vanilla` | Error-state EKF, Solà formulation, GNSS only |
| `esekfs_enhanced` | Solà ES-EKF + NHC + ZUPT |
| `iekf_vanilla` | Left-invariant EKF, GNSS only |
| `iekf_enhanced` | Left-invariant EKF + NHC + ZUPT |

`imu_only` has no parameters. The DL filters (`tlio`, `deep_kf`, `tartan_imu`,
`iekf_ai_imu_online`) are trained rather than tuned — see [training.md](training.md).

## The cost function

`ins_cost.py` defines the objective, a sum of three normalised metrics:

```
J = ATE_outage / 1 m  +  t_rel / 1 %  +  r_rel / 1 deg/km
```

The normalisers are operational thresholds, not fitted weights: 1 m is the NHTSA/ISO
lane-keeping accuracy figure, and the two relative-error cutoffs are the KITTI
leaderboard "elite" thresholds from Geiger et al. (2012). A component value of 1 means
one unit of acceptable degradation.

## Running a full LOO sweep

This is what produced the published parameters. `run_genetic_loo.sh` calls
`ins_genetic_cv.py` once per held-out sequence, tuning on the other six:

```bash
./run_genetic_loo.sh                                       # all 7 folds, all 6 filters
./run_genetic_loo.sh --filters esekfs_enhanced iekf_enhanced
./run_genetic_loo.sh --maxiter 60 --popsize 20             # larger budget
```

Defaults are `--maxiter 15 --popsize 10 --workers -1` (all cores). Budget roughly
6–10 hours per fold on a six-core machine; seven folds is an overnight run.

The clean KITTI sequences are `01 04 06 07 08 09 10`. Sequences `00`, `02` and `05`
have ~2 s data gaps and `03` has no raw data, so none of them are used.

## Running the tuner directly

```bash
python ins_genetic_cv.py --held-out 2011_09_30_drive_0028_extract     # one fold (seq 08)
python ins_genetic_cv.py iekf_enhanced --held-out <drive> --maxiter 30
python ins_genetic_cv.py                                              # no LOO split; CV aggregate
```

`--held-out` takes the full KITTI drive name, not the sequence number. The mapping is
in `KITTI_CLEAN_DRIVES` at the top of `ins_genetic_cv.py`; `run_genetic_loo.sh` does
the translation for you, which is the main reason to prefer it.

Other flags: `--2d` / `--3d` select the motion model, `--outages` and `--val-outages`
set how many outage windows are averaged per training and validation drive, `--seed`
defaults to 42.

## Where the results go

`filter_params.json`, in the same directory, keyed by filter, then mode, then split:

```json
{
  "iekf_enhanced": {
    "2d": {
      "__loo_held_2011_09_30_drive_0028_extract__": { "Qpos": 0.01, "Rpos": 2.5, "...": "..." },
      "__cv_kitti__": { "...": "..." }
    },
    "3d": { "...": "..." }
  }
}
```

`filter_params.get(key, mode_3d, dataset_name)` resolves in priority order:

1. `__loo_held_<dataset_name>__` — the matching LOO fold
2. `<dataset_name>` — a single-dataset tune
3. `__cv_kitti__` — the cross-validation aggregate

So `ins_compare.py --test-seq 08` automatically picks up the fold that never saw
sequence 08.

## Notes

- Re-tune if you change the outage window. The objective is defined against the
  window in `ins_config.py` (`OUTAGE_START`, `OUTAGE_DURATION`).
- 2D and 3D are separate searches. Tune the mode you intend to evaluate, or both.
- Results are saved after each filter finishes, so an interrupted run can be resumed
  by listing only the filters that are still missing.
