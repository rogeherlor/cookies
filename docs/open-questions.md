# Open questions and known caveats

Things a reader or reviewer may reasonably probe. None of these is a bug; each is a
choice, and a few are simply unverified.

## Methodological choices

**Deep IEKF trains on six sequences, the other three DL filters on five.** The nested
split spends one sequence on inner validation. Deep IEKF keeps its upstream protocol
(final epoch, no validation split) and therefore keeps all six. It is also the
best-performing DL model, and some of that margin may be the extra drive rather than
the method. Giving it the same 5/1/1 split would settle the question at the cost of
departing from the original recipe.

**DL filter noise parameters are inherited, not tuned.** `Q`, `R` and `P0` for the DL
filters are copied from the best classical filter of the same fold
(`_load_tuned_params`), whereas each classical filter gets its own GA search. Whether
that helps or hurts the DL rows has not been measured.

**Classical filters select on training cost; DL filters select on validation cost.**
Both exclude the test sequence, so neither leaks, but the two protocols are not
identical.

**Ground truth is FGO-Batch**, a smoothed estimate from the same GNSS/IMU data rather
than an independent reference. ATE is therefore error against a smoother's output.

**No consistency gate.** The ANEES gate was dropped when tuning moved to a pure
accuracy objective, so nothing currently rejects an over-confident covariance.

**TLIO is applied well outside its domain** — pedestrian head-mounted to vehicle. Its
246 m outage ATE on sequence 01 is a domain-mismatch result, not a defect.

**Single-sample results.** Training uses a fixed seed, so runs are reproducible but
there is no run-to-run variance reported. This does not affect the CPU-vs-Hailo
quantisation deltas, which share a checkpoint, but it does affect between-filter
comparisons in the accuracy tables.

## Not verified: fidelity to the source publications

The deployment pipeline has been audited end to end. Whether each filter faithfully
reproduces its source paper has not been. The known interpretive choices:

- **Deep KF** — the LSTM replaces the analytic prior only during a genuine outage,
  not on every tick. This is central to every Deep KF number and rests on a reading of
  Hosseinyalamdary (2018) rather than an explicit statement in the paper.
- **TLIO** — 50 epochs with a 10-epoch phase switch, against the paper's own schedule;
  window 100 upsampled to 200; GNSS added, which the original does not have.
- **Tartan IMU** — LoRA rank 8 on the transformer trunk with the backbone frozen; not
  checked against the paper's adaptation protocol. Separately, the LoRA folds were
  trained through `_TartanIMUBackbone.forward()` while it was missing a transpose
  (a bare `reshape` on a `(B, T, S, C)` tensor), so the adapters learned against
  scrambled per-step features. The reshape is now correct, but the folds have not been
  retrained against it.
- **Deep IEKF** — causal/online variant of AI-IMU, 400 epochs.

Checking these means reading four papers against four implementations. It is separate
work, and it is the item most likely to move a headline number.

## Superseded artefacts

Kept for traceability. None of them should be used for a result.

| Path | Why it was superseded |
|---|---|
| `full_benchmark_results/_superseded_2026-08-09_pi/` | leaking HEFs, test-selected checkpoints, no contamination check, inconsistent affinity |
| `artifacts/_pre_inner_val_backup/` | checkpoints selected on the test sequence |
| `artifacts/_nll_selected_backup/` | selected on inner-validation NLL, before selection moved to the journal metric |
| `scripts/positioning/hailo/*/_pre_per_fold/` | the original single all-folds binaries |
