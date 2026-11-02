# Reproduction guide

## Reference state

The authoritative numerical sources are the 2026-10-21 manuscript revision's
frozen rescoring program, fixed thirteen-feature caches, isolated development-only
phase checkpoints and native mapping heads. The exporter replaced historical
Python object graphs with JSON, LightGBM text and tensor-only state dictionaries.
Identity metadata, local paths and training IDs embedded in those objects are
not serialized into the public models. No model was retrained during export.

## Three distinct checks

1. **Frozen numerical reproduction.** `verify_release.py` checks all 316 poses,
   aggregates and four scoring paths against historical outputs. It uses the
   exact published development/comparison and five-fold splits. `--phases`
   checks the full-development phase checkpoint on all 316 recordings.
2. **Mapping training reproduction.** `train_backends.py` refits a specified
   mapping on the chronological odd/even aggregates of development participants
   only, including each original outer fold, then compares predictions with the
   frozen model. Scaling, reference blocks and thresholds stay fixed to their
   released fold-specific values. This tests head fitting conditional on those
   reference values; it does not repeat historical feature discovery or threshold
   selection.
3. **Frontend retraining.** `train_phase.py` implements the original six-epoch,
   two-layer BiLSTM with CRF pseudo-supervision. The supplied manual labels are
   evaluation-only. Different PyTorch, BLAS, CUDA or hardware versions can change
   trained weights; use frozen checkpoints for numerical comparison. Training
   execution coverage is disclosed in the validation reports.

Example frontend retraining:

```bash
python scripts/train_phase.py --data downloads/dataset-v1.0.0 --fold 0 --device cpu --output results/fold0_phase.pt
```

The original CRF used a known-label mask rather than a separate contiguous
sequence mask. This behavior is retained for faithful reproduction, including
unknown labels replaced with zero only where masked. The deployment decoder is
the original unmasked Viterbi path. Replacing it with a hard legal-transition mask
is a separate method and can change results.

The five held-out phase checkpoints reproduce all 367,840 cached frame labels
on the reference CUDA runtime. CPU inference differs at one frame in fold 3;
the strict CPU fold-check script records this discrepancy and exits nonzero.
This is a measured device-dependent Viterbi boundary difference. The frozen
full-development checkpoint matches all 480,320 labels on the tested CPU runtime.
Use `--device cuda:0` with `verify_phase_folds.py` for the original fold-inference
device. Cached fold features remain the input to exact numerical table reproduction.
Full six-epoch neural retraining has not been rerun for release certification;
the training path is covered by a finite-loss/gradient/update smoke test.

## Statistical unit and leakage controls

Participants are the resampling and split units; cycles and odd/even pairs from
one participant remain in one fold. Each outer feature table was produced with
the corresponding outer-training phase parser. The `full242` table uses the
full-development parser and must not substitute for fold-specific tables during
out-of-fold evaluation. Comparison participants never train mapping heads.
The 74-person cohort was examined in historical exploratory work; its status
does not establish an untouched prospective external validation.

## Exact conventions

- Input time: uniform 20 fps; pixel geometry retained.
- Angle: shoulder-to-elbow versus shoulder-to-hip; denominator epsilon 1e-9.
- Phase input: bilateral and mean angles, degrees/frame gradient; per-recording
  standardization using population standard deviation plus 1e-6; float32.
- Derivatives: whole-recording Savitzky–Golay window 9, polynomial order 2;
  NumPy gradients multiplied by fps, then differentiated again for acceleration.
- Cycles: consecutive rising/peak/falling/valley runs; five zero-based,
  half-open boundaries; 10-frame end margins.
- Aggregation: coordinate-wise cycle medians; chronological odd/even subsets.
- Scaling: frozen training median/IQR with historical floor values; clip [-6,6].
- Mapping predictions are unclipped for MSE and clipped for scoring.
- Score: `100 / (1 + D)`; D is the sum of budget-weighted block penalties.
- F1 amplitude uses a two-sided reference anchor. Symmetry and trunk penalties
  use the physical zero anchor. Published cutpoints are retained at full precision.
- Percentile: right-continuous count of same-backend reference scores <= query,
  divided by 242. Percentiles from different backends use their own score reference.

## Tolerances and exclusions

Feature and score reproduction accepts absolute error below 1e-9, with exact
band equality. Mapping retraining accepts prediction error below 1e-8 in robust
standardized feature units. Phase prediction comparisons report frame disagreement
counts; only zero disagreements passes the fixed-checkpoint check.

This release does not reproduce all twelve tabular benchmark families, the
original T-JEPA training, institutional physical-fitness associations, full
feature-discovery procedures, clinical validation, or GPU latency tables.
Processed RGB videos are not numerically equivalent to the unredacted acquisition;
the cached body coordinates are the paper-reproduction input.
