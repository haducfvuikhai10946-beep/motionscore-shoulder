# MotionScore: phase-aware shoulder-raise profiling

[中文说明](README_zh.md)

Research software for seated bilateral shoulder raises from monocular video.
The pipeline combines RTMPose body coordinates, a BiLSTM–CRF phase parser,
thirteen kinematic descriptors, optional cross-cycle mapping, and a transparent
reference-deviation score with empirical percentile ranking.

This release corresponds to the manuscript **Phase-Aware Kinematic Profiling and
Reference-Deviation Scoring for Action Quality Assessment of Seated Shoulder
Raises From Monocular Video**, revision dated 2026-10-21. No publication DOI is
claimed. See [reproducibility](docs/REPRODUCIBILITY.md) and the machine-readable
[validation reports](docs/validation/).

## Scope

The reproducible core contains Identity, Ridge, ElasticNet and the paper's
LightGBM reference, including the full-development and five fold-specific
scorers and phase checkpoints. Other neural/tabular benchmark backends,
institutional fitness-record linkage and hardware latency experiments are
outside this release's verified reproduction scope. The common thresholds
were inherited from the historical T-JEPA development branch and remain frozen;
this package does not re-derive them by training a new T-JEPA encoder.

MotionScore is a task-specific index of deviation from the study reference
population. It has not been validated as a clinical diagnosis, expert quality
rating, general fitness estimate or cross-session reliability measure.

## Install

### Small example without model downloads

After cloning the repository, the included Ridge example needs only the base
dependencies. Its goldens were exported from the original frozen program.

```bash
python -m pip install -e .
motionscore score-features --input examples/example_features.csv --models examples/models --backend Ridge --output results/example.json
motionscore score-pose --input examples/example_pose.csv --labels examples/example_labels.npz --models examples/models --backend Ridge --output results/example_pose.json
```

Compare with `examples/example_expected.json`. The supplied labels allow this
example to run without PyTorch, MMPose, a GPU or full-dataset downloads.

### Full reproduction environment

Python 3.10 is the tested reference. Install in a new environment:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[phase,trees,train,test]"
python scripts/download_release.py --kind models
python scripts/download_release.py --kind dataset
```

For reference versions see [requirements-reference.txt](requirements-reference.txt).
PyTorch wheels depend on your operating system and CUDA runtime. A CPU install
can run all cached-pose reproduction steps; MMPose is only needed for RGB input.

## Reproduce the released results

```bash
python -m pytest -q
python scripts/verify_release.py --data downloads/dataset-v1.0.0 --models downloads/models-v1.0.0 --phases
python scripts/train_backends.py --data downloads/dataset-v1.0.0 --models downloads/models-v1.0.0 --backend Ridge
python scripts/train_backends.py --data downloads/dataset-v1.0.0 --models downloads/models-v1.0.0 --backend LightGBM
```

The verifier compares outputs against **historical-program goldens**, checks
participant separation, reconstructs all thirteen descriptors for 316 recordings,
and reports participant-level out-of-fold MSE and ICC. With `--phases`, it also
checks the full-development phase model against cached labels. Fold-specific
phase predictions are covered by a separate audit described in the reproduction
guide. Numerical tolerances and actual coverage are recorded in the reports.

## Score your data

`input_features.csv` must contain the thirteen named columns in
[paper_protocol.json](configs/paper_protocol.json). Units and formulas are in
[FEATURES.md](docs/FEATURES.md).

```bash
motionscore score-features --input input_features.csv --models downloads/models-v1.0.0/full --backend LightGBM --output results/scores.json
motionscore score-pose --input body_points.csv --models downloads/models-v1.0.0/full --backend Ridge --output results/pose_score.json
```

Pose CSVs contain eight body joints, pixel coordinates and relative seconds at
20 fps. No odd/even split is needed for full-recording inference; the CLI reports
full/odd/even rows together for inspection. The current CLI requires at least
two retained cycles so those diagnostic rows are defined. Outputs include D,
MotionScore, reference band, budget-weighted contributions and the full-model
reference percentile. Higher MotionScore means lower reference deviation.

For new videos see [VIDEO.md](docs/VIDEO.md). The published videos have opaque
head redaction and selective masking of additional people, with room/equipment
background retained, no audio and random study identifiers. They are
altered visual data; rerunning a pose estimator on them is not guaranteed to
recover the original cached coordinates or paper results.

## Data and licenses

The dataset contains 242 development and 74 comparison recordings with fixed
participant-level folds. Data assets are documented in [DATA_CARD.md](docs/DATA_CARD.md).
Random IDs replace original identifiers; no identity map or fitness workbook is
distributed. Body motion may still carry identifying information; full anonymity
is not asserted.

Original project code and project-trained weights: Apache-2.0. Study data:
CC BY 4.0, subject to the rights statement in the data card. Upstream dependencies
and optional pretrained pose models retain their own terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Cite the software using
[CITATION.cff](CITATION.cff).
