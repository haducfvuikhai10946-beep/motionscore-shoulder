# MotionScore v1.0.0

Reproducible core of the 2026-10-21 shoulder-raise manuscript revision.

- Identity, Ridge, ElasticNet and LightGBM scoring with frozen references,
  full-development weights and five fold-specific configurations.
- Tensor-only BiLSTM–CRF checkpoints; no project pickle/joblib loading required.
- De-identified eight-joint coordinates, phase training/evaluation labels,
  fixed participant splits and independent historical numerical goldens.
- Scene-preserving processed videos with opaque head and selective
  additional-person masking, removed audio and randomized filenames.
- Download checksums, small offline example, unit tests, fresh-environment
  verification, mapping retraining and an explicit reproducibility guide.

Numerical checks reproduce the thirteen descriptors, scores and all four
out-of-fold mapping metrics. Ridge, ElasticNet and LightGBM head retraining
matches the original predictions. The original CUDA fold phase predictions
match all 367,840 cached labels. CPU inference has one different frame in fold 3;
that measured difference is retained in the reports. Full neural retraining,
all twelve tabular benchmark backends and fitness-record linkage are outside
the verified release scope.

Processed RGB inputs can change estimated coordinates and scores. Cached body
poses are the paper-numerical-reproduction inputs. MotionScore is a task-specific
reference-deviation index, not an independently validated clinical assessment.
