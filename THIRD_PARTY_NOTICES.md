# Third-party components

This repository does not vendor upstream model libraries or claim ownership of
their weights. Check the installed version's license and notices before redistribution.

| Component | Use | Upstream license/source |
|---|---|---|
| NumPy | numerical arrays | BSD-3-Clause, https://github.com/numpy/numpy |
| SciPy | Savitzky–Golay filter/statistics | BSD-3-Clause, https://github.com/scipy/scipy |
| pandas | data tables | BSD-3-Clause, https://github.com/pandas-dev/pandas |
| scikit-learn | optional retraining | BSD-3-Clause, https://github.com/scikit-learn/scikit-learn |
| PyTorch | phase model | BSD-style, https://github.com/pytorch/pytorch |
| pytorch-crf | CRF layer | MIT, https://github.com/kmkurn/pytorch-crf |
| LightGBM | reference mapping | MIT, https://github.com/microsoft/LightGBM |
| OpenCV | optional video processing | Apache-2.0 for the referenced version, https://github.com/opencv/opencv |
| FFmpeg | external video encoding tool | build-dependent LGPL/GPL; binary is not redistributed, https://ffmpeg.org/legal.html |
| MMPose / MMDetection / MMCV / MMEngine | optional RGB inference | Apache-2.0 source; check checkpoint/model terms, https://github.com/open-mmlab |

The supplied phase checkpoints were trained by this project from initialization;
they are not redistributed RTMPose/RTMDet pretrained weights. The release does not
bundle Ultralytics, MotionBERT, V-JEPA, TabPFN, or T-JEPA dependencies from the
historical workspace. Historical experiment names describe provenance, not a
claim that every such experiment is reproduced here.
