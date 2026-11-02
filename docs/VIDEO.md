# Optional RGB frontend

The tested numerical reproduction path starts from the released body coordinates.
RGB inference requires the OpenMMLab stack in a separate environment; it is not
installed by the base package. The original stack used torch 2.1.0+cu118,
mmcv 2.1.0, mmdet 3.2.0, mmpose 1.3.2 and mmengine 0.10.7.
Setuptools must remain below 81 for this legacy MMEngine environment; the
release audit used setuptools 80.9.0. Use compatible
platform-specific MMCV wheels; do not assume a CPU-only base install supplies them.

The original pose model is RTMPose-m, AIC/COCO pretrained, input 256×192; detector
is RTMDet-nano COCO-person, input 320. Obtain configs and checkpoints from
the upstream MMPose model zoo. They retain their upstream license and provenance
and are not included as project-trained weights. The checkpoint/config hashes
used for the local video audit are recorded in its validation report.

```bash
python -m motionscore.video --input recording.mp4 --output results/body_points.csv --det-config /path/to/rtmdet_nano_320-8xb32_coco-person.py --det-weights /path/to/detector.pth --pose-config /path/to/rtmpose-m_8xb256-420e_coco-256x192.py --pose-weights /path/to/pose.pth --device cuda:0
motionscore score-pose --input results/body_points.csv --models downloads/models-v1.0.0/full --output results/video_score.json
```

The frontend applies person score >0.3, NMS IoU 0.3, and the historical combined
60/20/20 area, image-center and previous-box IoU selection. It stops if detection
fails instead of inventing coordinates. Use 20 fps recordings. New data with
missing poses need an explicitly evaluated cleaning procedure; silent interpolation
or timing conversion would change the protocol.

Head and selective additional-person redaction change detector and pose input. Equality to the paper
coordinates is not claimed for public processed videos. See the video audit
report for measured differences and actual execution coverage.

The release audit executed this entry point on one scene-preserving processed
recording (1,520 frames). The median coordinate difference from the historical
cache was 0.706 pixels, and the LightGBM MotionScore changed by +0.485 points;
both paths retained seven cycles. This is one execution example, not evidence
of cohort-wide redaction equivalence. See `docs/validation/video_entry_audit.json`.
