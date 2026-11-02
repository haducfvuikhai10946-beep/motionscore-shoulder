# Fixed feature definitions

All descriptors are evaluated per retained cycle and aggregated by the median.
L/R are shoulder angles; v and a are whole-recording smoothed angle derivatives.
The exact implementation is in `src/motionscore/features.py`.

| Feature | Definition | Unit | Penalty anchor |
|---|---|---|---|
| F1 | Mean of left/right cycle angular ranges | degree | reference median |
| F2 | Absolute difference of left/right ranges | degree | zero |
| F3 | Mean absolute acceleration during raising, both sides | degree/s² | reference median |
| F4 | Mean absolute acceleration during lowering, both sides | degree/s² | reference median |
| F5 | Mean absolute cycle velocity, both sides | degree/s | reference median |
| F6 | Mean absolute cycle acceleration, both sides | degree/s² | reference median |
| F7 | Peak-hold duration / complete cycle duration | ratio | reference median |
| F8 | mean(abs(vL-vR)) / mean(abs(vL)+abs(vR)) | ratio | zero |
| F9 | RMS trunk tilt from image vertical | degree | zero |
| F10 | Population SD of signed cycle trunk tilt | degree | zero |
| F11 | Shoulder midpoint horizontal p95-p05 / recording median shoulder width | ratio | zero |
| F12 | Hip midpoint horizontal p95-p05 / recording median shoulder width | ratio | zero |
| F13 | Pooled bilateral p95 absolute velocity / pooled mean absolute velocity | ratio | reference median |

The reference median is the observed study norm, not a clinical optimum.
For example, larger-than-reference F1 can lower the score under its two-sided
penalty. Parsing errors can influence F3/F4/F7; precise manual/automatic phase
equivalence is not assumed.
