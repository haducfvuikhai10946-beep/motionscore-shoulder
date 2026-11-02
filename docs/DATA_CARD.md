# Dataset card

## Contents and task

316 university participants, one seated bilateral shoulder-raise recording each;
242 development participants and 74 comparison participants. Monocular acquisition
is 640×480 at 20 fps. Public identifiers are random and preserve the original
participant-level experimental assignments. `participants.csv` supplies the
development/comparison split and held-out fold; `folds.json` supplies ordered
training and test lists. Original recording dates, filenames and identity maps
are not released.

`poses/*.npz` contains `values` and Unicode `columns` arrays for eight shoulder,
elbow, wrist and hip points, confidence values and relative seconds. Load with
`np.load(path, allow_pickle=False)`. Pixel coordinates preserve original geometry.
`labels/*.npz` contains frozen full-development-parser labels. `training/*.npz`
contains phase inputs `X`, rule-supervised training labels `y`, and independent
manual labels where available (`manual`, unknown=-1). Missing manual labels are
unknown, not inferred negatives. Manual labels must not enter training.

`features/full242.csv`, `fixed74.csv`, and `outer0.csv`–`outer4.csv` contain full,
odd and even cycle medians; some files also retain boundary-shift diagnostics.
Use only `shift_frames == 0` for the primary analysis. Never replace an outer
feature table with the full-development table for a held-out comparison.
`expected_*.csv` contains independent historical-program scoring goldens.

## Video transformation

Released videos retain the room/equipment background and have an opaque white
head rectangle following shoulder geometry, selective boxes covering additional
people, no audio and
new filenames. The video encoder retains frame count, 20 fps and resolution;
re-encoding changes RGB pixels. Metadata is newly generated without source
timestamps. A documented review of redacted contact sheets complements structural
checks; it is not a guarantee that every possible identifying cue was removed.
Additional-person locations are sampled with RTMDet at one-second intervals;
neighboring detections are expanded by a 26-pixel temporal margin. This is an
automated redaction aid rather than exhaustive manual per-frame annotation.
The transformation can obscure joints if hands cross the head region. Use the
original-analysis body coordinates for paper numerical reproduction.

The project owner confirmed authorization to publicly distribute the processed
videos on 2026-11-02. This records the owner's instruction, not an independent
audit of institutional or participant documents. Names, student numbers,
institutional fitness workbooks and the private identifier map are excluded.

## Access, rights and limitations

Data assets are versioned GitHub Release downloads linked by the repository
asset manifest and SHA-256 checksums. No DOI is claimed unless an archival record
is actually created. Dataset license: Creative Commons Attribution 4.0
International, https://creativecommons.org/licenses/by/4.0/.
The publisher is responsible for the rights to distribute these study records;
the license cannot grant rights the publisher does not possess.

Body shape, clothing and motion can retain identifying information. Random IDs
and head masks do not guarantee irreversible anonymization. Please do not attempt
to re-identify participants; this is an ethical request rather than an additional
condition to the CC BY license. The sample and fixed apparatus limit population,
camera and exercise generalization. Institutional fitness outcomes are excluded;
the package cannot reproduce those record-linkage analyses.
