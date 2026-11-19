# Multimodal Sensor Uncertainty Under Progressive Sensor Degradation

Compact handover of the final paper code and saved evidence for the multimodal
RADIATE detector study. The manuscript is included as IEEE Paper.pdf.

## Contents

- src/ contains the RADIATE dataset adapter, sensor preprocessing, degradation,
  detector, training, and evaluation code.
- config/default-calib.yaml and configs/experiment_protocol.yaml record sensor
  geometry and the frozen experimental protocol.
- scripts/reproduce_results.py regenerates the paper tables from saved run-level
  evaluations. scripts/verify_handover.py checks the included evidence against
  the canonical tables and manifest.
- outputs/assets/ contains the canonical figures, tables, method factsheet, and
  final multiseed manifest. Other small files under outputs/ are the saved
  run-level inputs and deterministic provenance needed for table verification.

## Verify the saved paper results

From the repository root, with Python 3.10 or newer:

    python scripts/verify_handover.py
    python scripts/reproduce_results.py --output-dir outputs/reproduced

The first command compares the included run-level records with the canonical
CSV, JSON, and LaTeX tables, checks the deterministic source SHA-256, and
confirms the manifest's output files are present. It does not verify the
historical archive, which is not part of this compact handover. The original
reproduce_results.py --check option expects that archive; use
verify_handover.py for this package. Regeneration writes into a separate
ignored directory and does not change canonical assets.

## Model code and external assets

Install direct Python dependencies with:

    python -m pip install -r requirements.txt

The RADIATE image sequences and the selected epoch-3 checkpoint_best.pth are
not included. The original runtime and end-to-end multiseed inference driver
are also unavailable in this package. Obtain the authorized data and exact
checkpoint from the project owner before attempting model inference. This
handover reproduces published aggregate tables from saved run-level
evaluations; it does not establish a one-command neural inference replay.

## Interpretation boundaries

The 87 clean validation frames (627-713) are from the same recording as the
622 training frames and were used for checkpoint selection. They are not an
independent sequence-level test set. The fog_6_0 sequence was held out from
training and selection. Three degradation seeds represent stochastic
inference-time realizations with one frozen model, not three trained models.

Confidence is the class-specific sigmoid CenterNet heatmap score. Calibration
metrics are computed over evaluated predictions after ignored-region
exclusions; missed objects appear in recall and mAP instead. The presence of a
local checkpoint or a saved table does not independently establish the
provenance of the published neural run.