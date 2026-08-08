# HEM-FM v4 local execution system

HEM-FM v4 is a fail-closed research pipeline for training and evaluating
multi-endpoint echocardiography models. It turns a gated training specification
into executable checks for hardware, DICOM calibration, patient-disjoint splits,
shortcut controls, architecture selection, PEFT/optimizer selection, specialist
training, external-data challengers, failure detection, and out-of-fold
pseudo-label generation.

This repository contains orchestration code and tests only. It does **not**
contain clinical data, patient identifiers, pretrained weights, feature caches,
gate evidence from restricted datasets, or a claim of clinical validation.

## Implemented workflow

- BF16-first CUDA runtime checks with deterministic multi-GPU smoke tests.
- Full-corpus DICOM inventory and physical calibration validation.
- Patient-level splitting before feature extraction, with fatal leakage checks.
- Caliper/overlay shortcut controls and clean-frame negative controls.
- Physical-space augmentation round-trip validation.
- Four-backbone frozen architecture ladder.
- LoRA, rsLoRA, DoRA, and PiSSA comparison with AdamW/Muon and EMA.
- Full ViT-L scalar schedule: frozen head, DoRA PEFT on the final four blocks,
  then low-rate selective unfreezing of the last block. Runs use compact,
  resumable per-seed checkpoints and never select on the locked test.
- Resumable four-backbone feature extraction on local storage.
- Three-seed, six-endpoint specialist training and grouped OOF pseudo-labels.
- CAMUS dense LV segmentation with DINOv3 and V-JEPA 2.1 challengers.
- TED full-cycle temporal-consistency training with the official CAMUS test
  identities kept sealed.
- MIMIC LV-volume-extension training with spatial calibration, biplane-first
  labels, heteroscedastic uncertainty, and physiological consistency losses.
- Unity anatomical landmark localization with physical-distance evaluation.
- EV9V nine-view routing with exam-disjoint validation and calibrated metrics.
- CardiacUDA, EchoCP, and CardiacNet external failure/abstention evaluation.
- Hash-verified, licence-aware research-dataset acquisition and promotion rules.
- Sequential evidence gates that keep the internal test set sealed.

External/vendor validation and prospective multi-reader repeat acquisition remain
separate evidence requirements. They cannot be replaced by local retrospective
training.

## What the system does

The codebase supports a deliberately modular echocardiography workflow:

- **Dense segmentation:** LV cavity, myocardium, and left atrium masks across a
  cine, with boundary and uncertainty outputs.
- **Temporal anatomy:** full-cycle segmentation support that rewards consistent
  anatomy across adjacent frames.
- **View routing:** calibrated classification across nine standard and modified
  echocardiographic views so endpoint heads receive appropriate evidence.
- **Landmark localization:** 31 anatomy landmarks with missing-label masking and
  evaluation in physical millimetres when spacing is available.
- **Scalar prediction:** patient-level ensembles for EF, LVEDV, LVESV, LVOT
  diameter, RV basal diameter, and aortic peak velocity.
- **Reliability controls:** aleatoric and epistemic uncertainty, input
  sufficiency checks, out-of-distribution scoring, risk-coverage curves, and
  fail-closed abstention.

The components are useful research building blocks, but the repository does not
ship a clinical product or authorize autonomous diagnosis.

## Reference local development run — 8 August 2026

The following aggregate results were produced on the local development and
validation partitions while all declared test identities remained sealed. They
are included for reproducibility and honest model selection; no private rows,
identifiers, images, weights, or feature caches are committed.

| Component | Development evidence | Decision |
| --- | ---: | --- |
| MIMIC-IV-ECHO inventory | 525,422 readable DICOMs; 507,434 with usable spatial calibration | Corpus and calibration audit passed |
| Core specialists | 72 runs: 4 frozen backbones × 6 endpoints × 3 seeds | Completed; retained as the current scalar baseline |
| CAMUS DINOv3 dense segmentation | 0.912 mean foreground Dice | Best dense LV development checkpoint retained |
| CAMUS V-JEPA 2.1 dense segmentation | 0.909 mean foreground Dice | Retained as a challenger |
| TED DINOv3 temporal segmentation | 0.889 mean foreground Dice | Selected for temporal support only |
| EV9V DINOv3 view routing | 0.924 balanced accuracy; 0.918 macro-F1 | Promoted for view routing only |
| Unity DINOv3 landmarks | 7.76 mm mean error; 0.923 visibility accuracy | Anatomy support only; not an endpoint claim |
| External OOD/failure audit | 772 usable sequences; AUROC 0.851; sensitivity 0.496; specificity 0.924 | Not promoted: sensitivity missed the predeclared 0.60 minimum |
| PanEcho MIMIC-LV comparison | EF 7.60 percentage points; EDV 26.08 mL; ESV 13.39 mL MAE | Did not beat the required thresholds |

The two 12-epoch MIMIC-LV challengers were also stopped by validation evidence,
not by an arbitrary epoch target. DINOv3's best checkpoint occurred at epoch 6
(EF 8.36 percentage points, EDV 22.36 mL, ESV 13.26 mL MAE) and V-JEPA 2.1's at
epoch 7 (8.28, 23.83, and 14.08 respectively); later epochs worsened the
development score. The best checkpoints were preserved. More epochs are not
automatically better.

### Current scalar gate status

The primary scalar gate remains closed because every endpoint is above the
specification's development MAE ceiling:

| Endpoint | Current MAE | Required MAE | Status |
| --- | ---: | ---: | --- |
| EF | 6.82 percentage points | ≤ 4.00 | Miss |
| LVEDV | 20.32 mL | ≤ 12.00 | Miss |
| LVESV | 13.49 mL | ≤ 9.00 | Miss |
| LVOT diameter | 1.42 mm | ≤ 1.20 | Miss |
| RV basal diameter | 5.08 mm | ≤ 2.80 | Miss |
| Aortic peak velocity | 0.51 m/s | ≤ 0.18 | Miss |

The learned large-residual detectors also remain below their predeclared 0.70
AUROC minimum. Deterministic insufficiency checks do abstain correctly when
required calibration or adequate cine evidence is absent, but that does not
substitute for a successful clinical failure detector.

Accordingly, G6 and G7 are closed, G8 has not been frozen, and the locked
internal test has not been opened. A clinical-validation claim additionally
requires train-excluded-vendor, temporal, multicentre, and prospective
multi-reader repeat-acquisition evidence.

## Safety and data boundary

The intended layout separates a read-only corpus mount from a local mutable
runtime. All caches, logs, manifests containing identifiers, model weights, and
evidence must stay outside Git. See [SECURITY.md](SECURITY.md) before running or
publishing changes.

MIMIC-derived artifacts and EchoJEPA-MIMIC ancestry are Research Track inputs.
Commercial use requires a separate provenance and licence review. No dataset or
checkpoint licence is granted by this repository.

## Requirements

- Windows PowerShell
- Python 3.12
- A CUDA-capable PyTorch environment (the reference run used PyTorch 2.11/CUDA
  12.8 on two RTX 5080 GPUs)
- Authorized local access to the datasets and checkpoints referenced in your
  private configuration

Public research datasets are optional challengers, not implicit training data.
Each has a declared role and must pass a held-out promotion rule. Controlled
datasets such as EchoNet-Dynamic, EchoNet-LVH, EchoNet-Pediatric, and EchoRisk
must be obtained by the researcher under their own account and terms.
The exact official portals, access steps, and safe handoff procedure are listed
in [DATASET-ACCESS-HANDOFF.md](DATASET-ACCESS-HANDOFF.md).

## Setup

```powershell
Copy-Item .\configs\protocol.example.yaml .\configs\protocol.yaml
# Edit protocol.yaml with your private local paths and permitted data sources.
.\scripts\setup_blackwell.ps1
```

Run the test suite before touching data:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

The main validation sequence is:

```powershell
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml runtime smoke
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml inventory dicom --workers 16 --resume
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml splits audit
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml research-data audit
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml gates status
```

After G0-G5 evidence has passed, start the resumable long run:

```powershell
.\scripts\launch_week_training.ps1 -AcknowledgePassedGates
```

The acknowledgement is not a bypass. The launcher calls `gates assert --through
G5` and stops if evidence, hashes, splits, provenance, or required assets are
missing.

Development-only research challengers can then be run explicitly:

```powershell
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml dense-lv train --backbone dinov3_vitb
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml temporal train --backbone vjepa21_vitb
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml mimic-lv train --backbone dinov3_vitb
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml unity-landmarks train
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml ev9v-view train
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml external-ood audit
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml panecho audit
```

An individually downloaded EchoNet-Dynamic archive can be structurally audited
before any long job, then converted to a local 16-frame cache. The importer
optimizes only the official `TRAIN` split, reserves `VAL` for development
selection, and never decodes or trains on the official `TEST` videos:

```powershell
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml echonet-dynamic audit --archive C:\path\to\EchoNet-Dynamic.zip
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml echonet-dynamic stage --archive C:\path\to\EchoNet-Dynamic.zip --workers 8
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml echonet-dynamic train --device 0
```

The resulting three-seed EF/ESV/EDV model is transfer pretraining, not a direct
core-route replacement. Its encoder/adapters must be fine-tuned and rechecked
on the MIMIC patient-disjoint development cohort before they can be promoted.
The verified archive contains 7,465 `TRAIN`, 1,288 `VAL`, and 1,277 reserved
`TEST` videos. A four-video CUDA smoke run completed all schedule phases without
decode errors or reserved-test access.

The final scalar schedule first creates a deduplicated local cine cache from the
authorized read-only corpus, then assigns three endpoints to each GPU:

```powershell
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml staged-final cache --workers 12
.\scripts\launch_staged_final_scalar.ps1 -SkipTests
```

The reference cache contains 11,783 unique development cines (9.25 GB) for
10,229 endpoint rows. Its completed audit reported zero decode errors, no
train/validation patient overlap, and no locked-test access. Training remains a
development challenger until all three seeds finish and its ensemble beats the
current route under the predeclared promotion rule.

These commands do not open the locked test automatically. A challenger that
misses its promotion threshold is retained as negative evidence and does not
replace the current model.

## Repository map

- `src/hemfm/` — pipeline, gates, pilots, extraction, and specialist training.
- `tests/` — unit tests for calibration, splitting, storage, provenance, and
  training contracts.
- `configs/protocol.example.yaml` — sanitized configuration template.
- `scripts/` — environment setup, validation, launch, continuation, and status.
- `PROTOCOL-COVERAGE.md` — requirement-to-enforcement ledger.
- `DATASETS.md` — dataset roles, access boundaries, licences, and promotion
  rules.
- `DATASET-ACCESS-HANDOFF.md` — official account-only download links and the
  local ingestion handoff.

## Status semantics

“Passed” means the corresponding machine-readable gate evidence exists and its
contract checks succeed. It does not mean that later external or prospective
gates have been satisfied. The pipeline deliberately refuses to infer evidence
that has not been collected.

## Licence

The original code in this repository is MIT-licensed. Datasets, papers,
third-party source trees, and model checkpoints retain their own licences; see
[THIRD_PARTY.md](THIRD_PARTY.md).
