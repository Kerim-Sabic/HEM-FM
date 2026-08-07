# HEM-FM v4 local execution system

HEM-FM v4 is a fail-closed research pipeline for training and evaluating
multi-endpoint echocardiography models. It turns a gated training specification
into executable checks for hardware, DICOM calibration, patient-disjoint splits,
shortcut controls, architecture selection, PEFT/optimizer selection, specialist
training, and out-of-fold pseudo-label generation.

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
- Resumable four-backbone feature extraction on local storage.
- Three-seed, six-endpoint specialist training and grouped OOF pseudo-labels.
- Sequential evidence gates that keep the internal test set sealed.

External/vendor validation and prospective multi-reader repeat acquisition remain
separate evidence requirements. They cannot be replaced by local retrospective
training.

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
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml runtime validate
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml inventory dicom --workers 16 --resume
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml splits audit
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml gates status
```

After G0-G5 evidence has passed, start the resumable long run:

```powershell
.\scripts\launch_week_training.ps1 -AcknowledgePassedGates
```

The acknowledgement is not a bypass. The launcher calls `gates assert --through
G5` and stops if evidence, hashes, splits, provenance, or required assets are
missing.

## Repository map

- `src/hemfm/` — pipeline, gates, pilots, extraction, and specialist training.
- `tests/` — unit tests for calibration, splitting, storage, provenance, and
  training contracts.
- `configs/protocol.example.yaml` — sanitized configuration template.
- `scripts/` — environment setup, validation, launch, continuation, and status.
- `PROTOCOL-COVERAGE.md` — requirement-to-enforcement ledger.

## Status semantics

“Passed” means the corresponding machine-readable gate evidence exists and its
contract checks succeed. It does not mean that later external or prospective
gates have been satisfied. The pipeline deliberately refuses to infer evidence
that has not been collected.

## Licence

The original code in this repository is MIT-licensed. Datasets, papers,
third-party source trees, and model checkpoints retain their own licences; see
[THIRD_PARTY.md](THIRD_PARTY.md).
