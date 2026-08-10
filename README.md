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

**Execution passed != clinical gate passed.** A successful smoke test, pilot,
cache build, or three-seed development run proves only that its declared local
checks completed. HEM-FM v4 scientific gates have separate, fail-closed evidence
requirements; absent scientific evidence is always reported as a failed gate.

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
- Exclusive per-GPU reservation and duplicate-wave refusal for unattended runs.
- Hash-bound, provenance-sealed completion manifests so a finished seed can be
  reused or resumed only when its inputs and artifacts still verify.
- Separate local execution-readiness and scientific-gate status, with the
  internal test set kept sealed.

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
| Staged ViT-L scalar challengers | 18 runs: 6 endpoints × 3 seeds; frozen → DoRA → selective unfreeze | All completed and improved their prior development baselines; all still miss v4 full-coverage targets |
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

The primary scalar gate remains closed because every completed three-seed
staged challenger is above the specification's full-coverage programme target:

| Endpoint | Staged ensemble MAE | Prior baseline MAE | v4 full-coverage target | Status |
| --- | ---: | ---: | ---: | --- |
| EF | 6.363708 percentage points | 6.815433 | ≤ 4.00 | Miss |
| LVEDV | 20.093339 mL | 20.320021 | ≤ 12.00 | Miss |
| LVESV | 13.168489 mL | 13.487596 | ≤ 9.00 | Miss |
| LVOT diameter | 1.417459 mm | 1.423636 | ≤ 1.20 | Miss |
| RV basal diameter | 4.480584 mm | 5.075636 | ≤ 2.80 | Miss |
| Aortic peak velocity | 0.500563 m/s | 0.514412 | ≤ 0.18 | Miss |

Each staged report has `passed: true` and `promotion_eligible: true` because all
three development seeds completed, stayed finite, preserved the test lock, and
improved the prior development baseline. Neither field is a G6 scientific pass.

No learned large-residual detector passes the predeclared 0.70 AUROC minimum.
The LVESV uncertainty comparator reaches 0.761396, but its learned detector is
0.699430 and the poor-quality validation class is absent. Deterministic
insufficiency checks abstain correctly when calibration or adequate cine
evidence is absent, but that does not substitute for a successful clinical
failure detector.

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

## How to use it

Everything runs through one entry point, `python -m hemfm`, plus a small set of
PowerShell orchestration scripts for long unattended waves. Every command writes
a JSON report, prints it, and exits non-zero when its declared checks fail. There
is no "force" flag: a command either produces its evidence or fails closed.

The intended order is:

1. **Set up** the environment and private config (below).
2. **Verify the runtime** — `runtime smoke`, then the test suite.
3. **Build the data foundation** — `inventory`, `splits`, `calibration`,
   `cohort`. Splitting always precedes decoding.
4. **Run the controls and pilots** — `augmentation`, `shortcuts`, `provenance`,
   `architecture`, `tuning`.
5. **Check the gates** — `gates status`, then `gates assert --through G5`.
6. **Train** — `specialists`, `staged-final`, and the challenger routes.
7. **Audit** — `g6 audit`, `g7 audit`, `external-ood audit`, `panecho audit`.
8. **Freeze** — `g8 lock`, only after everything upstream passes.

Steps 6-8 never unlock themselves. A challenger that misses its promotion
threshold is retained as negative evidence and does not replace the current
model.

## Setup

```powershell
Copy-Item .\configs\protocol.example.yaml .\configs\protocol.yaml
# Edit protocol.yaml with your private local paths and permitted data sources.
.\scripts\setup_blackwell.ps1
```

`setup_blackwell.ps1` creates a Python 3.12 virtual environment, installs
PyTorch 2.11 for CUDA 12.8, and installs this project with its `dev` and `train`
extras. Pass `-Environment <path>` to place the environment elsewhere.

`configs/protocol.yaml` is git-ignored and is the only place private paths
belong. The loader refuses to start if `protocol` is not `HEM-FM-v4.0` or if any
mutable path (staging, checkpoints, runs, private, reports, evidence) is not an
absolute path on the local drive. The read-only DICOM corpus is the one
permitted network source.

Run the test suite before touching data. It is offline, needs no GPU, and no
dataset:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp C:\hemfm-t
```

Keep `--basetemp` short. Several tests build nested Git checkouts and cache
trees inside the temporary directory; a deep default temp path exceeds the
Windows `MAX_PATH` limit and makes `git init` fail with unrelated errors.

The main validation sequence is:

```powershell
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml runtime smoke
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml inventory dicom --workers 16 --resume
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml splits audit
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml research-data audit
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml gates status
```

After the G0-G5 local execution prerequisites have passed, start the resumable
long run:

```powershell
.\scripts\launch_week_training.ps1 -AcknowledgePassedGates
```

The acknowledgement is not a bypass. The launcher calls `gates assert --through
G5`, which checks local execution prerequisites and stops if required hashes,
splits, provenance, assets, or subcheck reports are missing. It does not declare
the v4 scientific gates passed.

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
.\.venv\Scripts\python.exe -m hemfm --config .\configs\protocol.yaml echonet-trace train --device 0
```

The resulting three-seed EF/ESV/EDV model is transfer pretraining, not a direct
core-route replacement. Its encoder/adapters must be fine-tuned and rechecked
on the MIMIC patient-disjoint development cohort before they can be promoted.
The verified archive contains 7,465 `TRAIN`, 1,288 `VAL`, and 1,277 reserved
`TEST` videos. A four-video CUDA smoke run completed all schedule phases without
decode errors or reserved-test access.
The cache also preserves the exact traced ED/ES RGB frames, original frame
indices, and rasterized LV masks from `VolumeTracings.csv`; these annotations
remain available for dense LV and phase-specific challengers instead of being
discarded during uniform temporal sampling.

The `echonet-trace` route uses those exact expert frames for a separate
three-seed EchoJEPA ViT-L binary LV segmentation challenger with boundary and
heteroscedastic uncertainty heads. Its validation ensemble is majority voted
and remains transfer-only until subsequent patient-disjoint MIMIC development
testing improves and matched failure detection passes.

For an unattended local sequence, `launch_post_scalar_queue.ps1` waits for all
six three-seed scalar reports, validates the code, hash-verifies and caches only
EchoNet `TRAIN`/`VAL`, then distributes the three transfer seeds across both
GPUs and finalizes their validation ensemble. It then runs a traced-segmentation
CUDA smoke, distributes three dense seeds across both GPUs, and finalizes that
validation ensemble as well.
The queue then reruns the G7 integration audit and final gate status. A staged
scalar route is selected only when its complete three-seed ensemble improves
the current G6 baseline; accuracy improvement does not waive the separate
failure-detection requirement.

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

## Command reference

Every command is `python -m hemfm [--config <path>] <command> <action> [options]`.
`--config` defaults to `configs/protocol.yaml`. `--device` selects GPU 0 or 1.
Training routes accept `--epochs`, `--frozen-epochs`, `--max-train`, and
`--max-validation` to bound a development run; each has a route-specific default,
and `smoke` actions use small ones deliberately.

| Command | Actions | What it does | Notable options |
| --- | --- | --- | --- |
| `runtime` | `smoke`, `challengers` | BF16/CUDA deterministic multi-GPU smoke plus the EchoJEPA checkpoint tensor smoke; `challengers` smokes the DINOv3 and V-JEPA 2.1 ViT-L/ViT-B checkpoints | — |
| `inventory` | `labels`, `dicom` | Measurement/label inventory; full-corpus DICOM metadata scan bound to the hashed split manifest and record-list hash | `--workers`, `--limit`, `--resume` |
| `splits` | `build`, `audit` | Build or re-audit the immutable patient-level split manifest | — |
| `calibration` | `validate` | Sampled physical spatial and spectral calibration validation against real DICOM headers | `--max-files` |
| `cohort` | `build` | Assemble per-endpoint patient cohorts from measurement evidence | `--confidence-min` |
| `augmentation` | `validate` | Physical-space augmentation round-trip and affine/label parity check | `--trials` |
| `shortcuts` | `validate` | Caliper/overlay shortcut controls and clean-frame negative controls | `--train-per-target`, `--validation-per-target`, `--workers`, `--device` |
| `provenance` | `validate` | Write and check the licence/track provenance record enforcing the Track R / Track C firewall | — |
| `architecture` | `pilot` | Four-backbone frozen architecture ladder | `--train-per-target`, `--validation-per-target` |
| `tuning` | `pilot` | LoRA / rsLoRA / DoRA / PiSSA against AdamW/Muon with EMA, plus the decision ledger | `--train-count`, `--validation-count`, `--epochs`, `--device` |
| `specialists` | `extract`, `train`, `fusion` | Resumable four-backbone feature extraction; three-seed six-endpoint frozen specialists; multi-backbone fusion and grouped OOF pseudo-labels | — |
| `dense-lv` | `manifest`, `smoke`, `train` | CAMUS dense LV cavity / myocardium / left-atrium segmentation | `--backbone {vjepa21_vitb,dinov3_vitb}` |
| `temporal` | `smoke`, `train` | TED full-cycle temporal-consistency segmentation with the official CAMUS test identities sealed | `--backbone` |
| `research-data` | `audit` | Hash-verified, licence-aware research-dataset audit and promotion rules | — |
| `echonet-dynamic` | `audit`, `stage`, `smoke`, `train`, `seed`, `finalize` | Structural archive audit, 16-frame `TRAIN`/`VAL` cache build, EF/ESV/EDV transfer training, single-seed run, three-seed ensemble | `--archive` (required for `audit`/`stage`), `--hash`, `--workers`, `--frames`, `--resolution`, `--limit`, `--seed` |
| `echonet-trace` | `smoke`, `train`, `seed`, `finalize` | Expert ED/ES traced-frame binary LV segmentation challenger with boundary and heteroscedastic uncertainty heads | `--seed` |
| `mimic-lv` | `stage`, `smoke`, `train` | MIMIC LV-volume extension with spatial calibration, biplane-first labels, heteroscedastic uncertainty, and physiological consistency losses | `--source-root`, `--metadata`, `--backbone`, `--ema-decay`, `--run-tag` |
| `unity-landmarks` | `smoke`, `train` | 31 anatomical landmarks with missing-label masking and physical-millimetre evaluation | — |
| `ev9v-view` | `smoke`, `train` | Nine-view routing with exam-disjoint validation and calibrated metrics | — |
| `external-ood` | `smoke`, `audit` | CardiacUDA / EchoCP / CardiacNet external failure and abstention evaluation | `--max-per-dataset` |
| `panecho` | `smoke`, `audit` | PanEcho comparison on the MIMIC-LV route | `--max-train`, `--max-validation` |
| `staged-final` | `cache`, `smoke`, `train`, `audit` | Deduplicated local cine cache; the staged ViT-L scalar schedule (frozen → DoRA PEFT → selective unfreeze); matched three-seed uncertainty audit | `--target`, `--workers`, `--limit` |
| `g6` | `audit` | Specialist holdout and failure-detection audit | — |
| `g7` | `audit` | Integration audit across promoted routes | — |
| `g8` | `lock` | Freeze and hash the analysis before one-touch external testing | — |
| `gates` | `status`, `assert` | Report execution and scientific gate state; assert prerequisites | `--through G0`…`G9` (default `G5`) |
| `pilot` | `functional` | Cached functional-feature pilot | `--seed`, `--device`, `--epochs`, `--batch-size` |

`--target` accepts `EF`, `LVEDV`, `LVESV`, `LVOT_DIAMETER`, `RV_BASAL_DIAMETER`,
and `AV_PEAK_VELOCITY`.

Exit codes are meaningful and scriptable: `0` means the declared checks passed,
`2` means the command ran and declared failure, and any other non-zero code is an
error. Commands that decode MIMIC data first assert that the source-bound
development split is present and still validates, so a missing or altered split
manifest stops the run before a single image is read.

## Orchestration scripts

`scripts/` holds the PowerShell entry points for unattended waves. Each accepts
`-Environment` and `-Config` overrides.

| Script | Purpose |
| --- | --- |
| `setup_blackwell.ps1` | Create the Python 3.12 venv and install pinned PyTorch plus the project extras. |
| `run_gate0.ps1` | Runtime smoke, full test suite, and split audit in one pass. |
| `run_inventory.ps1` | Label inventory → DICOM inventory → physical-calibration validation. |
| `run_functional_pilots.ps1` | Three-seed cached functional pilot spread across both GPUs. |
| `localize_assets.ps1` | Copy the EchoJEPA source, checkpoints, and feature caches into the local runtime tree. |
| `launch_week_training.ps1` | `-AcknowledgePassedGates` asserts G0-G5, then starts resumable feature extraction and specialist training. |
| `continue_week_training.ps1` | Resume specialist training once extraction reports completion. |
| `launch_staged_final_scalar.ps1` | The staged ViT-L scalar wave: three endpoints per GPU. `-Workers`, `-Epochs`, `-SkipTests`. |
| `launch_post_scalar_queue.ps1` | The unattended EchoNet wave. Waits for all six three-seed scalar reports, validates the code, hash-verifies and caches only `TRAIN`/`VAL`, distributes three transfer seeds across both GPUs, finalizes, repeats for traced segmentation, then reruns the G7 audit and gate status. Requires `-Archive`. |
| `launch_post_queue_matched_audits.ps1` | Waits on the queue's process ID, then runs the matched staged uncertainty audits grouped per GPU. |
| `launch_post_training_data_foundation.ps1` | Runs the v2 source-bound inventory and split foundation once the active GPU job exits. |
| `resume_research_queue.ps1` | Resume the research training queue after an interruption. |
| `run_staged_scalar_worker.ps1`, `run_staged_audit_group.ps1`, `run_echonet_dynamic_seed_worker.ps1`, `run_echonet_trace_seed_worker.ps1` | Single-GPU workers invoked by the launchers. Runnable directly for one device. |
| `training_status.ps1`, `live_monitor.ps1` | Print `launch.json`/`status.json`, or watch a refreshing console dashboard. |
| `monitor_server.py`, `live_dashboard.html` | Local browser dashboard over the same status files. |
| `download_*.py`, `extract_*.py` | Helpers for account-gated dataset acquisition and extraction under your own terms. |

## Run integrity and completion evidence

A multi-day, multi-GPU wave is the easiest place for a silent failure to be
mistaken for a result. Two mechanisms make that fail closed.

**Exclusive GPU reservation.** Every worker that touches a GPU opens
`work/gpu-locks/gpu_<device>.lock` with no sharing before it does anything else,
and releases it in a `finally` block. A second worker aimed at the same device
fails immediately instead of quietly halving throughput or exhausting VRAM. The
launchers additionally scan running processes for another live worker on the same
repository and config, and refuse to start a duplicate wave.

**Hash-bound completion manifests.** `python -m hemfm.worker_completion` seals
what a seed ran against and re-checks what it produced:

```powershell
# Seal the inputs before a seed starts.
.\.venv\Scripts\python.exe -m hemfm.worker_completion snapshot-provenance `
    --config .\configs\protocol.yaml --repository . --output launch_provenance.json

# Bind the finished artifacts to that snapshot.
.\.venv\Scripts\python.exe -m hemfm.worker_completion record-seed `
    --kind transfer --artifact-root <runs>\echonet_dynamic_transfer --seed 1103 `
    --mode full --requested-epochs 10 --config .\configs\protocol.yaml --repository . `
    --not-before <ISO-8601-UTC> --launch-provenance launch_provenance.json

# Re-verify later, independently.
.\.venv\Scripts\python.exe -m hemfm.worker_completion verify-seed `
    --kind transfer --artifact-root <runs>\echonet_dynamic_transfer --seed 1103 `
    --mode full --requested-epochs 10 --config .\configs\protocol.yaml --repository .

# And that a matched uncertainty audit is fresh and complete.
.\.venv\Scripts\python.exe -m hemfm.worker_completion verify-matched-audit `
    --evidence-root <evidence> --target EF --mode full `
    --config .\configs\protocol.yaml --repository . --not-before <ISO-8601-UTC>
```

`verify-provenance` re-checks a snapshot on its own. `--kind` is `transfer` or
`trace`.

The snapshot binds the config file, the `src/hemfm` source tree, the functional
checkpoint SHA-256, the EchoJEPA checkout (HEAD, tree, worktree cleanliness, and
a content hash), the EchoNet stage evidence, and every cache file the development
manifest actually references. `record-seed` refuses to write a manifest if any of
those changed since launch. `verify-seed` refuses to accept a manifest whose
report summary, artifact hashes, or provenance no longer match. Reports are
validated structurally as well: history must cover frozen → PEFT → selective
unfreeze in that order over contiguous one-based epochs, an early stop needs four
post-best epochs, booleans must be real JSON booleans rather than `"true"` or
`1`, and `locked_test_accessed` must be exactly `false`.

This is what lets the queue scripts reuse work safely. An existing `report.json`
is reused only when `verify-seed` passes against the current inputs, and a
partial run is resumed only when its `launch_provenance.json` still matches. A
`checkpoint_best.pt` with no resumable last checkpoint is treated as an orphan
and stops the wave rather than being silently overwritten.

Windows PowerShell 5.1 can also hand back a `Process` whose `ExitCode` is null
when `Start-Process` was used without `-Wait`. The launchers never read that as
success: a null exit code is accepted only when the worker's stdout carries its
completion marker, both logs are newer than the launch, stderr holds no fatal
signature, and every seed passes `verify-seed`.

## Where the outputs go

Nothing generated by a run belongs in Git. The five configured roots are all
local and all ignored:

| Root | Contents |
| --- | --- |
| `evidence_root/G0`…`G9` | Gate evidence JSON, one file per check. |
| `report_root` | Non-gate reports such as smokes and inventories. |
| `run_root` | Per-run logs, `status.json`, checkpoints, and per-seed artifacts including `completion_manifest.json`. |
| `private_root` | Identifier-bearing manifests, inventories, and per-sample predictions. |
| `local_staging_root` | Dataset caches such as `cache_16x224`. |

`.gitignore` additionally blocks weights, feature caches, DICOM, video, archives,
and any `*private*.csv` or `*manifest*.csv`. Check `git status` before committing
if you have changed where the runtime writes.

## Repository map

- `src/hemfm/` — pipeline, gates, pilots, extraction, and specialist training.
  `cli.py` is the single entry point; `worker_completion.py` is a separate
  fail-closed module for provenance snapshots and completion manifests.
- `tests/` — unit tests for calibration, splitting, storage, provenance,
  completion evidence, and training contracts.
- `configs/protocol.example.yaml` — sanitized configuration template.
- `scripts/` — environment setup, validation, launch, continuation, and status.
- `PROTOCOL-COVERAGE.md` — requirement-to-enforcement ledger.
- `V4-EXECUTION-LEDGER.md` — section-by-section specification coverage and the
  current gate ledger.
- `DATASETS.md` — dataset roles, access boundaries, licences, and promotion
  rules.
- `DATASET-ACCESS-HANDOFF.md` — official account-only download links and the
  local ingestion handoff.

## Status semantics

`gates status` reports two deliberately different states:

- `execution_checks_passed` / `execution_sequentially_unlocked` mean the local
  mechanical and proxy prerequisites needed to run the next research job are
  present.
- `passed` / `scientific_gate_passed` mean the direct HEM-FM v4 gate evidence
  exists and passed. `sequentially_unlocked` additionally requires every
  preceding scientific gate to have passed.

The current G0-G5 artifacts are execution subchecks, not full v4 gate evidence.
G6-G9 remain closed. The pipeline deliberately refuses to infer scientific,
external, prospective, or commercial evidence that has not been collected.

## Licence

The original code in this repository is MIT-licensed. Datasets, papers,
third-party source trees, and model checkpoints retain their own licences; see
[THIRD_PARTY.md](THIRD_PARTY.md).
