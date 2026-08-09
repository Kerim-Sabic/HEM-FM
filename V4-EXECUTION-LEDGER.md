# HEM-FM v4 execution ledger

Snapshot: 2026-08-09 10:33 UTC. HEM-FM v4.0 is the master protocol. Material
from v2.1 is admitted only as a compatible challenger and must pass the v4
promotion gates; it cannot weaken a v4 requirement.

This ledger maps the specification to repository enforcement and aggregate
development evidence. It does not inspect or reproduce patient-level evidence,
private manifest rows, restricted data, or sealed TEST data.

Status meanings:

- **Implemented**: the narrow engineering rule is enforced and tested. This is
  not a scientific or clinical pass.
- **Execution-only**: local machinery or proxy evidence exists, but the direct
  scientific requirement is not established.
- **Experiment pending**: the pre-registered experiment, full specialist, or
  promotion evidence is incomplete or failed.
- **External-blocked**: completion requires controlled-access, locked external,
  commercial-lineage, or prospective evidence not available from the current
  retrospective local run.

## Honest current evidence

All G0-G9 scientific gates are closed. G0-G5 local execution checks pass; G6-G9
local execution checks do not. `passed: true` or `promotion_eligible: true` in a
development-run report means only that the run completed its declared local
checks and, where stated, improved its prior development comparator. It does not
mean that G6, G7, clinical validation, or superiority passed.

The completed three-seed staged scalar ensembles improved the prior local
baselines but miss every v4 full-coverage programme target:

| Endpoint | Development MAE | v4 full-coverage target | Result |
| --- | ---: | ---: | --- |
| EF | 6.363708 percentage points | <= 4.00 | Miss |
| LVEDV | 20.093339 mL | <= 12.00 | Miss |
| LVESV | 13.168489 mL | <= 9.00 | Miss |
| LVOT diameter | 1.417459 mm | <= 1.20 | Miss |
| RV basal diameter | 4.480584 mm | <= 2.80 | Miss |
| Aortic peak velocity | 0.500563 m/s | <= 0.18 | Miss |

The CardiacUDA multi-chamber baseline, the caliper-mining pilot, and the v2
DICOM inventory/split implementation are unpromoted. Their code or execution
subchecks do not become scientific evidence until their complete, predeclared,
patient/domain-disjoint evidence passes. EchoNet-Dynamic and expert-trace routes
are transfer challengers only; the official EchoNet TEST split and the internal
MIMIC test remain sealed.

## Numbered specification sections

| Section | v4 requirement | Repository enforcement or current evidence | Status | Next action |
| --- | --- | --- | --- | --- |
| 00 | Treat the specification as a labelled research operating system. | `README.md`, `PROTOCOL-COVERAGE.md`, `configs/protocol.example.yaml`, and `src/hemfm/gates.py` separate verified mechanics, proposals, and scientific gates. | Implemented | Keep every new claim tied to direct evidence; never infer a pass from execution. |
| 01 | Follow v4 decisions: physical units, task-specific augmentation, controlled architecture/optimizer ladders, source evidence, uncertainty, provenance, and minimum sufficient complexity. | Configuration, runtime, calibration, augmentation, provenance, training, and gate modules encode the main decisions; scientific promotions remain closed. | Execution-only | Finish each pre-registered gate in order and retain the simplest winning specialist. |
| 02 | Apply v3 corrections, especially task-conditioned flip and honest interpretation below reader disagreement. | `src/hemfm/augmentations.py`, `src/hemfm/dicom_preprocess.py`, and parity tests track transforms and keep Doppler/landmark policies separate. No reader-noise superiority inference is made. | Implemented | Prove each optional transform on cross-domain endpoint evidence before promotion. |
| 03 | Use claim registry, pre-registered margins, negative controls, reader modelling, invariance registry, failure library, provenance, external-first selection, reproducible envelopes, and kill gates. | Hashing, provenance, split, shortcut, external-OOD, gate, and integration checks cover a subset; a complete immutable claim registry and replayable failure library do not yet exist. | Execution-only | Add claim/failure manifests and require them in promotion and release checks. |
| 04 | Enforce L0-L3 claims and a Pareto promotion gate with no critical regression. | `src/hemfm/gates.py`, `src/hemfm/integration_audit.py`, and `src/hemfm/external_lock.py` fail closed and keep engineering evidence separate from analytical/external claims. | Execution-only | Attach patient-level confidence intervals, subgroup checks, latency/VRAM, and locked external evidence to each claim ID. |
| 05 | Meet six full-coverage and accepted-case targets while reporting coverage, abstention, calibration, source evidence, and reproducibility. | Targets are encoded in config and aggregate scalar results are reported above; all six full-coverage targets currently miss. | Experiment pending | Improve source-linked specialists, run matched uncertainty audits, and do not open TEST until G6-G8 prerequisites pass. |
| 06 | Use verified public resources only for their declared role and licence track. | `DATASETS.md`, `DATASET-ACCESS-HANDOFF.md`, `THIRD_PARTY.md`, `src/hemfm/research_datasets.py`, and dataset audits record role, licence, hashes, and promotion limits. | Execution-only | Obtain account-gated assets under their terms only when they fill a predeclared evidence gap; re-audit every lineage. |
| 07 | Complete MIMIC metadata inventory, calibration parser, and immutable patient split before decoding/training. | `src/hemfm/inventory.py`, `src/hemfm/splits.py`, `src/hemfm/calibration_validation.py`, and CLI fail-closed bindings implement v2 Parquet/journal/split machinery. The current v2 production scan is not yet accepted as final evidence. | Experiment pending | Run the full resumable metadata-only v2 scan after the active GPU queue, bind it to the immutable split, sample physical plausibility, and issue the separate G1 evidence. |
| 08 | Mine calipers only with OCR/glyph, physical verification, clean-parent recovery, shortcut controls, and blinded P1-P5 audit. | `src/hemfm/caliper_pilot.py`, `src/hemfm/shortcut_controls.py`, and tests enforce geometry-verifiable acceptance and clean-frame controls. The P1-P5 yield/precision evidence is incomplete. | Experiment pending | Complete the stratified manual and two-reviewer audits; promote only if pre-set precision and clean-frame yield pass. |
| 09 | Use UnityLV-MultiX for reader distributions/noise and Unity labels for research landmarks, never restricted commercial weights. | `src/hemfm/unity_landmarks.py` supplies a research landmark baseline and provenance keeps Unity in Track R; full 11-reader MultiX analysis is absent. | Experiment pending | Build the reader-preserving consensus/uncertainty analysis without collapsing expert IDs or transferring weights to Track C. |
| 10 | Respect Platinum/Gold/Gold-mined/Silver/Bronze supervision boundaries and model reader/acquisition effects. | Cohort and OOF code keep patient groups and weak-label outputs separate; dense public tasks mask unavailable labels. A full tier-aware reader/acquisition observation model is not integrated. | Execution-only | Add immutable tier/reader/acquisition fields to every training example and restrict each loss by the v4 permission table. |
| 11 | Physically separate Research Track R and Commercial Track C ancestry. | `src/hemfm/provenance.py` rejects research sources or numerical ancestors in Track C; tests cover the firewall. | Implemented | Keep this repository Research Track; build a separate permitted Track C manifest and training lineage before any commercial claim. |
| 12 | Test the hierarchical V-JEPA 2.1 thesis against DINOv3 and functional V-JEPA, specialist by specialist. | V-JEPA 2.1, functional V-JEPA, and DINOv3 loaders/challengers exist, but no final heterogeneous architecture has passed G4/G6. | Experiment pending | Run fixed-recipe specialist comparisons and promote only target-specific gains that survive robustness and calibration. |
| 13 | Search on ViT-B, commit on ViT-L, use ViT-g only as teacher/cache, and fit the 16 GB envelope. | Runtime asset checks, `src/hemfm/architecture_pilot.py`, and staged ViT-L training follow frozen -> PEFT -> selective unfreeze; ViT-g is not trained. | Execution-only | Complete same-cohort capacity comparisons and record actual VRAM, throughput, and checkpoint ancestry. |
| 14 | Run the A/B/B+/C/D/E architecture ladder with fixed recipes and pre-registered margins. | Architecture ladder execution evidence exists, plus dense and fusion pilots; `architecture_promotion_gate.json` is absent. | Experiment pending | Run identical-split A/B/B+/C first, then D/E only if warranted; record critical subgroup non-regression. |
| 15 | Route from DICOM priors and learned view/measurability/quality/phase/source signals while retaining multiple candidates. | DICOM metadata parsing, EV9V view routing, preprocessing, and candidate cohort code provide partial routing. Full multi-candidate source posterior and calibrated measurability integration are incomplete. | Execution-only | Integrate priors and learned outputs into a study-level candidate set, then evaluate wrong-view rejection and source recall. |
| 16 | Build hierarchical dense features with masks, boundaries, distributional landmarks, uncertainty, and sub-pixel decoding. | CAMUS, EchoNet trace, Unity landmark, and multi-chamber challenger modules implement parts of this output family. They are separate baselines, not the complete ViT-L hierarchy. | Experiment pending | Compare tap sets and sub-pixel decoders on patient-disjoint gold data; require physical-distance and uncertainty calibration. |
| 17 | Integrate LV contours/landmarks/phase/adequacy/trajectory, calibrated biplane volumes, direct regression, and disagreement. | Scalar, CAMUS/TED, EchoNet trace/transfer, and MIMIC LV-extension routes cover separate pieces; the integrated source-linked biplane expert is absent and scalar targets miss. | Experiment pending | Train the integrated calibrated A4C+A2C LV expert and prove volume/EF consistency plus matched failure detection. |
| 18 | Predict explainable LVOT anatomy, endpoints, perpendicularity, convention, calibration, and uncertainty. | The scalar LVOT route and caliper pilot exist; no promoted anatomical LVOT endpoint/axis specialist exists. | Experiment pending | Acquire/verify clean PLAX endpoint supervision and train a distributional, calibrated geometry head with flip off by default. |
| 19 | Gate RV adequacy before calibrated basal geometry and exact endpoint prediction. | The scalar RV route exists; CardiacUDA provides an unpromoted multi-chamber segmentation baseline, not exact RV-basal endpoint truth. | Experiment pending | Complete domain-disjoint CardiacUDA evidence, then train exact ED basal endpoints from mined/owned annotations and test adequacy first. |
| 20 | Keep a separate calibrated near-native Doppler pathway with envelope, peak, beat validity, clipping, window rank, and asymmetric risk. | Spectral region calibration and an AV scalar route exist; a full envelope/beat/window specialist has not been trained. | Experiment pending | Build source-linked CW labels and the separate spectral head; validate velocity-axis invariance and severe-case underestimation. |
| 21 | Compare beat transformer, GDKVM, SAM2-style propagation, and justified hybrids. | `src/hemfm/ted_temporal.py` provides temporal segmentation evidence, but there is no controlled GDKVM/SAM2/beat-transformer ladder. | Experiment pending | Run same-backbone temporal-memory comparisons using contour/endpoint stability and volume trajectory, not Dice alone. |
| 22 | Maintain a task-level invariance registry and transfer EAGT conclusions only through controlled cross-domain tests. | `src/hemfm/augmentations.py` enforces affine/label parity and task restrictions; G3 cross-domain benefit evidence is absent. | Execution-only | Run the predeclared cross-dataset augmentation study and preserve narrower policies for landmarks and Doppler. |
| 23 | Apply separate conservative recipes for LV contours, landmark/diameter tasks, and Doppler. | Dense training and transform tests implement portions of the recipes; exact policy sweeps and Doppler stress evidence are incomplete. | Execution-only | Freeze per-task configs only after cross-domain gains and physical invariance both pass. |
| 24 | Use AdamW as baseline and test Muon only in a controlled ViT-B comparison before ViT-L. | `src/hemfm/optimizer_peft_pilot.py` and G5 execution evidence exercise AdamW/Muon mechanics; no external-benefit promotion exists. | Execution-only | Run optimizer-only comparisons with identical model/data/augmentation and compare EMA, calibration, cost, and external score. |
| 25 | Compare LoRA, rsLoRA, DoRA, PiSSA, and selective/full fine-tuning under a fixed budget. | PEFT pilot code covers all named methods; staged scalars used DoRA and selective unfreezing. This is not a complete external, fixed-budget promotion experiment. | Execution-only | Run controlled ranks/placements and keep the smallest method matching quality; do not share raw Doppler adapters. |
| 26 | Evaluate EMA by default, then WiSE-FT, SWA, and soups without mixing provenance tracks. | EMA is used in specialist/staged training. WiSE-FT/SWA/soup selection and provenance-compatible OOD comparisons are not complete. | Experiment pending | Compare last/EMA/WiSE/SWA/soup on the same checkpoints and reject any geometry or calibration regression. |
| 27 | Use BF16/FP32 correctness first; permit FP8/MXFP8 only after measured parity. | Runtime/config enforce BF16, log the platform, and keep FP8 disabled by default. | Implemented | Leave FP8 off until the BF16 reference is stable; then run explicit numerical and endpoint parity if speed warrants it. |
| 28 | Use staged learning rates/schedules; reserve muP/u-muP for compatible from-scratch scale families. | Config and staged trainers implement frozen, PEFT, and low-rate selective-unfreeze schedules. muP is correctly not treated as a core drop-in. | Execution-only | Tune rates on development only, record schedule/throughput, and avoid muP unless a Track C from-scratch family is started. |
| 29 | Use structured dense/landmark/phase/uncertainty/calibration losses and measure multi-task gradient conflict before sharing. | Dense, trace, landmark, scalar, and chamber modules implement subsets of these losses; systematic gradient-conflict evidence is absent. | Experiment pending | Establish competent specialists first, then measure shared-block gradient cosine and separate tasks before adding gradient surgery. |
| 30 | Use strict OOF heterogeneous teachers, geometry-aware confidence gates, and no Bronze dense supervision. | OOF cohort/pseudo-label machinery and gate evidence exist; full tier-aware manual Silver audit and heterogeneous source-target teacher proof are incomplete. | Execution-only | Audit OOF lineage, teacher/source circularity, Gold replay, and every Silver acceptance condition before raising weak-label weight. |
| 31 | Learn from source-linked physician corrections with replay and immutable foundation teachers. | No end-to-end correction ingestion, weighting, replay, and hard-negative lifecycle is production-ready. | Experiment pending | Define the correction schema and offline replay experiment before any online or clinician-facing update path. |
| 32 | Generalize through data, representation, objectives, and vendor/site evaluation; benchmark TTA offline but do not ship it first. | Augmentation, EMA, external-OOD, and subgroup-oriented audit code exist; comprehensive multi-vendor/site selection evidence does not. No online TTA is shipped. | Experiment pending | Run train-excluded vendor/site stress tests with frozen inference; quarantine any TTA experiment behind rollback and locked evaluation. |
| 33 | Separate aleatoric, epistemic, and structural uncertainty; calibrate abstention independently from conformal intervals. | `src/hemfm/staged_scalar_audit.py`, `src/hemfm/calibration.py`, and OOD audits implement ensemble uncertainty, raw interval coverage, failure AUROC, and risk-coverage. Matched audits/promotion evidence are not complete for all routes. | Experiment pending | Finish matched three-seed audits on identical cohorts and require calibration, failure detection, and selective-risk checks together. |
| 34 | Validate repeat acquisition and multiple readers; use Unity only to validate analysis code before a prospective Platinum cohort. | Local Unity landmark work cannot supply repeat acquisitions or the required owned multi-reader cohort. | External-blocked | Conduct the pre-specified 150-250-patient, two-reader, three-repeat acquisition study and estimate variance components. |
| 35 | Evaluate internal patient, temporal, vendor, hospital, scanner, pathology, quality, repeat, and locked multicentre holdouts. | Patient-disjoint internal development and some public-domain stress audits exist; required temporal/train-excluded-vendor/multicentre/prospective holdouts are incomplete. | External-blocked | Assemble the missing locked holdouts without using them for tuning; touch the final external test only after G8 freezes. |
| 36 | Report endpoint-specific accuracy, calibration, selective risk, reproducibility, patient bootstrap CIs, and subgroup results. | Specialist, staged-audit, external-OOD, and lock modules compute a useful subset; complete patient bootstrap, all endpoint metrics, and all subgroups are absent. | Execution-only | Finalize immutable analysis scripts and cohort denominators before any TEST access. |
| 37 | Maintain a replayable failure library and negative controls for leakage, calibration, overlays, views, missing evidence, OOD, clipping, labels, transforms, and duplicates. | Split, calibration, shortcut, augmentation, research-data, and OOD tests implement several fail-closed controls. The full versioned failure library is incomplete and the learned failure detector misses its threshold. | Execution-only | Version every false accept/failure case and require full replay with no unexplained regression before promotion. |
| 38 | Meet accepted-case accuracy, reproducibility, and source-evidence targets together. | Accepted-case thresholds are encoded, but current scalar evidence misses the looser full-coverage targets and prospective reproducibility is unavailable. | Experiment pending | Improve local full-coverage performance first, then validate calibrated coverage and repeat-acquisition targets externally. |
| 39 | Operate within two independent 16 GB RTX 5080s using BF16, measured distributed execution, and local mutable storage. | `src/hemfm/runtime.py`, storage checks, config, and launch scripts enforce the hardware floor, independent devices, BF16, and local outputs. | Implemented | Continue logging VRAM/throughput/kernel path and keep network data sources read-only. |
| 40 | Use immutable L0/L1/L2/L3/LD caches with source identity, transforms, affine, checkpoint, projection, and code hashes. | Cached scalar, frozen extraction, EchoNet staging, and local storage machinery exist; the exact complete five-tier cache contract is not yet proven corpus-wide. | Execution-only | Finish source-bound inventory first, then materialize only the cache tiers justified by each experiment and audit every key/hash. |
| 41 | Record run/model/data/transform/metric/claim manifests with immutable environment and lineage. | Hashes, seeds, split manifests, provenance records, status files, and logged jobs exist. Full container, transform, metric, and claim manifests are not uniformly required. | Execution-only | Define one manifest schema and make every training/evaluation entry point fail closed on missing fields. |
| 42 | Apply the v4 debugging playbook to leakage, calibration, shortcuts, source mismatch, temporal instability, optimizer/precision, pseudo-label, and conformal failures. | Tests and audits expose many named failure modes and reports preserve negative evidence; diagnostic replay is not comprehensive. | Execution-only | Convert each playbook symptom into a reproducible check or failure-library case with an owner and disposition. |
| 43 | Complete the day 1-30 recipe before large specialist commitment. | Many assets, pilots, caches, and development runs exist, but official probe reproduction, v2 inventory acceptance, caliper P1-P5, cross-domain augmentation, full architecture promotion, and MultiX analysis remain open. | Experiment pending | Close G0-G5 direct scientific evidence in protocol order; treat completed large runs as development evidence only. |
| 44 | Run the 120-day G0-G9 release sequence and stop on every failed gate. | `src/hemfm/gates.py` separates execution and scientific status and fails closed. No scientific gate currently passes. | Execution-only | Do not unlock TEST or release claims; follow the gate actions in the table below. |
| 45 | Appendix A1 reference hyperparameters. | `configs/protocol.example.yaml` and trainers encode the principal BF16, 16-frame, three-seed, ViT-B/ViT-L, EMA, PEFT, and OOF defaults; several appendix sweeps remain unrun. | Execution-only | Record measured deviations per run and finish only the sweeps needed by a predeclared question. |
| 46 | Appendix A2 task-specific augmentation skeleton. | Transform code/tests encode physical parity and conservative task policies, but the full skeleton is not yet a frozen winning configuration. | Execution-only | Store per-task winning configs after G3 cross-domain evidence, including parity and affine updates. |
| 47 | Appendix A3 training-loop contract. | Training modules implement BF16, task augmentation, staged tuning, losses, EMA, checkpoints, and status logging in route-specific forms. | Execution-only | Standardize source/tier/transform logging and compare last/EMA/averaging candidates under one promotion interface. |
| 48 | Appendix A4 annotation schemas for LV, linear dimensions, and AV Doppler. | Dataset/model modules represent many fields, but reader/acquisition/convention and exact source identity are not complete across every route. | Execution-only | Validate a versioned schema at ingestion and reject target-specific examples missing required evidence or calibration. |
| 49 | Appendix A5 controlled architecture/optimizer/augmentation/temporal/conformal matrix. | Architecture, PEFT, augmentation, temporal, and uncertainty pilots exist separately; the full fixed-variable matrix is incomplete. | Experiment pending | Pre-register each row, reuse identical splits/heads, and report effect size, robustness, calibration, and cost. |
| 50 | Appendix A6 release checklist. | Split seals, calibration/shortcut/OOF/provenance and external-lock controls cover part of the checklist; claim registry, full failure replay, external evidence, and Track C lineage remain open. | Execution-only | Turn every checklist item into a hashed required artifact and keep release blocked until all direct evidence passes. |
| 51 | Appendix A7 empirical requirements for a true 10/10 programme. | The repository explicitly records these as unmet; local retrospective training cannot resolve multi-reader repeatability, locked multicentre validation, or Track C parity. | External-blocked | Complete the listed controlled experiments, prospective study, one-touch external test, and clean Track C comparison. |
| 52 | Appendix A8 verified source register. | `DATASETS.md`, `THIRD_PARTY.md`, acquisition scripts, and research-data audits preserve source/role/licence metadata without publishing data or weights. | Execution-only | Re-verify terms and exact hashes at acquisition and before every redistribution or lineage decision. |

## Appendix label crosswalk

The PDF numbers the appendix pages 45-52 and labels them A1-A8. The rows below
make both identifier systems explicit.

| Appendix | Number | Repository mapping | Status | Next action |
| --- | ---: | --- | --- | --- |
| A1 Reference hyperparameters | 45 | Config and route-specific trainer defaults; not all recommended sweeps completed. | Execution-only | Log every measured override and promotion result. |
| A2 Augmentation configuration | 46 | Physical transform and task-policy code/tests; no G3 scientific pass. | Execution-only | Freeze only cross-domain winners. |
| A3 Training-loop pseudocode | 47 | Route-specific BF16/staged/EMA training loops and provenance/status outputs. | Execution-only | Unify source, tier, transform, and averaging manifests. |
| A4 Annotation schemas | 48 | Partial LV/landmark/chamber/scalar/spectral schema coverage. | Execution-only | Require complete target-specific evidence fields at ingestion. |
| A5 Ablation matrix | 49 | Separate pilots exist; controlled full matrix is unfinished. | Experiment pending | Run only pre-registered, fixed-variable comparisons. |
| A6 Release checklist | 50 | Partial automated checks; scientific release remains blocked. | Execution-only | Convert every unchecked item into required hashed evidence. |
| A7 True 10/10 gaps | 51 | Prospective, locked external, and clean Track C evidence are unavailable locally. | External-blocked | Collect those studies; no local substitute is valid. |
| A8 Source register | 52 | Repository dataset/third-party ledgers and source-aware acquisition audits. | Execution-only | Re-verify terms and hashes at use and release time. |

## Gate ledger

The gate engine's current aggregate status was read without opening private rows
or any sealed TEST payload.

| Gate | Current enforcement/evidence | Scientific status | Ledger status | Next action |
| --- | --- | --- | --- | --- |
| G0 Reproduction | Hardware, runtime, and EchoJEPA tensor smoke execution checks pass; `official_probe_reproduction.json` is absent. | Closed | Experiment pending | Reproduce the official/frozen probes within documented tolerance. |
| G1 Calibration | Inventory summary, calibration tests, and split-audit execution checks pass; direct `physical_calibration_gate.json` is absent. The v2 inventory replacement remains unpromoted. | Closed | Experiment pending | Complete the source-bound v2 scan and sampled spatial/spectral plausibility gate. |
| G2 Caliper pilot | Shortcut and clean-frame execution checks pass; `caliper_pilot_gate.json` is absent. | Closed | Experiment pending | Finish P1-P5 precision/yield and blinded clean-frame evidence. |
| G3 Augmentation | Physical parity execution check passes; `cross_domain_augmentation_gate.json` is absent. | Closed | Experiment pending | Prove cross-domain gain without calibration-sensitive regression. |
| G4 Architecture | Ladder and licence-provenance execution checks pass; `architecture_promotion_gate.json` is absent. | Closed | Experiment pending | Complete fixed-recipe B/B+/C/D/E margins and subgroup non-regression. |
| G5 Optimizer/PEFT | Optimizer/PEFT pilot and decision-ledger execution checks pass; `optimizer_peft_promotion_gate.json` is absent. | Closed | Experiment pending | Demonstrate external benefit or equal quality at materially lower cost. |
| G6 Specialist | Training and OOF artifacts exist, but specialist holdouts and failure detection declare failure. All scalar full-coverage targets miss; required temporal/vendor evidence is absent. | Closed | Experiment pending | Improve the specialists and pass matched accuracy, temporal, vendor, failure, calibration, and selective-risk criteria. |
| G7 Integration | `integration.json` exists and declares failure; G6 is not passed. | Closed | Experiment pending | Integrate only specialists that pass G6 and prove source/multiview improvement without hiding missing evidence. |
| G8 External lock | The lock artifact exists and declares failure because upstream evidence is incomplete; no locked test is opened. | Closed | External-blocked | Finalize and hash the analysis only after G7, then freeze before one-touch external testing. |
| G9 Commercial lineage | No passing commercial-lineage artifact exists; the present build is Research Track. | Closed | External-blocked | Train and audit a separate Track C lineage using only explicitly permitted ancestors. |

## Claim boundary

The current repository is a research execution system with promising but
incomplete development components. It is not clinically validated, not cleared
for autonomous diagnosis, and not demonstrated to be best in the world. Those
claims remain impossible until the locked multicentre and prospective
multi-reader repeat-acquisition evidence in v4 is collected and passes.
