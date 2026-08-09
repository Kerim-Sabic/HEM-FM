# Protocol coverage ledger

| Specification requirement | Enforcement | Repository state |
|---|---|---|
| Two-GPU Blackwell-class reference floor | Runtime assertions for GPU count, model, memory, compute capability, and BF16 | Implemented |
| BF16 first; FP8 only after parity | Precision policy; FP8 disabled by default | Implemented |
| Patient split before features | Deterministic patient grouping and fatal leakage audit | Implemented |
| Physical DICOM calibration | Region inventory, multi-region classification, unit conversion, and round-trip tests | Implemented |
| Transform chains | 3x3 homogeneous composition and subpixel round-trip tests | Implemented |
| Caliper/overlay shortcut controls | G2 execution subcheck | Implemented; not the full caliper-pilot gate |
| Clean-frame negative controls | G2 execution subcheck | Implemented; not the full caliper-pilot gate |
| OOF pseudo-labels only | Patient-grouped folds and versioned private outputs | Implemented for development; quality gates still pending |
| V-JEPA 2.1 primary / V-JEPA 2 functional / DINOv3 challenger | Asset pinning and G4 ladder evidence | Implemented |
| ViT-B search / ViT-L staged final / ViT-g teacher only | Frozen configuration contract | ViT-L six-endpoint staged scalar schedule completed; scientific promotion remains closed |
| AdamW+EMA baseline / Muon challenger | G5 execution subcheck | Implemented; external-benefit gate not established |
| LoRA, rsLoRA, DoRA, PiSSA | G5 execution subcheck | Implemented; external-benefit gate not established |
| Research / Commercial track separation | Provenance types reject restricted ancestry in Commercial Track | Implemented |
| Locked internal test | Separate execution/scientific sequential status and sealed-test policy | Implemented; test remains sealed |
| G6 specialist gate | Internal, temporal, vendor holdouts plus failure detection | Closed: all six scalar targets miss, learned failure detection fails, and required temporal/train-excluded-vendor holdouts are absent |
| G7 integration gate | Source selection and multi-view improvement without hiding missing evidence | `integration.json` is wired fail-closed and currently reports `passed: false` |
| G8 external lock | Pre-specified analysis finalized and external test frozen | `external_analysis_lock.json` is wired fail-closed and currently reports `passed: false` |
| G9 commercial lineage | Track C ancestry fully permitted and auditable | Research build only; not eligible to pass |
| True 10/10 evidence | Locked multi-centre test and 150-250 patient multi-reader repeat acquisition | Requires future external/prospective work |

## Gate status ledger

**Execution passed != clinical gate passed.** Existing smoke tests, pilots and
development reports remain useful execution subchecks. They authorize only the
next local research computation; they do not satisfy a HEM-FM v4 scientific
gate. `gates status` therefore reports scientific `passed` separately from
`execution_checks_passed`.

| Gate | Local execution evidence | Missing or failed v4 pass condition | Scientific status |
|---|---|---|---|
| G0 Reproduction | Hardware/runtime checks and EchoJEPA tensor smoke | Official/frozen probe metrics within documented tolerance | Closed |
| G1 Calibration | Inventory, unit/calibration tests and split audit | Separate consolidated physical-calibration gate evidence | Closed |
| G2 Caliper pilot | Shortcut probes and clean-frame negative control | Pre-set mining precision and clean-frame recovery yield | Closed |
| G3 Augmentation | Physical transform/label parity | Cross-domain improvement with no calibration-sensitive regression | Closed |
| G4 Architecture | Four-backbone development ladder and provenance | Pre-registered B/B+/C/D/E margin with no critical subgroup regression | Closed |
| G5 Optimizer/PEFT | AdamW/Muon and PEFT execution pilot plus decision ledger | External benefit or equal quality at materially lower cost | Closed |
| G6 Specialist | Three-seed specialists, OOF outputs, holdout and failure audits | Accuracy, temporal, train-excluded-vendor and failure-detection conditions | Closed |
| G7 Integration | Fail-closed study integration audit | G6 and complete integration conditions | Closed |
| G8 External lock | Pre-specified lock builder | Passing G7 before the analysis can freeze | Closed |
| G9 Commercial lineage | Research-track provenance enforcement | Fully permitted and auditable Track C ancestry | Closed |

The completed staged scalar development ensembles are EF 6.363708 percentage
points, LVEDV 20.093339 mL, LVESV 13.168489 mL, LVOT 1.417459 mm, RV basal
4.480584 mm and AV Vmax 0.500563 m/s MAE. The corresponding v4 full-coverage
targets are 4.00, 12.00, 9.00, 1.20, 2.80 and 0.18, respectively; all six miss.

Storage policy: GPU compute and every mutable artifact are local to the training
workstation. A corpus may be mounted from a read-only volume when it cannot fit
locally, but no training output is written back to that source.

The launch command cannot train through missing local execution prerequisites:
it raises an error and lists absent subcheck files. A command-line
acknowledgement cannot convert those subchecks into passed scientific evidence.
