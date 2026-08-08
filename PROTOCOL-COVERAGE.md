# Protocol coverage ledger

| Specification requirement | Enforcement | Repository state |
|---|---|---|
| Two-GPU Blackwell-class reference floor | Runtime assertions for GPU count, model, memory, compute capability, and BF16 | Implemented |
| BF16 first; FP8 only after parity | Precision policy; FP8 disabled by default | Implemented |
| Patient split before features | Deterministic patient grouping and fatal leakage audit | Implemented |
| Physical DICOM calibration | Region inventory, multi-region classification, unit conversion, and round-trip tests | Implemented |
| Transform chains | 3x3 homogeneous composition and subpixel round-trip tests | Implemented |
| Caliper/overlay shortcut controls | Required G2 evidence | Implemented |
| Clean-frame negative controls | Required G2 evidence | Implemented |
| OOF pseudo-labels only | Patient-grouped folds and versioned private outputs | Implemented for development; quality gates still pending |
| V-JEPA 2.1 primary / V-JEPA 2 functional / DINOv3 challenger | Asset pinning and G4 ladder evidence | Implemented |
| ViT-B search / ViT-L staged final / ViT-g teacher only | Frozen configuration contract | Partially implemented; full staged final tuning remains future work |
| AdamW+EMA baseline / Muon challenger | Required G5 evidence | Implemented |
| LoRA, rsLoRA, DoRA, PiSSA | Required G5 evidence | Implemented |
| Research / Commercial track separation | Provenance types reject restricted ancestry in Commercial Track | Implemented |
| Locked internal test | Sequential gates and sealed-test policy | Implemented |
| G6 specialist gate | Internal, temporal, vendor holdouts plus failure detection | Internal audit, CAMUS dense challengers, MIMIC LV extension, TED temporal, Unity landmarks, EV9V routing, and external OOD evaluator implemented; promotion evidence remains run-dependent |
| G7 integration gate | Source selection and multi-view improvement without hiding missing evidence | Fail-closed audit implemented; pass remains dependent on G6 and routing/temporal results |
| G8 external lock | Pre-specified analysis finalized and external test frozen | Analysis-lock artifact implemented; it cannot freeze until G7 passes |
| G9 commercial lineage | Track C ancestry fully permitted and auditable | Research build only; not eligible to pass |
| True 10/10 evidence | Locked multi-centre test and 150-250 patient multi-reader repeat acquisition | Requires future external/prospective work |

Storage policy: GPU compute and every mutable artifact are local to the training
workstation. A corpus may be mounted from a read-only volume when it cannot fit
locally, but no training output is written back to that source.

The launch command cannot train through missing evidence: it raises an error and
lists absent gate files. A command-line acknowledgement cannot convert missing
evidence into passed evidence.

