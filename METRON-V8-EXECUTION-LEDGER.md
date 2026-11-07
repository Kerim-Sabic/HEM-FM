# METRON v8 execution ledger

Status: Stage 0 readiness started. No METRON pixels decoded and no GPU training launched.

## Protocol authority

- METRON v8 is the master accuracy and measurement protocol.
- HEM-FM v7 supplies the detailed dataset, architecture and stage implementation.
- HEM-FM v4 remains a reproducible baseline. Its locked TEST assets stay sealed.
- All targets in the source specifications are preregistered goals, not achieved results.

## Corrected reference-standard policy

UnityLV-MultiX is sealed evaluation-only. It is not used to fit the reader-noise
model and is not used for tuning. Development reader noise must come from
patient-disjoint development rater audits and development multi-rater subsets.
This removes the v8 ambiguity in which MultiX both influenced training and was
described as an untouched evaluation noise floor.

## Programme order

| Stage | Work | GPU-hours | Gate state |
|---|---|---:|---|
| S0 | Ingest, calibration, geometry, splits, caches, baseline reproduction | 300 | Readiness audit running |
| S0.5 | Reader-noise model and soft targets | 20 | Locked |
| S1 | Domain-adaptive self-supervision | 1,100 | Locked |
| S2 | Classification stack | 150 | Locked |
| S3 | Polar, SDF and mandatory CNN segmentation arms | 650 | Locked |
| S4 | Hybrid landmarks, cascade and mining | 600 | Locked |
| S5 | Doppler heads | 250 | Locked |
| S6 | Motion and biomechanics | 550 | Locked |
| S7 | Semi-supervised expansion | 650 | Locked |
| S8 | Joint tuning, calibration and abstention | 380 | Locked |
| S9 | METRON-B and METRON-Edge distillation | 200 | Locked |

## Non-negotiable controls

1. Patient-level identity precedes all decoding and caching.
2. No locked TEST pixel may be read during development.
3. Every transform carries the physical affine in the same operation.
4. Restricted data and weights are never placed in the public repository.
5. A failed gate stops the queue. Epoch count cannot override a failed gate.
6. Every metric reports patient-level intervals, coverage and abstention reasons.
7. No clinical-readiness or superiority claim is made from this public-data build.

## 2026-08-11 storage audit

- The workstation exposes one local drive: 1,862.2 GiB total and 48.8 GiB free.
- Existing HEM-FM runtime assets occupy about 214.5 GiB and are retained. They
  already include CAMUS, CardiacNet-PAH/ASD, CardiacUDA, EchoCP,
  EchoNet-Dynamic, EchoXFlow demo, EV9V, HMC-QU, MIMIC LV extension, TED and
  Unity assets plus checkpoints and evidence.
- Confirmed game installation content totals 396.85 GiB. The exact audited
  cleanup targets are encoded in `scripts/reclaim_game_space.ps1`; save-data
  locations are excluded.
- Removing every audited game would raise projected free space only to about
  445.7 GiB. That is still below the approximately 515 GB S0 cache estimate and
  cannot satisfy the full programme's raw-data and cache footprint.
- A local 4 TB-or-larger volume remains the correct capacity fix. A rolling,
  delete-and-redownload design would reduce reproducibility and training
  throughput and is not adopted silently as an equivalent protocol.
