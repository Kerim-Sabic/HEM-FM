# Dataset roles and promotion policy

The pipeline is designed to use more data only when it improves a predeclared
capability without degrading calibrated core endpoints. Dataset availability is
not, by itself, a reason to merge it into training.

| Dataset | Local role | Access / licence boundary | Promotion rule |
|---|---|---|---|
| MIMIC-IV-ECHO | Primary research corpus for six calibrated scalar specialists | Credentialed PhysioNet data; never redistributed | Core patient-disjoint validation and gate evidence |
| MIMIC LV-volume extension | Direct-video EF, LVEDV, and LVESV challenger | Credentialed PhysioNet data; development cines staged locally, internal test sealed | Improve normalized development MAE without degrading uncertainty or physiology checks |
| CAMUS | Dense LV cavity, myocardium, and atrium initialization | Public research dataset; observe its licence and citation terms | Mean foreground Dice on official development validation |
| TED | Full-cycle temporal consistency | CC BY-NC-SA 4.0 plus dataset terms | Improve temporal validation while CAMUS-linked test identities remain sealed |
| Unity | Multi-view anatomical landmarks and physical-distance evaluation | CC BY-NC-SA 4.0 for images, labels, metadata, and released weights; MIT code | Development-only point accuracy, visibility, and physical-error checks |
| EV9V | Nine-view routing and acquisition-quality control | CC BY 4.0 | Exam-disjoint balanced accuracy and macro-F1 must both be at least 0.75 |
| CardiacUDA | Multi-centre A4C domain stress | Apache 2.0 | External failure/segmentation evidence only until public partition semantics are unambiguous |
| EchoCP | Contrast-bubble out-of-distribution stress | Apache 2.0 | Failure/abstention evaluation only; never updates scalar heads |
| CardiacNet PAH/ASD | Disease-specific out-of-distribution stress | Public research release; verify author/distributor terms | Failure/abstention evaluation only; never updates scalar heads |
| HMC-QU | Metadata and mask-shape audit | Published CC BY-NC-SA terms | Excluded from image training because the public package contains no raw cines |
| EchoXFlow demo | Zarr, Croissant, beamspace, Doppler, and ECG format smoke test | CC BY-NC-SA 4.0 | Format checks only: one examination is not representative performance evidence |
| EchoNet-Dynamic / LVH | High-priority controlled-access challengers | Researcher registration and use agreement required | Run only after authorized acquisition and a frozen patient-disjoint analysis plan |
| EchoNet-Pediatric | Pediatric age/size-shift validation and LV-function challenger | Individual Stanford registration; non-commercial research-use agreement | Keep pediatric evaluation separate from adult deployment; promote only under a predeclared pediatric analysis |
| ECHOVIEW | MIMIC view-router calibration and audit labels | Credentialed PhysioNet access and DUA | Treat its 23-class predictions as weak labels, not ground truth; clinician validation was limited to a small sample |
| EchoRisk | Multi-centre temporal external validation | Synapse registration/team access required | Hidden challenge test remains sealed until the external analysis is frozen |

No dataset, checkpoint, feature cache, patient identifier, or locally generated
evidence file belongs in the public repository. See `SECURITY.md` and
`THIRD_PARTY.md` before publishing code or weights.

## Released model challengers

| Model | Intended comparison | Licence / lineage rule | Integration decision |
|---|---|---|---|
| EchoJEPA / V-JEPA 2.1 | Primary temporal and dense MIMIC-pretrained backbones | Research Track ancestry; preserve checkpoint and source revisions | Already included in the frozen ladder and task-specific challengers |
| DINOv3 | General visual challenger for dense anatomy and landmarks | Preserve Meta model terms and exact checkpoint hash | Promote per endpoint only after patient-disjoint held-out improvement |
| [EchoPrime](https://github.com/echonet/EchoPrime) | View-informed study aggregation and source selection | Official repository is MIT; preserve model-data revision and research provenance | Study-level challenger; never allowed to hide missing required evidence |
| [PanEcho](https://github.com/CarDS-Yale/PanEcho) | Frozen 39-task, view-agnostic reference | Weights are CC BY-NC-SA 4.0; code is AGPLv3; Research Track only | Keep as an external benchmark unless a separate compatible integration is justified and beats held-out baselines |
| [EchoFM](https://github.com/SekeunKim/EchoFM) | Standalone frozen representation comparison | CC BY-NC-ND 4.0; the authors prohibit commercial use and derivatives, including models trained on its outputs | Do not fine-tune, distil, or train HEM-FM on EchoFM outputs; no integration without written permission |

