# Dataset access handoff

These datasets require an individual account, an agreement, or credentialed access. Complete the steps at the official links below. Leave every completed download in the normal Windows **Downloads** folder with its original filename; do not rename or extract it. Before any training job can use a package, it will be hashed, inventoried, licence-checked, and smoke-tested in place. It will be copied into controlled local storage only when required and after a free-space check.

## 1. EchoRisk

- Project and access: <https://www.synapse.org/Synapse%3Asyn72001386>
- Study overview: <https://echorisk-miccai.github.io/>
- Action: sign in to Synapse, register or join the required team, accept the terms, and download only the released training/validation material.
- Guardrail: do not download, inspect, or expose the hidden final-test labels. EchoRisk is reserved for prespecified multi-centre and temporal external validation, not general pretraining.

## 2. EchoNet-Dynamic

- Dataset portal: <https://stanford.redivis.com/datasets/66s1-2hsmzj5rn>
- Research-use agreement: <https://echonet.github.io/dynamic/>
- Action: sign in, complete the individual Stanford agreement, and download the official package.
- Role: apical four-chamber LV segmentation, cardiac-phase, and ejection-fraction challenger. Do not treat its videos as physically calibrated geometry unless the release provides the required calibration evidence.

## 3. EchoNet-LVH

- Dataset portal: <https://stanford.redivis.com/datasets/cchq-0srz1fy9a>
- Research-use agreement: <https://echonet.github.io/lvh/>
- Action: sign in, complete the individual agreement, and download the official package.
- Role: parasternal long-axis wall/chamber landmark challenger. Keep it in the research track under its non-commercial terms.

## 4. EchoNet-Pediatric

- Dataset portal: <https://stanfordaimi.azurewebsites.net/datasets/a84b6be6-0d33-41f9-8996-86e5df53b005>
- Research-use agreement: <https://echonet.github.io/pediatric/index.html>
- Action: sign in, accept the individual non-commercial research-use agreement, and download the official package.
- Role: a separate pediatric age/size-shift analysis. It must not be naively mixed into the adult training distribution or used to claim adult generalization.

## 5. ECHOVIEW

- Dataset portal: <https://physionet.org/content/echoview/0.1/>
- Action: use a credentialed PhysioNet account, complete the required CITI Data or Specimens research training, and accept the data-use agreement.
- Role: a weak MIMIC view-routing challenger. Its labels are model predictions with a limited clinician-reviewed sample, so it is not ground truth for clinical endpoints.

## After each download

1. Keep the original archive name and leave it in **Downloads**.
2. Do not extract or edit the package.
3. Message the exact filename that finished downloading.
4. The pipeline will compute hashes, check licensing/provenance, inventory patients and studies, and run a smoke test before a full job is allowed. Any staging copy is made only after checking local free space.

