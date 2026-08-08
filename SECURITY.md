# Security and responsible data handling

This project is designed for controlled medical-imaging research environments.

Never commit, upload, or attach:

- DICOM files or any other patient-level clinical data;
- subject, study, accession, or series identifiers;
- private manifests, pseudo-labels, feature caches, thresholds, or evidence files;
- model weights trained on restricted data;
- credentials, tokens, private URLs, or machine-specific configuration.

Keep `configs/protocol.yaml` private. The public template is
`configs/protocol.example.yaml`. Treat every derived artifact as restricted until
its provenance and disclosure policy have been reviewed.

The code is research software, not a medical device. Do not use outputs for
clinical decisions. External validation and prospective evidence are required
before making performance or generalization claims.

If sensitive data is accidentally committed, stop publication, revoke exposed
credentials if applicable, and remove the data from the entire Git history before
resuming work.

