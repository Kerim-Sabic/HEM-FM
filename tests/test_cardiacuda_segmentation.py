from __future__ import annotations

import copy
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

import hemfm.cardiacuda_segmentation as cardiacuda_segmentation
from hemfm.cardiacuda_segmentation import (
    BASELINE_SCOPE,
    CardiacUDCFrameDataset,
    GLOBAL_CLASS_TO_INDEX,
    IGNORE_INDEX,
    TRANSFORM_POLICY,
    FrameRecord,
    SealedTestError,
    ViewConditionedUNet,
    VolumeRecord,
    active_class_mask,
    apply_spatial_transform,
    audit_optional_cardiacnet,
    build_manifest,
    export_checkpoint_overlay,
    export_contour_overlay,
    invert_spatial_transform,
    remap_mask,
    segmentation_loss,
    split_domain_disjoint,
    train,
    validate_transform_record,
)


def _record(domain: str, patient: str, *, sealed: bool = False) -> VolumeRecord:
    return VolumeRecord(
        image_path=Path(f"/{domain}/{patient}-4_image.nii.gz"),
        label_path=Path(f"/{domain}/{patient}-4_label.nii.gz"),
        cohort=domain,
        domain=domain,
        patient_id=f"{domain}:{patient}",
        view="A4C",
        sealed=sealed,
    )


def _write_pair(directory: Path, stem: str, *, image_offset: float = 0.0) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    image = np.linspace(0, 255, 32 * 24 * 3, dtype=np.float32).reshape(32, 24, 3) + image_offset
    label = np.zeros((32, 24, 3), dtype=np.int16)
    label[4:14, 3:9, 1] = 1
    label[16:27, 3:9, 1] = 2
    label[16:27, 14:21, 1] = 3
    label[4:14, 14:21, 1] = 4
    # The authors' loader requires raw mask sum >100. Exactly 100 is noise, not
    # an annotated frame; this catches accidental >= or any-nonzero filtering.
    label[0:5, 0:5, 0] = 4
    nib.save(nib.Nifti1Image(image, np.eye(4)), directory / f"{stem}-4_image.nii.gz")
    nib.save(nib.Nifti1Image(label, np.eye(4)), directory / f"{stem}-4_label.nii.gz")


def test_a4c_raw_labels_are_remapped_to_global_chambers() -> None:
    raw = np.array([[0, 1, 2, 3, 4]], dtype=np.int16)

    remapped = remap_mask(raw, "A4C")

    assert remapped.tolist() == [[0, 1, 3, 4, 2]]
    assert active_class_mask("A4C").tolist() == [True, True, True, True, True, False]
    assert active_class_mask("PA").tolist() == [True, False, False, False, False, True]


def test_invalid_view_labels_fail_closed() -> None:
    with pytest.raises(ValueError, match="Unexpected PLAX label"):
        remap_mask(np.array([[4]], dtype=np.int16), "PLAX")
    with pytest.raises(ValueError, match="Unsupported view"):
        remap_mask(np.array([[1]], dtype=np.int16), "A2C")


def test_missing_structures_are_ignored_in_loss_not_background() -> None:
    torch.manual_seed(7)
    logits = torch.randn(1, 6, 8, 8)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, 1:4, 1:4] = GLOBAL_CLASS_TO_INDEX["LV"]
    target[:, 4:7, 4:7] = GLOBAL_CLASS_TO_INDEX["RV"]
    class_valid = torch.from_numpy(active_class_mask("PLAX")).unsqueeze(0)
    outputs = {"logits": logits.clone(), "log_variance": torch.zeros(1, 1, 8, 8)}
    changed = logits.clone()
    changed[:, [3, 4, 5]] = torch.tensor([900.0, -900.0, 500.0]).view(1, 3, 1, 1)

    loss_a, parts = segmentation_loss(outputs, target, class_valid)
    loss_b, _ = segmentation_loss(
        {"logits": changed, "log_variance": torch.zeros(1, 1, 8, 8)}, target, class_valid
    )

    assert torch.allclose(loss_a, loss_b, atol=1e-6)
    assert set(parts) == {"heteroscedastic_focal", "dice_loss", "boundary_loss"}


def test_model_is_view_conditioned_and_outputs_uncertainty() -> None:
    model = ViewConditionedUNet(base_channels=4)
    images = torch.randn(2, 1, 64, 64)
    views = torch.tensor([0, 1], dtype=torch.long)

    outputs = model(images, views)

    assert outputs["logits"].shape == (2, 6, 64, 64)
    assert outputs["log_variance"].shape == (2, 1, 64, 64)


def test_domain_disjoint_split_excludes_sealed_data() -> None:
    records = [_record("Site_G", "g1"), _record("Site_G", "g2"), _record("Site_R", "r1")]

    train, validation, selected = split_domain_disjoint(records, "R")

    assert selected == "Site_R"
    assert {item.domain for item in train} == {"Site_G"}
    assert {item.domain for item in validation} == {"Site_R"}
    assert not ({item.patient_id for item in train} & {item.patient_id for item in validation})
    with pytest.raises(SealedTestError):
        split_domain_disjoint([*records, _record("sealed", "locked", sealed=True)], "R")


def test_manifest_never_decodes_sealed_directory(tmp_path: Path) -> None:
    _write_pair(tmp_path / "Site_G_100", "patient1")
    _write_pair(tmp_path / "Site_R_126", "patient-11", image_offset=7.0)
    sealed = tmp_path / "label_all_frame"
    sealed.mkdir()
    (sealed / "locked-4_image.nii.gz").write_bytes(b"not-a-nifti")
    (sealed / "locked-4_label.nii.gz").write_bytes(b"not-a-nifti")

    manifest = build_manifest(tmp_path)

    assert len(manifest.train_frames) == 1
    assert len(manifest.validation_frames) == 1
    assert len(manifest.sealed_volumes) == 1
    report = manifest.report()
    assert report["checks"]["sealed_test_not_decoded"] is True
    assert report["annotated_frame_criterion"] == {
        "rule": "sum(raw_mask_values) > threshold",
        "threshold": 100.0,
        "source": cardiacuda_segmentation.AUTHOR_CODE_URL,
    }
    sample = CardiacUDCFrameDataset(manifest.train_frames, resolution=32)[0]
    assert sample["image"].shape == (1, 32, 32)
    assert set(torch.unique(sample["target"]).tolist()) == {IGNORE_INDEX, 0, 1, 2, 3, 4}
    transform_record = json.loads(sample["transform_json"])
    assert transform_record["augmentations"][0] == {
        "name": "horizontal_flip",
        "enabled": False,
        "applied": False,
        "parity_flag": False,
    }
    assert manifest.report()["content_deduplication"]["cross_partition_duplicates"] == 0
    first_fingerprint = manifest.fingerprint
    (sealed / "second-4_image.nii.gz").write_bytes(b"still-not-a-nifti")
    (sealed / "second-4_label.nii.gz").write_bytes(b"still-not-a-nifti")
    changed_sealed_inventory = build_manifest(tmp_path)
    assert changed_sealed_inventory.fingerprint != first_fingerprint
    assert changed_sealed_inventory.sealed_inventory["pixel_bytes_read"] is False


def test_development_dataset_rejects_sealed_frames() -> None:
    sealed = _record("sealed", "locked", sealed=True)
    with pytest.raises(SealedTestError):
        CardiacUDCFrameDataset([FrameRecord(sealed, 0)])
    assert not hasattr(cardiacuda_segmentation, "SEALED_TEST_TOKEN")
    assert not hasattr(cardiacuda_segmentation, "unlock_sealed_test")


def test_sealed_paths_fail_before_overlay_or_dataset_decode(tmp_path: Path) -> None:
    locked = tmp_path / "label_all_frame" / "locked-4_image.nii.gz"
    disguised = VolumeRecord(
        image_path=locked,
        label_path=locked.with_name("locked-4_label.nii.gz"),
        cohort="development",
        domain="Site_G",
        patient_id="development:locked",
        view="A4C",
        sealed=False,
    )

    with pytest.raises(SealedTestError):
        CardiacUDCFrameDataset([FrameRecord(disguised, 0)])
    with pytest.raises(SealedTestError):
        export_checkpoint_overlay(
            tmp_path / "missing.pt",
            locked,
            tmp_path / "overlay.png",
            frame_index=0,
            view="A4C",
            device="cpu",
        )
    with pytest.raises(SealedTestError):
        cardiacuda_segmentation._nifti_digest(locked)


def test_overlay_refuses_a_view_absent_from_checkpoint_training(tmp_path: Path) -> None:
    model = ViewConditionedUNet(base_channels=4)
    checkpoint = tmp_path / "a4c-only.pt"
    torch.save(
        {
            "scope": BASELINE_SCOPE,
            "schema_version": 2,
            "settings": {"base_channels": 4, "resolution": 32},
            "state_dict": model.state_dict(),
            "trained_views": ["A4C"],
            "global_classes": cardiacuda_segmentation.GLOBAL_CLASSES,
            "view_names": cardiacuda_segmentation.VIEW_NAMES,
            "sealed_test_accessed": False,
            "transform_policy_fingerprint": cardiacuda_segmentation._policy_fingerprint(
                TRANSFORM_POLICY
            ),
            "config": {
                "scope": BASELINE_SCOPE,
                "view": "A4C",
                "settings": {"base_channels": 4, "resolution": 32},
                "purpose": "adversarial-test",
            },
        },
        checkpoint,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["config_fingerprint"] = cardiacuda_segmentation._canonical_fingerprint(
        payload["config"]
    )
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="not trained for PLAX"):
        export_checkpoint_overlay(
            checkpoint,
            tmp_path / "public-1_image.nii.gz",
            tmp_path / "overlay.png",
            frame_index=0,
            view=None,
            device="cpu",
        )


def test_validation_dice_ignores_unlabelled_pixels() -> None:
    class FixedModel:
        def eval(self):
            return self

        def __call__(self, images, views):
            logits = torch.full((1, 6, 2, 2), -10.0)
            logits[:, 0] = 10.0
            logits[:, 1, 0, 0] = 20.0  # True LV.
            logits[:, 1, 0, 1] = 20.0  # Prediction in ignored region.
            return {"logits": logits, "log_variance": torch.zeros(1, 1, 2, 2)}

    loader = [
        {
            "image": torch.zeros(1, 1, 2, 2),
            "target": torch.tensor([[[1, -100], [0, 0]]]),
            "view_id": torch.tensor([0]),
            "class_valid": torch.from_numpy(active_class_mask("A4C")).unsqueeze(0),
            "patient_id": ["patient-one"],
        }
    ]

    metrics = cardiacuda_segmentation._evaluate(FixedModel(), loader, "cpu")

    assert metrics["dice"]["LV"] == pytest.approx(1.0)


def test_overlay_contains_contours_and_uncertainty(tmp_path: Path) -> None:
    from PIL import Image

    image = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    prediction = np.zeros((32, 32), dtype=np.int64)
    prediction[5:15, 5:15] = GLOBAL_CLASS_TO_INDEX["LV"]
    prediction[17:27, 17:27] = GLOBAL_CLASS_TO_INDEX["RV"]
    uncertainty = np.zeros((32, 32), dtype=np.float32)
    uncertainty[:, 16:] = 1.0
    output = tmp_path / "overlay.png"

    export_contour_overlay(image, prediction, uncertainty, output, view="A4C")

    assert output.read_bytes().startswith(b"\x89PNG")
    rendered = np.asarray(Image.open(output))[:32]
    assert np.any(np.all(rendered == np.array([0, 230, 255]), axis=2))
    assert np.any(np.all(rendered == np.array([255, 175, 0]), axis=2))
    assert len(set(int(value) for value in rendered[0, 31])) > 1


def test_cardiacnet_disease_videos_are_not_treated_as_pixel_masks(tmp_path: Path) -> None:
    disease = tmp_path / "CardiacNet-ASD" / "ASD"
    disease.mkdir(parents=True)
    (disease / "001_label.nii").write_bytes(b"video-placeholder")

    audit = audit_optional_cardiacnet(tmp_path)

    assert audit["present"] is True
    assert audit["eligible_for_segmentation"] is False
    assert audit["pixel_mask_pairs"] == 0


def test_letterbox_geometry_is_recorded_and_invertible_without_shape_distortion() -> None:
    image = np.arange(20 * 40, dtype=np.float32).reshape(20, 40)
    target = np.zeros((20, 40), dtype=np.int16)
    target[4:16, 10:30] = 1

    resized_image, resized_target, transform = apply_spatial_transform(image, target, 80)

    assert resized_image.shape == (80, 80)
    assert transform.resized_height == 40
    assert transform.resized_width == 80
    assert transform.scale_x == transform.scale_y == 2.0
    assert (transform.pad_top, transform.pad_bottom) == (20, 20)
    assert np.all(resized_target[:20] == IGNORE_INDEX)
    assert np.array_equal(
        invert_spatial_transform(resized_target, transform, mode="nearest"), target
    )
    forward = np.asarray(transform.forward_matrix)
    inverse = np.asarray(transform.inverse_matrix)
    assert np.allclose(inverse @ forward, np.eye(3), atol=1e-9)

    large_image = np.repeat(np.repeat(image, 4, axis=0), 4, axis=1)
    large_target = np.repeat(np.repeat(target, 4, axis=0), 4, axis=1)
    _, downscaled_target, downscale = apply_spatial_transform(
        large_image, large_target, 40
    )
    assert (downscale.resized_height, downscale.resized_width) == (20, 40)
    assert downscale.scale_x == downscale.scale_y == 0.25
    restored = invert_spatial_transform(downscaled_target, downscale, mode="nearest")
    assert restored.shape == large_target.shape
    assert np.array_equal(restored, large_target)


def test_transform_log_rejects_unproven_flip_and_geometry_tampering(tmp_path: Path) -> None:
    _write_pair(tmp_path, "patient1")
    record = VolumeRecord(
        image_path=tmp_path / "patient1-4_image.nii.gz",
        label_path=tmp_path / "patient1-4_label.nii.gz",
        cohort="Site_G_100",
        domain="Site_G",
        patient_id="Site_G_100:patient1",
        view="A4C",
    )
    frame = CardiacUDCFrameDataset([FrameRecord(record, 1)], resolution=32, augment=True)[0]
    logged = json.loads(frame["transform_json"])
    validate_transform_record(logged)
    assert logged["augmentations"][0]["enabled"] is False
    assert logged["augmentations"][0]["applied"] is False

    flipped = json.loads(frame["transform_json"])
    flipped["augmentations"][0]["applied"] = True
    with pytest.raises(ValueError, match="horizontal flip"):
        validate_transform_record(flipped)

    malformed = json.loads(frame["transform_json"])
    malformed["spatial"]["pad_left"] += 1
    with pytest.raises(ValueError, match="geometry is inconsistent"):
        validate_transform_record(malformed)

    forged_matrix = json.loads(frame["transform_json"])
    forged_matrix["spatial"]["forward_matrix"] = np.eye(3).tolist()
    forged_matrix["spatial"]["inverse_matrix"] = np.eye(3).tolist()
    with pytest.raises(ValueError, match="does not encode"):
        validate_transform_record(forged_matrix)


def test_exact_decoded_content_leakage_across_domains_fails_closed(tmp_path: Path) -> None:
    _write_pair(tmp_path / "Site_G_100", "patient1")
    _write_pair(tmp_path / "Site_R_126", "patient2")
    sealed = tmp_path / "label_all_frame"
    sealed.mkdir()
    (sealed / "locked-4_image.nii.gz").write_bytes(b"must-not-decode")
    (sealed / "locked-4_label.nii.gz").write_bytes(b"must-not-decode")

    with pytest.raises(ValueError, match="Exact decoded image-content leakage"):
        build_manifest(tmp_path)


def test_content_dedup_canonicalizes_numeric_dtype_and_rejects_validation_duplicates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cross-dtype"
    train_directory = root / "Site_G_100"
    validation_directory = root / "Site_R_126"
    train_directory.mkdir(parents=True)
    validation_directory.mkdir(parents=True)
    numeric = np.arange(24 * 16 * 2, dtype=np.int16).reshape(24, 16, 2)
    label = np.zeros_like(numeric)
    label[2:14, 2:10, 1] = 1
    label[14:22, 2:10, 1] = 4
    for directory, dtype in ((train_directory, np.int16), (validation_directory, np.float32)):
        nib.save(
            nib.Nifti1Image(numeric.astype(dtype), np.eye(4)),
            directory / "patient1-4_image.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(label.astype(dtype), np.eye(4)),
            directory / "patient1-4_label.nii.gz",
        )
    with pytest.raises(ValueError, match="Exact decoded image-content leakage"):
        build_manifest(root)

    within = tmp_path / "within-validation"
    _write_pair(within / "Site_G_100", "train", image_offset=19.0)
    _write_pair(within / "Site_R_126", "validation-one")
    _write_pair(within / "Site_R_126", "validation-two")
    with pytest.raises(ValueError, match="duplicates within"):
        build_manifest(within)


def test_unapproved_invalid_development_volume_cannot_be_silently_dropped(
    tmp_path: Path,
) -> None:
    _write_pair(tmp_path / "Site_G_100", "invalid")
    _write_pair(tmp_path / "Site_R_126", "valid", image_offset=5.0)
    invalid_label_path = tmp_path / "Site_G_100" / "invalid-4_label.nii.gz"
    invalid_label = np.zeros((32, 24, 3), dtype=np.int16)
    nib.save(nib.Nifti1Image(invalid_label, np.eye(4)), invalid_label_path)

    with pytest.raises(ValueError, match="unapproved corrupt/invalid"):
        build_manifest(tmp_path)


def test_unexpected_missing_or_orphan_labels_cannot_shrink_development_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "missing-label"
    _write_pair(root / "Site_G_100", "keep")
    _write_pair(root / "Site_G_100", "missing", image_offset=3.0)
    _write_pair(root / "Site_R_126", "validation", image_offset=9.0)
    (root / "Site_G_100" / "missing-4_label.nii.gz").unlink()

    with pytest.raises(ValueError, match="Unexpected missing development label"):
        build_manifest(root)

    orphan_root = tmp_path / "orphan-label"
    _write_pair(orphan_root / "Site_G_100", "train")
    _write_pair(orphan_root / "Site_R_126", "validation", image_offset=4.0)
    orphan = np.zeros((8, 8, 1), dtype=np.int16)
    nib.save(
        nib.Nifti1Image(orphan, np.eye(4)),
        orphan_root / "Site_G_100" / "orphan-4_label.nii.gz",
    )
    with pytest.raises(ValueError, match="label has no paired image"):
        build_manifest(orphan_root)

    complete_root = tmp_path / "complete-pair-removal"
    _write_pair(complete_root / "Site_G_100", "keep")
    _write_pair(complete_root / "Site_G_100", "removed", image_offset=6.0)
    _write_pair(complete_root / "Site_R_126", "validation", image_offset=12.0)
    approved = cardiacuda_segmentation._development_file_inventory(complete_root)
    monkeypatch.setitem(
        cardiacuda_segmentation.APPROVED_DEVELOPMENT_FILE_INVENTORIES,
        approved["fingerprint"],
        {"entry_count": approved["entry_count"], "dataset": "synthetic complete fixture"},
    )
    (complete_root / "Site_G_100" / "removed-4_image.nii.gz").unlink()
    (complete_root / "Site_G_100" / "removed-4_label.nii.gz").unlink()
    shrunken_manifest = build_manifest(complete_root)
    assert shrunken_manifest.development_file_inventory["approved"] is False
    with pytest.raises(ValueError, match="complete CardiacUDC development-file inventory"):
        train(
            complete_root,
            tmp_path / "blocked-run",
            mode="smoke",
            seed=7,
            device="cpu",
        )


def test_evaluation_aggregates_by_patient_and_bootstraps_dice_hd95_boundary() -> None:
    class TwoPatientModel:
        def eval(self):
            return self

        def __call__(self, images, views):
            logits = torch.full((2, 6, 4, 4), -10.0)
            logits[:, 0] = 10.0
            logits[0, 1, 0, 0] = 30.0  # Patient one is perfect for LV.
            return {"logits": logits, "log_variance": torch.zeros(2, 1, 4, 4)}

    targets = torch.zeros(2, 4, 4, dtype=torch.long)
    targets[0, 0, 0] = 1
    targets[1, 1:3, 1:3] = 1
    loader = [
        {
            "image": torch.zeros(2, 1, 4, 4),
            "target": targets,
            "view_id": torch.tensor([0, 0]),
            "class_valid": torch.from_numpy(active_class_mask("A4C")).repeat(2, 1),
            "patient_id": ["patient-one", "patient-two"],
        }
    ]

    metrics = cardiacuda_segmentation._evaluate(
        TwoPatientModel(), loader, "cpu", bootstrap_seed=91, bootstrap_replicates=200
    )

    assert metrics["patient_level"]["patient_count"] == 2
    assert metrics["dice"]["LV"] == pytest.approx(0.5)
    assert metrics["hd95_common_grid_pixels"]["LV"] > 0
    assert metrics["boundary_dice_2px_common_grid"]["LV"] == pytest.approx(0.5)
    summary = metrics["patient_level"]["per_class"]["LV"]["dice"]
    assert summary["unit"] == "patient"
    assert summary["patients"] == 2
    assert summary["replicates"] == 200
    assert summary["lower"] <= summary["estimate"] <= summary["upper"]
    assert set(metrics["patient_level"]["per_class"]) == {"LV"}
    assert (
        metrics["patient_level"]["undefined_reference_absent_class_patient_pairs_excluded"]
        == 6
    )

    missing_ids = dict(loader[0])
    missing_ids.pop("patient_id")
    with pytest.raises(ValueError, match="requires explicit patient identifiers"):
        cardiacuda_segmentation._evaluate(
            TwoPatientModel(), [missing_ids], "cpu", bootstrap_replicates=20
        )


def test_smoke_checkpoint_is_complete_resumable_and_fingerprint_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    output = tmp_path / "run"
    _write_pair(root / "Site_G_100", "patient1")
    _write_pair(root / "Site_R_126", "patient2", image_offset=11.0)
    inventory = cardiacuda_segmentation._development_file_inventory(root)
    monkeypatch.setitem(
        cardiacuda_segmentation.APPROVED_DEVELOPMENT_FILE_INVENTORIES,
        inventory["fingerprint"],
        {"entry_count": inventory["entry_count"], "dataset": "synthetic smoke fixture"},
    )

    report = train(root, output, mode="smoke", seed=123, device="cpu")

    checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["scope"] == BASELINE_SCOPE
    assert checkpoint["epoch_completed"] == 1
    assert checkpoint["terminal_state"] == "horizon_complete"
    runtime_identity = checkpoint["config"]["runtime_identity"]
    assert runtime_identity["requested_device"] == "cpu"
    assert runtime_identity["scipy"]
    assert "cublas_workspace_config" in runtime_identity
    assert runtime_identity["module_sha256"] == cardiacuda_segmentation._file_sha256(
        Path(cardiacuda_segmentation.__file__)
    )
    assert {
        "model_state_dict",
        "ema_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "rng_state",
        "loader_generator_states",
        "transform_log_state",
        "best_artifact",
    } <= set(checkpoint)
    assert report["evaluation_unit"] == "patient"
    assert report["trained_views"] == ["A4C"]
    log_lines = (output / "augmentation_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert log_lines
    envelope = json.loads(log_lines[0])
    validate_transform_record(envelope["sample"])
    assert envelope["sample"]["augmentations"][0]["applied"] is False

    original_log = (output / "augmentation_log.jsonl").read_bytes()
    (output / "augmentation_log.jsonl").write_bytes(b"")
    with pytest.raises(ValueError, match="shorter than its checkpoint anchor"):
        train(
            root,
            output,
            mode="smoke",
            seed=123,
            device="cpu",
            resume_checkpoint=output / "last.pt",
        )
    (output / "augmentation_log.jsonl").write_bytes(original_log + b"aborted-partial-line")

    resumed = train(
        root,
        output,
        mode="smoke",
        seed=123,
        device="cpu",
        resume_checkpoint=output / "last.pt",
    )
    assert resumed["epochs_completed"] == 1
    assert resumed["resumed_from"] == str(output / "last.pt")
    assert resumed["terminal_state"] == "horizon_complete"
    assert (output / "augmentation_log.jsonl").read_bytes() == original_log
    assert Path(resumed["recovered_aborted_augmentation_tail"]).read_bytes() == b"aborted-partial-line"

    early_stop_payload = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    early_stop_payload["terminal_state"] = "early_stopped"
    early_stop_payload["bad_epochs"] = early_stop_payload["config"]["settings"]["patience"]
    early_stop_checkpoint = output / "early-stop.pt"
    torch.save(early_stop_payload, early_stop_checkpoint)
    before_early_resume = (output / "augmentation_log.jsonl").read_bytes()
    early_resumed = train(
        root,
        output,
        mode="smoke",
        seed=123,
        device="cpu",
        resume_checkpoint=early_stop_checkpoint,
    )
    assert early_resumed["terminal_state"] == "early_stopped"
    assert early_resumed["epochs_completed"] == 1
    assert (output / "augmentation_log.jsonl").read_bytes() == before_early_resume

    with pytest.raises(ValueError, match="configuration fingerprint changed"):
        train(
            root,
            output,
            mode="smoke",
            seed=123,
            device="cpu:0",
            resume_checkpoint=output / "last.pt",
        )

    with pytest.raises(ValueError, match="configuration fingerprint changed"):
        train(
            root,
            output,
            mode="smoke",
            seed=124,
            device="cpu",
            resume_checkpoint=output / "last.pt",
        )

    nonfinite = copy.deepcopy(checkpoint)
    tensor_name = next(
        name
        for name, tensor in nonfinite["model_state_dict"].items()
        if tensor.is_floating_point() and tensor.numel()
    )
    corrupted_tensor = nonfinite["model_state_dict"][tensor_name].clone()
    corrupted_tensor.reshape(-1)[0] = float("nan")
    nonfinite["model_state_dict"][tensor_name] = corrupted_tensor
    with pytest.raises(FloatingPointError, match="Non-finite tensor"):
        cardiacuda_segmentation._validate_resume_checkpoint(
            nonfinite,
            manifest_fingerprint=checkpoint["manifest_fingerprint"],
            config_fingerprint=checkpoint["config_fingerprint"],
        )
