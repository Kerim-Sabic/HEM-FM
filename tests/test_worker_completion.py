from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from hemfm.worker_completion import (
    CompletionError,
    main,
    record_seed_completion,
    snapshot_provenance,
    verify_matched_audit,
    verify_provenance,
    verify_seed_completion,
)


SEEDS = [1103, 2207, 3301]
AUDIT_CHECKS = {
    "development_rows_only": True,
    "patient_disjoint": True,
    "exact_immutable_validation_cohort": True,
    "minimum_validation_samples": True,
    "three_seed_full_ensemble": True,
    "finite_positive_uncertainty": True,
    "locked_test_not_accessed": True,
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _git(checkout: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _fixture(tmp_path: Path, *, kind: str = "transfer") -> dict[str, Path]:
    repository = tmp_path / "repository"
    source = repository / "src" / "hemfm"
    source.mkdir(parents=True)
    (source / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    functional_checkpoint = tmp_path / "assets" / "functional.pt"
    functional_checkpoint.parent.mkdir()
    functional_checkpoint.write_bytes(b"verified functional checkpoint")
    functional_sha256 = hashlib.sha256(functional_checkpoint.read_bytes()).hexdigest()
    echojepa_source = tmp_path / "vendor" / "EchoJEPA"
    echojepa_source.mkdir(parents=True)
    (echojepa_source / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (echojepa_source / "package_marker.py").write_bytes(b"")
    _git(echojepa_source, "init", "-q")
    _git(echojepa_source, "add", ".")
    _git(
        echojepa_source,
        "-c",
        "user.name=HEMFM Test",
        "-c",
        "user.email=hemfm@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "fixture",
    )
    echojepa_revision = _git(echojepa_source, "rev-parse", "HEAD")
    evidence_root = tmp_path / "evidence"
    _write_json(
        evidence_root / "G4" / "echonet_dynamic_stage.json",
        {
            "schema_version": 1,
            "passed": True,
            "mode": "full",
            "training_ready": True,
            "source": "https://echonet.github.io/dynamic/",
            "archive_sha256": "c" * 64,
            "development_records": 2,
            "split_counts": {"TRAIN": 1, "VAL": 1},
            "frames": 16,
            "resolution": 224,
            "trace_annotated_records": 4,
            "trace_frames_cached": 8,
            "trace_cache_contents": "exact ED/ES frames and masks",
            "official_test_records_reserved": 2,
            "official_test_videos_accessed": False,
            "locked_mimic_test_accessed": False,
            "errors": [],
            "track": "R",
        },
    )
    private_root = tmp_path / "private"
    staging_root = tmp_path / "staging"
    development_manifest = (
        staging_root
        / "datasets"
        / "research"
        / "echonet-dynamic"
        / "development_manifest_private.csv"
    )
    development_manifest.parent.mkdir(parents=True)
    cache_root = development_manifest.parent / "cache_16x224"
    train_cache = cache_root / "train" / "a.npz"
    validation_cache = cache_root / "val" / "b.npz"
    train_cache.parent.mkdir(parents=True)
    validation_cache.parent.mkdir(parents=True)
    train_cache.write_bytes(b"train cache")
    validation_cache.write_bytes(b"validation cache")
    with development_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["FileName", "Split", "cache_path"])
        writer.writerow(["a", "TRAIN", str(train_cache)])
        writer.writerow(["b", "VAL", str(validation_cache)])
    config = tmp_path / "protocol.yaml"
    _write_json(
        config,
        {
            "protocol": "HEM-FM-v4.0",
            "paths": {
                "evidence_root": str(evidence_root),
                "private_root": str(private_root),
                "local_staging_root": str(staging_root),
                "functional_checkpoint": str(functional_checkpoint),
                "echojepa_source": str(echojepa_source),
            },
            "assets": {
                "echojepa_source_revision": echojepa_revision,
                "functional_checkpoint_sha256": functional_sha256,
            },
            "training": {"seeds": SEEDS},
            "targets": {"EF": {}},
        },
    )
    artifact_root = tmp_path / kind
    seed_root = artifact_root / "seed_1103_full"
    seed_root.mkdir(parents=True)
    report = {
        "passed": True,
        "seed": 1103,
        "epochs_completed": 3,
        "best_epoch": 3,
        "history": [
            {"epoch": 1, "stage": "frozen"},
            {"epoch": 2, "stage": "peft"},
            {"epoch": 3, "stage": "selective_unfreeze"},
        ],
        "official_test_videos_accessed": False,
        "locked_test_accessed": False,
    }
    _write_json(seed_root / "report.json", report)
    (seed_root / "checkpoint_best.pt").write_bytes(b"best checkpoint")
    (seed_root / "checkpoint_last.pt").write_bytes(b"last checkpoint")
    prediction_name = (
        "validation_predictions_private.csv"
        if kind == "transfer"
        else "validation_masks_private.npz"
    )
    (seed_root / prediction_name).write_bytes(b"private prediction")
    launch_provenance = tmp_path / "launch_provenance.json"
    snapshot_provenance(
        config_path=config,
        repository=repository,
        output=launch_provenance,
    )
    return {
        "repository": repository,
        "source": source / "model.py",
        "functional_checkpoint": functional_checkpoint,
        "echojepa_source": echojepa_source,
        "train_cache": train_cache,
        "validation_cache": validation_cache,
        "development_manifest": development_manifest,
        "evidence_root": evidence_root,
        "private_root": private_root,
        "config": config,
        "launch_provenance": launch_provenance,
        "artifact_root": artifact_root,
        "seed_root": seed_root,
    }


def _record(paths: dict[str, Path], *, kind: str = "transfer") -> dict[str, object]:
    return record_seed_completion(
        kind=kind,
        artifact_root=paths["artifact_root"],
        seed=1103,
        mode="full",
        requested_epochs=3,
        config_path=paths["config"],
        repository=paths["repository"],
        not_before=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        launch_provenance=paths["launch_provenance"],
    )


def _verify(paths: dict[str, Path], *, kind: str = "transfer") -> dict[str, object]:
    return verify_seed_completion(
        kind=kind,
        artifact_root=paths["artifact_root"],
        seed=1103,
        mode="full",
        requested_epochs=3,
        config_path=paths["config"],
        repository=paths["repository"],
    )


@pytest.mark.parametrize("kind", ["transfer", "trace"])
def test_record_and_verify_seed_manifest(kind: str, tmp_path: Path) -> None:
    paths = _fixture(tmp_path, kind=kind)
    recorded = _record(paths, kind=kind)
    verified = _verify(paths, kind=kind)

    manifest_path = paths["seed_root"] / "completion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert recorded["ok"] is True
    assert verified["ok"] is True
    assert set(manifest["artifacts"]) == {"report", "best", "last", "prediction"}
    assert all(row["size_bytes"] > 0 for row in manifest["artifacts"].values())
    assert all(len(row["sha256"]) == 64 for row in manifest["artifacts"].values())
    assert manifest["report_summary"]["stages"] == [
        "frozen",
        "peft",
        "selective_unfreeze",
    ]
    assert manifest["provenance"]["protocol"] == "HEM-FM-v4.0"
    assert len(manifest["provenance"]["source_tree"]["sha256"]) == 64
    assert (
        manifest["provenance"]["functional_checkpoint"]["sha256"]
        == hashlib.sha256(paths["functional_checkpoint"].read_bytes()).hexdigest()
    )
    assert (
        manifest["provenance"]["echojepa_checkout"]["head_revision"]
        == _git(paths["echojepa_source"], "rev-parse", "HEAD")
    )
    assert (
        manifest["provenance"]["echonet_stage_evidence"]["referenced_cache"][
            "file_count"
        ]
        == 2
    )
    assert (
        len(manifest["provenance"]["echonet_stage_evidence"]["observation"]["sha256"])
        == 64
    )
    assert len(manifest["launch_provenance_file"]["sha256"]) == 64


def test_provenance_snapshot_ignores_only_volatile_restage_fields(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    stage_path = paths["evidence_root"] / "G4" / "echonet_dynamic_stage.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    stage["created_utc"] = datetime.now(timezone.utc).isoformat()
    stage["created_this_run"] = 0
    stage["wall_seconds"] = 3.5
    _write_json(stage_path, stage)

    result = verify_provenance(
        config_path=paths["config"],
        repository=paths["repository"],
        input_path=paths["launch_provenance"],
    )
    assert result["ok"] is True


def test_provenance_snapshot_records_but_does_not_gate_on_git_head(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    snapshot = json.loads(paths["launch_provenance"].read_text(encoding="utf-8"))
    snapshot["provenance"]["git_head"] = "d" * 40
    _write_json(paths["launch_provenance"], snapshot)

    result = verify_provenance(
        config_path=paths["config"],
        repository=paths["repository"],
        input_path=paths["launch_provenance"],
    )
    assert result["ok"] is True


def test_provenance_ignores_editable_install_metadata(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    metadata = paths["repository"] / "src" / "hemfm_v4.egg-info"
    metadata.mkdir()
    (metadata / "PKG-INFO").write_text("generated metadata\n", encoding="utf-8")

    result = verify_provenance(
        config_path=paths["config"],
        repository=paths["repository"],
        input_path=paths["launch_provenance"],
    )
    assert result["ok"] is True


def test_provenance_binds_actual_functional_checkpoint(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["functional_checkpoint"].write_bytes(b"changed checkpoint")

    with pytest.raises(CompletionError, match="functional checkpoint SHA-256"):
        verify_provenance(
            config_path=paths["config"],
            repository=paths["repository"],
            input_path=paths["launch_provenance"],
        )


def test_provenance_binds_echojepa_worktree_content(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    (paths["echojepa_source"] / "model.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )

    with pytest.raises(CompletionError, match="launch provenance does not match"):
        verify_provenance(
            config_path=paths["config"],
            repository=paths["repository"],
            input_path=paths["launch_provenance"],
        )


def test_provenance_records_stable_dirty_echojepa_checkout(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    (paths["echojepa_source"] / "model.py").write_text(
        "VALUE = 3\n", encoding="utf-8"
    )
    snapshot = tmp_path / "dirty_launch_provenance.json"

    snapshot_provenance(
        config_path=paths["config"],
        repository=paths["repository"],
        output=snapshot,
    )
    result = verify_provenance(
        config_path=paths["config"],
        repository=paths["repository"],
        input_path=snapshot,
    )
    payload = json.loads(snapshot.read_text(encoding="utf-8"))

    assert result["ok"] is True
    assert payload["provenance"]["echojepa_checkout"]["dirty"] is True


def test_provenance_rejects_echojepa_revision_change(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    (paths["echojepa_source"] / "new.py").write_text("VALUE = 4\n", encoding="utf-8")
    _git(paths["echojepa_source"], "add", ".")
    _git(
        paths["echojepa_source"],
        "-c",
        "user.name=HEMFM Test",
        "-c",
        "user.email=hemfm@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "changed",
    )

    with pytest.raises(CompletionError, match="EchoJEPA HEAD does not match"):
        verify_provenance(
            config_path=paths["config"],
            repository=paths["repository"],
            input_path=paths["launch_provenance"],
        )


def test_provenance_binds_every_referenced_cache_artifact(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["train_cache"].write_bytes(b"changed train cache")

    with pytest.raises(CompletionError, match="launch provenance does not match"):
        verify_provenance(
            config_path=paths["config"],
            repository=paths["repository"],
            input_path=paths["launch_provenance"],
        )


def test_snapshot_rejects_duplicate_manifest_references(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    with paths["development_manifest"].open(
        "a", encoding="utf-8", newline=""
    ) as handle:
        csv.writer(handle).writerow(["a", "TRAIN", str(paths["train_cache"])])

    with pytest.raises(CompletionError, match="duplicate FileName"):
        snapshot_provenance(
            config_path=paths["config"],
            repository=paths["repository"],
            output=tmp_path / "duplicate.json",
        )


def test_snapshot_rejects_cache_path_escape(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    escaped = tmp_path / "outside" / "a.npz"
    escaped.parent.mkdir()
    escaped.write_bytes(b"outside")
    rows = list(csv.reader(paths["development_manifest"].read_text().splitlines()))
    rows[1][2] = str(escaped)
    with paths["development_manifest"].open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        csv.writer(handle).writerows(rows)

    with pytest.raises(CompletionError, match="escapes the staged cache root"):
        snapshot_provenance(
            config_path=paths["config"],
            repository=paths["repository"],
            output=tmp_path / "escaped.json",
        )


def test_snapshot_rejects_symlinked_cache_artifact(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    target = paths["train_cache"].with_name("target.npz")
    target.write_bytes(paths["train_cache"].read_bytes())
    paths["train_cache"].unlink()
    try:
        paths["train_cache"].symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    with pytest.raises(CompletionError, match="symlink or junction"):
        snapshot_provenance(
            config_path=paths["config"],
            repository=paths["repository"],
            output=tmp_path / "symlink.json",
        )


@pytest.mark.parametrize("changed", ["config", "source"])
def test_record_rejects_changes_after_launch_snapshot(
    changed: str, tmp_path: Path
) -> None:
    paths = _fixture(tmp_path)
    if changed == "config":
        config = json.loads(paths["config"].read_text(encoding="utf-8"))
        config["post_snapshot_change"] = True
        _write_json(paths["config"], config)
    else:
        paths["source"].write_text("VALUE = 99\n", encoding="utf-8")
    with pytest.raises(CompletionError, match="launch provenance does not match"):
        _record(paths)


def test_cli_prints_compact_success_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _fixture(tmp_path)
    result = main(
        [
            "record-seed",
            "--kind",
            "transfer",
            "--artifact-root",
            str(paths["artifact_root"]),
            "--seed",
            "1103",
            "--mode",
            "full",
            "--requested-epochs",
            "3",
            "--config",
            str(paths["config"]),
            "--repository",
            str(paths["repository"]),
            "--not-before",
            (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            "--launch-provenance",
            str(paths["launch_provenance"]),
        ]
    )
    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert json.loads(captured.out)["action"] == "record-seed"
    assert "\n" not in captured.out.rstrip("\n")


def test_verify_rejects_missing_manifest(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    with pytest.raises(CompletionError, match="missing completion manifest"):
        _verify(paths)


def test_verify_rejects_artifact_tamper(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _record(paths)
    with (paths["seed_root"] / "checkpoint_best.pt").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(CompletionError, match="artifact hash or size mismatch"):
        _verify(paths)


@pytest.mark.parametrize("changed", ["config", "source"])
def test_verify_rejects_changed_provenance(changed: str, tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _record(paths)
    if changed == "config":
        config = json.loads(paths["config"].read_text(encoding="utf-8"))
        config["unrelated_change"] = True
        _write_json(paths["config"], config)
    else:
        paths["source"].write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(CompletionError, match="provenance mismatch"):
        _verify(paths)


def test_record_rejects_stale_report(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    stale = datetime.now(timezone.utc) - timedelta(hours=2)
    report_path = paths["seed_root"] / "report.json"
    os.utime(report_path, (stale.timestamp(), stale.timestamp()))
    with pytest.raises(CompletionError, match="stale seed report"):
        record_seed_completion(
            kind="transfer",
            artifact_root=paths["artifact_root"],
            seed=1103,
            mode="full",
            requested_epochs=3,
            config_path=paths["config"],
            repository=paths["repository"],
            not_before=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            launch_provenance=paths["launch_provenance"],
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("passed", "true"),
        ("passed", 1),
        ("official_test_videos_accessed", "false"),
        ("official_test_videos_accessed", 0),
        ("locked_test_accessed", "false"),
        ("locked_test_accessed", 0),
    ],
)
def test_seed_report_rejects_string_and_numeric_booleans(
    field: str, invalid: object, tmp_path: Path
) -> None:
    paths = _fixture(tmp_path)
    report_path = paths["seed_root"] / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report[field] = invalid
    _write_json(report_path, report)
    with pytest.raises(CompletionError, match=field):
        _record(paths)


def _write_matched_csv(path: Path, *, rows: int = 2, omit: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "sample_id",
        "subject_id",
        "study_id",
        "target",
        "value",
        *[f"prediction_seed_{seed}" for seed in SEEDS],
        *[f"aleatoric_sigma_seed_{seed}" for seed in SEEDS],
        "prediction",
        "aleatoric_sigma",
        "ensemble_dispersion",
        "total_uncertainty",
        "absolute_error",
        "large_residual",
        "risk_rank_percentile",
    ]
    if omit:
        columns.remove(omit)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index in range(rows):
            row: dict[str, object] = {
                "sample_id": f"sample-{index}",
                "subject_id": f"subject-{index}",
                "study_id": f"study-{index}",
                "target": "EF",
                "value": 50 + index,
                "prediction": 49 + index,
                "aleatoric_sigma": 2.0,
                "ensemble_dispersion": 0.5,
                "total_uncertainty": 2.1,
                "absolute_error": 1.0,
                "large_residual": False,
                "risk_rank_percentile": 0.5,
            }
            for seed in SEEDS:
                row[f"prediction_seed_{seed}"] = 49 + index
                row[f"aleatoric_sigma_seed_{seed}"] = 2.0
            if omit:
                row.pop(omit, None)
            writer.writerow(row)


def _matched_fixture(tmp_path: Path) -> tuple[dict[str, Path], Path, Path]:
    paths = _fixture(tmp_path)
    private_csv = paths["private_root"] / "matched.csv"
    _write_matched_csv(private_csv)
    evidence_path = (
        paths["evidence_root"] / "G6" / "staged_final_ef_full_matched_uncertainty.json"
    )
    _write_json(
        evidence_path,
        {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "passed": True,
            "target": "EF",
            "mode": "full",
            "validation_samples": 2,
            "seeds": [{"seed": seed, "locked_test_accessed": False} for seed in SEEDS],
            "audit_checks": AUDIT_CHECKS,
            "private_predictions": str(private_csv),
            "locked_test_accessed": False,
        },
    )
    return paths, evidence_path, private_csv


def _verify_matched(
    paths: dict[str, Path], *, not_before: datetime | None = None
) -> dict[str, object]:
    return verify_matched_audit(
        evidence_root=paths["evidence_root"],
        target="EF",
        mode="full",
        config_path=paths["config"],
        repository=paths["repository"],
        not_before=(
            not_before or datetime.now(timezone.utc) - timedelta(minutes=1)
        ).isoformat(),
    )


def test_verify_matched_audit_accepts_fresh_complete_evidence(tmp_path: Path) -> None:
    paths, _, _ = _matched_fixture(tmp_path)
    result = _verify_matched(paths)
    assert result["ok"] is True
    assert result["seeds"] == SEEDS
    assert result["rows"] == 2
    assert len(result["private_csv_sha256"]) == 64


def test_verify_matched_audit_rejects_stale_evidence(tmp_path: Path) -> None:
    paths, evidence_path, _ = _matched_fixture(tmp_path)
    stale = datetime.now(timezone.utc) - timedelta(hours=2)
    os.utime(evidence_path, (stale.timestamp(), stale.timestamp()))
    with pytest.raises(CompletionError, match="stale matched audit evidence"):
        _verify_matched(paths)


@pytest.mark.parametrize("invalid", ["true", 1])
def test_verify_matched_audit_rejects_non_boolean_flags(
    invalid: object, tmp_path: Path
) -> None:
    paths, evidence_path, _ = _matched_fixture(tmp_path)
    report = json.loads(evidence_path.read_text(encoding="utf-8"))
    report["audit_checks"]["patient_disjoint"] = invalid
    _write_json(evidence_path, report)
    with pytest.raises(CompletionError, match="patient_disjoint"):
        _verify_matched(paths)


def test_verify_matched_audit_rejects_malformed_csv(tmp_path: Path) -> None:
    paths, _, private_csv = _matched_fixture(tmp_path)
    _write_matched_csv(private_csv, omit="total_uncertainty")
    with pytest.raises(CompletionError, match="missing columns: total_uncertainty"):
        _verify_matched(paths)


def test_cli_failure_is_concise_and_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _fixture(tmp_path)
    result = main(
        [
            "verify-seed",
            "--kind",
            "transfer",
            "--artifact-root",
            str(paths["artifact_root"]),
            "--seed",
            "1103",
            "--mode",
            "full",
            "--requested-epochs",
            "3",
            "--config",
            str(paths["config"]),
            "--repository",
            str(paths["repository"]),
        ]
    )
    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "missing completion manifest" in captured.err
