from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ensure_run_directories, load_config
from .calibration_validation import validate_real_calibration
from .camus_dense import run_camus_dense_pretraining, write_camus_acquisition_manifest
from .augmentations import verify_physical_augmentation_parity
from .architecture_pilot import run_architecture_ladder
from .cohorts import build_endpoint_cohorts
from .gates import assert_through, evidence_state
from .full_extraction import build_full_specialist_feature_cache
from .frozen_specialists import run_frozen_specialists
from .fusion_specialists import run_multibackbone_fusion
from .hashing import sha256_file
from .ev9v_view import run_ev9v_view_training
from .echonet_dynamic import (
    audit_echonet_dynamic_archive,
    finalize_echonet_dynamic_transfer,
    run_echonet_dynamic_transfer,
    run_echonet_dynamic_transfer_seed,
    stage_echonet_dynamic,
)
from .echonet_trace import (
    finalize_echonet_trace,
    run_echonet_trace_seed,
    run_echonet_trace_training,
)
from .external_ood import run_external_ood_audit
from .external_lock import run_external_lock
from .inventory import run_dicom_inventory, run_label_inventory
from .integration_audit import run_integration_audit
from .mimic_lv_extension import stage_mimic_lv_extension
from .mimic_lv_training import run_mimic_lv_training
from .optimizer_peft_pilot import run_optimizer_peft_pilot, write_decision_ledger
from .panecho_challenger import run_panecho_mimic_lv_audit
from .provenance_audit import write_licence_provenance
from .research_datasets import audit_research_datasets
from .runtime import dinov3_checkpoint_smoke, echojepa_checkpoint_smoke, gpu_smoke, vjepa21_checkpoint_smoke, vjepa21_vitb_checkpoint_smoke, write_report
from .shortcut_controls import run_shortcut_controls
from .specialist_audit import run_specialist_holdout_and_failure_audit
from .splits import (
    audit_development_cohort_alignment,
    audit_hashed_split_manifest,
    build_hashed_split_manifest,
    write_audit,
)
from .staged_final_scalar import run_staged_scalar_target, stage_staged_scalar_cache
from .staged_scalar_audit import run_staged_scalar_uncertainty_audit
from .ted_temporal import evaluate_ted_native_inference, run_ted_temporal_training
from .unity_landmarks import run_unity_landmark_training
from .train_cached import CachedPilotConfig, run_cached_functional_pilot


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2))


def _assert_mimic_development_split_ready(config: dict) -> dict:
    """Bind MIMIC training to matching v2 inventory/split/calibration evidence."""
    cohort_report = audit_development_cohort_alignment(
        Path(config["paths"]["private_root"]) / "endpoint_cohorts_development_private.csv",
        config["paths"]["record_list"],
        config["paths"]["private_root"],
        seed=int(config["splits"]["seed"]),
        train=float(config["splits"]["train"]),
        validation=float(config["splits"]["validation"]),
    )
    evidence_root = Path(config["paths"]["evidence_root"]) / "G1"

    def read_evidence(name: str) -> tuple[Path, dict]:
        path = evidence_root / name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"MIMIC training requires readable G1 evidence: {name}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"MIMIC training requires object-valued G1 evidence: {name}")
        return path, payload

    inventory_path, inventory = read_evidence("dicom_inventory_summary.json")
    _, split_evidence = read_evidence("split_audit.json")
    _, calibration = read_evidence("calibration_tests.json")
    manifest_sha256 = cohort_report["manifest_sha256"]
    record_list_sha256 = cohort_report["manifest_record_list_sha256"]
    inventory_summary_sha256 = sha256_file(inventory_path)
    inventory_enriched_fields = set(inventory.get("enriched_fields", []) or [])
    required_enriched_fields = {
        "image_type",
        "frames",
        "region_type",
        "vendor",
        "software",
        "measurement_screen_layout",
        "measurement_screen_layout_provenance",
    }

    checks = {
        "inventory_schema_v2": inventory.get("schema_version") == 2,
        "inventory_passed": inventory.get("passed") is True,
        "inventory_complete": (
            inventory.get("complete") is True
            and inventory.get("scan_target_complete") is True
            and inventory.get("status") == "complete"
        ),
        "inventory_metadata_only": (
            inventory.get("metadata_only") is True
            and inventory.get("pixel_data_read") is False
        ),
        "inventory_locked_test_pixels_sealed": (
            inventory.get("locked_test_pixels_accessed") is False
        ),
        "inventory_enrichment_complete": required_enriched_fields.issubset(
            inventory_enriched_fields
        ),
        "inventory_layout_is_candidate_prior": (
            inventory.get("measurement_screen_layout_provenance")
            == "metadata_only_candidate_prior_not_annotation_truth"
            and inventory.get("measurement_screen_layout_is_annotation_truth") is False
        ),
        "split_preceded_decoding": inventory.get("split_manifest_preceded_decoding") is True,
        "inventory_manifest_matches": inventory.get("split_manifest_sha256") == manifest_sha256,
        "inventory_record_list_matches": inventory.get("record_list_sha256") == record_list_sha256,
        "inventory_execution_checks_passed": bool(inventory.get("execution_checks"))
        and all(inventory.get("execution_checks", {}).values()),
        "split_evidence_schema_v2": split_evidence.get("schema_version") == 2,
        "split_evidence_passed": split_evidence.get("passed") is True,
        "split_evidence_manifest_matches": split_evidence.get("manifest_sha256") == manifest_sha256,
        "split_evidence_record_list_matches": (
            split_evidence.get("record_list_sha256") == record_list_sha256
        ),
        "calibration_schema_v2": calibration.get("schema_version") == 2,
        "calibration_passed": calibration.get("passed") is True,
        "calibration_metadata_only": (
            calibration.get("metadata_only") is True
            and calibration.get("pixel_data_read") is False
        ),
        "calibration_locked_test_pixels_sealed": (
            calibration.get("locked_test_pixels_accessed") is False
        ),
        "calibration_summary_matches": (
            calibration.get("inventory_summary_sha256") == inventory_summary_sha256
        ),
        "calibration_manifest_matches": calibration.get("split_manifest_sha256") == manifest_sha256,
        "calibration_record_list_matches": (
            calibration.get("record_list_sha256") == record_list_sha256
        ),
    }

    artifact_fields = {
        "inventory_csv": ("legacy_csv", "legacy_csv_sha256"),
        "instance_parquet": ("instance_parquet", "instance_parquet_sha256"),
        "region_parquet": ("region_parquet", "region_parquet_sha256"),
    }
    artifact_hashes: dict[str, str] = {}
    for name, (path_field, hash_field) in artifact_fields.items():
        artifact_path = Path(str(inventory.get(path_field, "")))
        exists = artifact_path.is_file()
        actual_hash = sha256_file(artifact_path) if exists else ""
        declared_hash = str(inventory.get(hash_field, ""))
        checks[f"{name}_exists"] = exists
        checks[f"{name}_inventory_hash_matches"] = bool(
            len(declared_hash) == 64 and actual_hash == declared_hash
        )
        checks[f"{name}_calibration_hash_matches"] = (
            calibration.get(
                "inventory_csv_sha256" if name == "inventory_csv" else hash_field
            )
            == actual_hash
        )
        artifact_hashes[name] = actual_hash

    binding_checks = calibration.get("inventory_binding", {}).get("checks", {})
    checks["calibration_binding_checks_passed"] = bool(binding_checks) and all(
        binding_checks.values()
    )
    if failed := [name for name, passed in checks.items() if not passed]:
        raise RuntimeError(
            "MIMIC training refused stale or mismatched data-foundation evidence: "
            + ", ".join(failed)
        )
    gate_state = assert_through(config, "G1")
    return {
        "passed": True,
        "cohort": cohort_report,
        "checks": checks,
        "artifact_hashes": artifact_hashes,
        "g1_execution_sequentially_unlocked": gate_state["gates"]["G1"][
            "execution_sequentially_unlocked"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="hemfm", description="HEM-FM v4.0 fail-closed local runner")
    parser.add_argument("--config", default=None)
    commands = parser.add_subparsers(dest="command", required=True)

    runtime = commands.add_parser("runtime")
    runtime.add_argument("action", choices=["smoke", "challengers"])

    inventory = commands.add_parser("inventory")
    inventory.add_argument("kind", choices=["labels", "dicom"])
    inventory.add_argument("--workers", type=int, default=16)
    inventory.add_argument("--limit", type=int, default=None)
    inventory.add_argument("--resume", action="store_true")

    splits = commands.add_parser("splits")
    splits.add_argument("action", choices=["build", "audit"])

    calibration = commands.add_parser("calibration")
    calibration.add_argument("action", choices=["validate"])
    calibration.add_argument("--max-files", type=int, default=512)

    cohort = commands.add_parser("cohort")
    cohort.add_argument("action", choices=["build"])
    cohort.add_argument("--confidence-min", type=float, default=0.75)

    augmentation = commands.add_parser("augmentation")
    augmentation.add_argument("action", choices=["validate"])
    augmentation.add_argument("--trials", type=int, default=2000)

    shortcuts = commands.add_parser("shortcuts")
    shortcuts.add_argument("action", choices=["validate"])
    shortcuts.add_argument("--train-per-target", type=int, default=60)
    shortcuts.add_argument("--validation-per-target", type=int, default=30)
    shortcuts.add_argument("--workers", type=int, default=8)
    shortcuts.add_argument("--device", type=int, default=1)

    provenance = commands.add_parser("provenance")
    provenance.add_argument("action", choices=["validate"])

    architecture = commands.add_parser("architecture")
    architecture.add_argument("action", choices=["pilot"])
    architecture.add_argument("--train-per-target", type=int, default=32)
    architecture.add_argument("--validation-per-target", type=int, default=16)

    tuning = commands.add_parser("tuning")
    tuning.add_argument("action", choices=["pilot"])
    tuning.add_argument("--train-count", type=int, default=24)
    tuning.add_argument("--validation-count", type=int, default=12)
    tuning.add_argument("--epochs", type=int, default=3)
    tuning.add_argument("--device", type=int, default=0)

    specialists = commands.add_parser("specialists")
    specialists.add_argument("action", choices=["extract", "train", "fusion"])

    dense_lv = commands.add_parser("dense-lv")
    dense_lv.add_argument(
        "action", choices=["manifest", "smoke", "train", "a2c-smoke", "a2c-train"]
    )
    dense_lv.add_argument("--device", type=int, default=0)
    dense_lv.add_argument("--epochs", type=int, default=None)
    dense_lv.add_argument("--frozen-epochs", type=int, default=None)
    dense_lv.add_argument("--max-train", type=int, default=None)
    dense_lv.add_argument("--max-validation", type=int, default=None)
    dense_lv.add_argument(
        "--backbone", choices=["vjepa21_vitb", "dinov3_vitb"], default="vjepa21_vitb"
    )

    research_data = commands.add_parser("research-data")
    research_data.add_argument("action", choices=["audit"])

    echonet_dynamic = commands.add_parser("echonet-dynamic")
    echonet_dynamic.add_argument(
        "action", choices=["audit", "stage", "smoke", "train", "seed", "finalize"]
    )
    echonet_dynamic.add_argument("--archive", default=None)
    echonet_dynamic.add_argument("--workers", type=int, default=8)
    echonet_dynamic.add_argument("--frames", type=int, default=16)
    echonet_dynamic.add_argument("--resolution", type=int, default=224)
    echonet_dynamic.add_argument("--limit", type=int, default=None)
    echonet_dynamic.add_argument("--hash", action="store_true")
    echonet_dynamic.add_argument("--device", type=int, default=0)
    echonet_dynamic.add_argument("--epochs", type=int, default=None)
    echonet_dynamic.add_argument("--max-train", type=int, default=None)
    echonet_dynamic.add_argument("--max-validation", type=int, default=None)
    echonet_dynamic.add_argument("--seed", type=int, default=None)

    echonet_trace = commands.add_parser("echonet-trace")
    echonet_trace.add_argument("action", choices=["smoke", "train", "seed", "finalize"])
    echonet_trace.add_argument("--device", type=int, default=0)
    echonet_trace.add_argument("--epochs", type=int, default=None)
    echonet_trace.add_argument("--max-train", type=int, default=None)
    echonet_trace.add_argument("--max-validation", type=int, default=None)
    echonet_trace.add_argument("--seed", type=int, default=None)

    temporal = commands.add_parser("temporal")
    temporal.add_argument("action", choices=["smoke", "train", "native-evaluate"])
    temporal.add_argument("--device", type=int, default=0)
    temporal.add_argument("--epochs", type=int, default=None)
    temporal.add_argument("--frozen-epochs", type=int, default=None)
    temporal.add_argument("--max-train", type=int, default=None)
    temporal.add_argument("--max-validation", type=int, default=None)
    temporal.add_argument("--seed", type=int, default=None)
    temporal.add_argument("--checkpoint", type=Path, default=None)
    temporal.add_argument("--overlap-frames", type=int, default=8)
    temporal.add_argument(
        "--backbone", choices=["vjepa21_vitb", "dinov3_vitb"], default="vjepa21_vitb"
    )

    mimic_lv = commands.add_parser("mimic-lv")
    mimic_lv.add_argument("action", choices=["stage", "smoke", "train"])
    mimic_lv.add_argument("--source-root", default=None)
    mimic_lv.add_argument("--metadata", default=None)
    mimic_lv.add_argument("--device", type=int, default=0)
    mimic_lv.add_argument("--epochs", type=int, default=None)
    mimic_lv.add_argument("--frozen-epochs", type=int, default=None)
    mimic_lv.add_argument("--max-train", type=int, default=None)
    mimic_lv.add_argument("--max-validation", type=int, default=None)
    mimic_lv.add_argument("--ema-decay", type=float, default=0.99)
    mimic_lv.add_argument("--run-tag", default=None)
    mimic_lv.add_argument(
        "--backbone", choices=["vjepa21_vitb", "dinov3_vitb"], default="vjepa21_vitb"
    )

    unity_landmarks = commands.add_parser("unity-landmarks")
    unity_landmarks.add_argument("action", choices=["smoke", "train"])
    unity_landmarks.add_argument("--device", type=int, default=1)
    unity_landmarks.add_argument("--epochs", type=int, default=None)
    unity_landmarks.add_argument("--frozen-epochs", type=int, default=None)
    unity_landmarks.add_argument("--max-train", type=int, default=None)
    unity_landmarks.add_argument("--max-validation", type=int, default=None)

    ev9v_view = commands.add_parser("ev9v-view")
    ev9v_view.add_argument("action", choices=["smoke", "train"])
    ev9v_view.add_argument("--device", type=int, default=0)
    ev9v_view.add_argument("--epochs", type=int, default=None)
    ev9v_view.add_argument("--frozen-epochs", type=int, default=None)
    ev9v_view.add_argument("--max-train", type=int, default=None)
    ev9v_view.add_argument("--max-validation", type=int, default=None)

    external_ood = commands.add_parser("external-ood")
    external_ood.add_argument("action", choices=["smoke", "audit"])
    external_ood.add_argument("--device", type=int, default=0)
    external_ood.add_argument("--max-per-dataset", type=int, default=None)

    panecho = commands.add_parser("panecho")
    panecho.add_argument("action", choices=["smoke", "audit"])
    panecho.add_argument("--device", type=int, default=0)
    panecho.add_argument("--max-train", type=int, default=None)
    panecho.add_argument("--max-validation", type=int, default=None)

    staged_final = commands.add_parser("staged-final")
    staged_final.add_argument(
        "action", choices=["cache", "smoke", "train", "audit"]
    )
    staged_final.add_argument(
        "--target",
        choices=[
            "EF", "LVEDV", "LVESV", "LVOT_DIAMETER",
            "RV_BASAL_DIAMETER", "AV_PEAK_VELOCITY",
        ],
        default="EF",
    )
    staged_final.add_argument("--device", type=int, default=0)
    staged_final.add_argument("--epochs", type=int, default=None)
    staged_final.add_argument("--max-train", type=int, default=None)
    staged_final.add_argument("--max-validation", type=int, default=None)
    staged_final.add_argument("--workers", type=int, default=12)
    staged_final.add_argument("--limit", type=int, default=None)

    g6 = commands.add_parser("g6")
    g6.add_argument("action", choices=["audit"])

    g7 = commands.add_parser("g7")
    g7.add_argument("action", choices=["audit"])

    g8 = commands.add_parser("g8")
    g8.add_argument("action", choices=["lock"])

    gates = commands.add_parser("gates")
    gates.add_argument("action", choices=["status", "assert"])
    gates.add_argument("--through", default="G5")

    pilot = commands.add_parser("pilot")
    pilot.add_argument("kind", choices=["functional"])
    pilot.add_argument("--seed", type=int, default=1103)
    pilot.add_argument("--device", type=int, default=0)
    pilot.add_argument("--epochs", type=int, default=120)
    pilot.add_argument("--batch-size", type=int, default=16)

    args = parser.parse_args()
    config = load_config(args.config)
    ensure_run_directories(config)
    evidence_root = Path(config["paths"]["evidence_root"])

    if args.command == "runtime":
        if args.action == "challengers":
            dino_report = dinov3_checkpoint_smoke(config)
            dense_report = vjepa21_checkpoint_smoke(config)
            dense_base_report = vjepa21_vitb_checkpoint_smoke(config)
            write_report(dino_report, Path(config["paths"]["report_root"]) / "dinov3_smoke.json")
            write_report(dense_report, Path(config["paths"]["report_root"]) / "vjepa21_vitl_smoke.json")
            write_report(dense_base_report, Path(config["paths"]["report_root"]) / "vjepa21_vitb_smoke.json")
            combined = {"dinov3": dino_report, "vjepa21_vitl": dense_report, "vjepa21_vitb": dense_base_report}
            _print(combined)
            raise SystemExit(0 if dino_report["passed"] and dense_report["passed"] and dense_base_report["passed"] else 2)
        report = gpu_smoke(config)
        write_report(report, evidence_root / "G0" / "runtime.json")
        write_report(report, evidence_root / "G0" / "hardware.json")
        checkpoint_report = echojepa_checkpoint_smoke(config) if report["passed"] else {
            "schema_version": 1,
            "passed": False,
            "error": "Skipped because the hardware/runtime smoke test failed",
        }
        write_report(checkpoint_report, evidence_root / "G0" / "echojepa_smoke.json")
        combined = {"hardware_runtime": report, "echojepa_checkpoint": checkpoint_report}
        _print(combined)
        raise SystemExit(0 if report["passed"] and checkpoint_report["passed"] else 2)
    if args.command == "inventory" and args.kind == "labels":
        report = run_label_inventory(config["paths"]["measurements"], config["paths"]["study_list"], config["targets"])
        write_report(report, Path(config["paths"]["report_root"]) / "label_inventory.json")
        _print(report)
        return
    if args.command == "inventory" and args.kind == "dicom":
        # The complete patient/study split is a day-zero prerequisite. Refuse
        # any DICOM decoding if the source-bound immutable manifest is absent or
        # no longer validates against the full record list.
        split_report = audit_hashed_split_manifest(
            config["paths"]["record_list"],
            config["paths"]["private_root"],
            seed=int(config["splits"]["seed"]),
            train=float(config["splits"]["train"]),
            validation=float(config["splits"]["validation"]),
        )
        report = run_dicom_inventory(
            config["paths"]["record_list"],
            config["paths"]["dicom_root"],
            Path(config["paths"]["private_root"]) / "dicom_inventory.csv",
            evidence_root / "G1" / "dicom_inventory_summary.json",
            workers=args.workers,
            limit=args.limit,
            resume=args.resume,
            expected_record_list_sha256=split_report["record_list_sha256"],
        )
        report["split_manifest_sha256"] = split_report["manifest_sha256"]
        report["split_manifest_preceded_decoding"] = True
        # A disconnected/partially mounted corpus must not look green merely
        # because one early file decoded. Any unreadable source instance needs
        # explicit disposition before this execution prerequisite can pass.
        report["execution_checks"] = {
            "full_source_scanned": bool(report["complete"]),
            "record_count_exact": report["records"] == report["record_list_records"],
            "all_instances_readable": report["readable"] == report["records"],
            "no_header_errors": report["errors"] == 0,
            "sop_instance_uids_complete": report["missing_sop_instance_uids"] == 0,
            "sop_instance_uids_unique": report["duplicate_sop_instance_uids"] == 0,
            "no_cross_patient_sop_uid_duplicates": report[
                "cross_patient_duplicate_sop_instance_uids"
            ] == 0,
            "spatial_calibration_present": report["spatially_calibrated"] > 0,
            "spectral_calibration_present": report["spectral_regions"] > 0,
            "multi_region_instances_present": report["multi_region"] > 0,
            "vendor_metadata_present": bool(report["manufacturers"]),
            "measurement_layout_candidate_complete": (
                sum(report["measurement_screen_layouts"].values()) == report["records"]
                and "UNAVAILABLE" not in report["measurement_screen_layouts"]
            ),
            "measurement_layout_provenance_explicit": (
                report["measurement_screen_layout_version"] == "hemfm.metadata_layout.v1"
                and report["measurement_screen_layout_provenance"]
                == "metadata_only_candidate_prior_not_annotation_truth"
                and report["measurement_screen_layout_is_annotation_truth"] is False
            ),
        }
        report["passed"] = all(report["execution_checks"].values())
        write_report(report, evidence_root / "G1" / "dicom_inventory_summary.json")
        _print(report)
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "splits":
        split_kwargs = {
            "record_list": config["paths"]["record_list"],
            "private_root": config["paths"]["private_root"],
            "seed": int(config["splits"]["seed"]),
            "train": float(config["splits"]["train"]),
            "validation": float(config["splits"]["validation"]),
        }
        if args.action == "build":
            report = build_hashed_split_manifest(**split_kwargs)
        else:
            report = audit_hashed_split_manifest(**split_kwargs)
        write_audit(report, evidence_root / "G1" / "split_audit.json")
        _print(report)
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "calibration":
        split_report = audit_hashed_split_manifest(
            config["paths"]["record_list"],
            config["paths"]["private_root"],
            seed=int(config["splits"]["seed"]),
            train=float(config["splits"]["train"]),
            validation=float(config["splits"]["validation"]),
        )
        report = validate_real_calibration(
            Path(config["paths"]["private_root"]) / "dicom_inventory.csv",
            config["paths"]["dicom_root"],
            evidence_root / "G1" / "calibration_tests.json",
            max_files=args.max_files,
            inventory_summary_json=evidence_root / "G1" / "dicom_inventory_summary.json",
            expected_split_manifest_sha256=split_report["manifest_sha256"],
            expected_record_list_sha256=split_report["record_list_sha256"],
        )
        _print(report)
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "cohort":
        report = build_endpoint_cohorts(config, confidence_min=args.confidence_min)
        _print(report)
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "augmentation":
        report = verify_physical_augmentation_parity(
            evidence_root / "G3" / "augmentation_physical_parity.json",
            trials=args.trials,
        )
        _print(report)
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "shortcuts":
        _assert_mimic_development_split_ready(config)
        report = run_shortcut_controls(
            config,
            max_train_per_target=args.train_per_target,
            max_validation_per_target=args.validation_per_target,
            workers=args.workers,
            device=args.device,
        )
        _print(report)
        passed = all(item["passed"] for item in report.values())
        raise SystemExit(0 if passed else 2)
    if args.command == "provenance":
        report = write_licence_provenance(config)
        _print(report)
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "architecture":
        _assert_mimic_development_split_ready(config)
        report = run_architecture_ladder(
            config,
            train_per_target=args.train_per_target,
            validation_per_target=args.validation_per_target,
        )
        _print(report)
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "tuning":
        _assert_mimic_development_split_ready(config)
        report = run_optimizer_peft_pilot(
            config,
            train_count=args.train_count,
            validation_count=args.validation_count,
            epochs=args.epochs,
            device=args.device,
        )
        ledger = write_decision_ledger(config, report)
        _print({"optimizer_peft_pilot": report, "decision_ledger": ledger})
        raise SystemExit(0 if report["passed"] and ledger["passed"] else 2)
    if args.command == "specialists":
        _assert_mimic_development_split_ready(config)
        if args.action == "extract":
            report = build_full_specialist_feature_cache(config)
        elif args.action == "train":
            report = run_frozen_specialists(config)
        else:
            report = run_multibackbone_fusion(config)
        _print(report)
        passed = (
            report["passed"]
            if "passed" in report
            else all(item["passed"] for item in report.values())
        )
        raise SystemExit(0 if passed else 2)
    if args.command == "dense-lv":
        a2c_manual = args.action in {"a2c-smoke", "a2c-train"}
        smoke_dense = args.action in {"smoke", "a2c-smoke"}
        dense_kwargs = {
            "views": ("2CH",) if a2c_manual else ("2CH", "4CH"),
            "annotation_mode": "manual_edes" if a2c_manual else "half_sequence",
            "run_tag": (
                f"camus_a2c_manual_edes_{args.backbone}_{'smoke' if smoke_dense else 'full'}"
                if a2c_manual
                else None
            ),
        }
        if args.action == "manifest":
            report = write_camus_acquisition_manifest(config)
        elif smoke_dense:
            report = run_camus_dense_pretraining(
                config,
                device=args.device,
                epochs=args.epochs or 2,
                frozen_epochs=args.frozen_epochs if args.frozen_epochs is not None else 1,
                maximum_train=args.max_train or 8,
                maximum_validation=args.max_validation or 4,
                mode="smoke",
                backbone=args.backbone,
                **dense_kwargs,
            )
        else:
            report = run_camus_dense_pretraining(
                config,
                device=args.device,
                epochs=args.epochs or 18,
                frozen_epochs=args.frozen_epochs if args.frozen_epochs is not None else 2,
                maximum_train=args.max_train,
                maximum_validation=args.max_validation,
                mode="full",
                backbone=args.backbone,
                **dense_kwargs,
            )
        _print({key: value for key, value in report.items() if key != "history"})
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "research-data" and args.action == "audit":
        report = audit_research_datasets(config)
        _print(report)
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "echonet-dynamic":
        if args.action in {"audit", "stage"} and not args.archive:
            parser.error("echonet-dynamic audit/stage requires --archive")
        if args.action == "audit":
            report = audit_echonet_dynamic_archive(
                config, args.archive, compute_hash=args.hash
            )
        elif args.action == "stage":
            report = stage_echonet_dynamic(
                config,
                args.archive,
                workers=args.workers,
                frames=args.frames,
                resolution=args.resolution,
                limit=args.limit,
            )
        elif args.action in {"smoke", "train"}:
            smoke = args.action == "smoke"
            report = run_echonet_dynamic_transfer(
                config,
                device=args.device,
                epochs=args.epochs or (3 if smoke else 10),
                maximum_train=args.max_train or (8 if smoke else None),
                maximum_validation=args.max_validation or (4 if smoke else None),
                mode="smoke" if smoke else "full",
            )
        elif args.action == "seed":
            if args.seed is None:
                parser.error("echonet-dynamic seed requires --seed")
            report = run_echonet_dynamic_transfer_seed(
                config,
                seed=args.seed,
                device=args.device,
                epochs=args.epochs or 10,
                maximum_train=args.max_train,
                maximum_validation=args.max_validation,
                mode="full",
            )
        else:
            report = finalize_echonet_dynamic_transfer(config, mode="full")
        _print(report)
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "echonet-trace":
        if args.action in {"smoke", "train"}:
            smoke = args.action == "smoke"
            report = run_echonet_trace_training(
                config,
                device=args.device,
                epochs=args.epochs or (3 if smoke else 8),
                maximum_train=args.max_train or (2 if smoke else None),
                maximum_validation=args.max_validation or (2 if smoke else None),
                mode="smoke" if smoke else "full",
            )
        elif args.action == "seed":
            if args.seed is None:
                parser.error("echonet-trace seed requires --seed")
            report = run_echonet_trace_seed(
                config,
                seed=args.seed,
                device=args.device,
                epochs=args.epochs or 8,
                maximum_train=args.max_train,
                maximum_validation=args.max_validation,
                mode="full",
            )
        else:
            report = finalize_echonet_trace(config, mode="full")
        _print({key: value for key, value in report.items() if key != "seeds"})
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "temporal":
        if args.action == "native-evaluate":
            if args.checkpoint is None:
                parser.error("temporal native-evaluate requires --checkpoint")
            report = evaluate_ted_native_inference(
                config,
                checkpoint=args.checkpoint,
                device=args.device,
                overlap_frames=args.overlap_frames,
                maximum_validation=args.max_validation,
            )
            _print({key: value for key, value in report.items() if key != "per_sequence"})
            raise SystemExit(0 if report["passed"] else 2)
        smoke = args.action == "smoke"
        report = run_ted_temporal_training(
            config,
            device=args.device,
            epochs=args.epochs or (2 if smoke else 10),
            frozen_epochs=args.frozen_epochs if args.frozen_epochs is not None else 1,
            maximum_train=args.max_train or (8 if smoke else None),
            maximum_validation=args.max_validation or (4 if smoke else None),
            mode="smoke" if smoke else "full",
            backbone=args.backbone,
            training_seed=args.seed,
        )
        _print({key: value for key, value in report.items() if key != "history"})
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "mimic-lv":
        if args.action == "stage":
            report = stage_mimic_lv_extension(
                config, source_root=args.source_root, metadata_path=args.metadata
            )
        else:
            smoke = args.action == "smoke"
            report = run_mimic_lv_training(
                config,
                device=args.device,
                epochs=args.epochs or (2 if smoke else 8),
                frozen_epochs=args.frozen_epochs if args.frozen_epochs is not None else 1,
                maximum_train=args.max_train or (8 if smoke else None),
                maximum_validation=args.max_validation or (4 if smoke else None),
                mode=("smoke" if smoke else "full") + (f"_{args.run_tag}" if args.run_tag else ""),
                backbone=args.backbone,
                ema_decay=args.ema_decay,
            )
        _print(report)
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "unity-landmarks":
        smoke = args.action == "smoke"
        report = run_unity_landmark_training(
            config,
            device=args.device,
            epochs=args.epochs or (2 if smoke else 8),
            frozen_epochs=args.frozen_epochs if args.frozen_epochs is not None else 1,
            maximum_train=args.max_train or (16 if smoke else None),
            maximum_validation=args.max_validation or (8 if smoke else None),
            mode="smoke" if smoke else "full",
        )
        _print({key: value for key, value in report.items() if key != "history"})
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "ev9v-view":
        smoke = args.action == "smoke"
        report = run_ev9v_view_training(
            config,
            device=args.device,
            epochs=args.epochs or (2 if smoke else 8),
            frozen_epochs=args.frozen_epochs if args.frozen_epochs is not None else 1,
            maximum_train=args.max_train or (16 if smoke else None),
            maximum_validation=args.max_validation or (8 if smoke else None),
            mode="smoke" if smoke else "full",
        )
        _print({key: value for key, value in report.items() if key != "history"})
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "external-ood":
        smoke = args.action == "smoke"
        report = run_external_ood_audit(
            config,
            device=args.device,
            maximum_per_dataset=args.max_per_dataset or (2 if smoke else None),
            mode="smoke" if smoke else "full",
        )
        _print({key: value for key, value in report.items() if key != "records"})
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "panecho":
        smoke = args.action == "smoke"
        report = run_panecho_mimic_lv_audit(
            config,
            device=args.device,
            maximum_train=args.max_train or (20 if smoke else None),
            maximum_validation=args.max_validation or (10 if smoke else None),
            mode="smoke" if smoke else "full",
        )
        _print(report)
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "staged-final":
        if args.action in {"cache", "smoke", "train"}:
            _assert_mimic_development_split_ready(config)
        if args.action == "cache":
            report = stage_staged_scalar_cache(
                config, workers=args.workers, limit=args.limit
            )
        elif args.action == "audit":
            report = run_staged_scalar_uncertainty_audit(
                config,
                target=args.target,
                device=args.device,
            )
        else:
            smoke = args.action == "smoke"
            report = run_staged_scalar_target(
                config,
                target=args.target,
                device=args.device,
                epochs=args.epochs or (3 if smoke else 10),
                maximum_train=args.max_train or (8 if smoke else None),
                maximum_validation=args.max_validation or (4 if smoke else None),
                mode="smoke" if smoke else "full",
            )
        _print({key: value for key, value in report.items() if key != "seeds"})
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "g6" and args.action == "audit":
        report = run_specialist_holdout_and_failure_audit(config)
        _print(report)
        raise SystemExit(0 if all(item["passed"] for item in report.values()) else 2)
    if args.command == "g7" and args.action == "audit":
        report = run_integration_audit(config)
        _print(report)
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "g8" and args.action == "lock":
        report = run_external_lock(config)
        _print(report)
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "gates" and args.action == "status":
        _print(evidence_state(config))
        return
    if args.command == "gates" and args.action == "assert":
        _print(assert_through(config, args.through))
        return
    if args.command == "pilot":
        _assert_mimic_development_split_ready(config)
        pilot_config = CachedPilotConfig(
            seed=args.seed,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=min(30, max(5, args.epochs // 4)),
        )
        report = run_cached_functional_pilot(
            config["paths"]["prior_feature_root"],
            Path(config["paths"]["run_root"]) / "pilots",
            pilot_config,
        )
        _print({key: value for key, value in report.items() if key != "history"})
        return


if __name__ == "__main__":
    main()
