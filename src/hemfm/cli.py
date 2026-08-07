from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ensure_run_directories, load_config
from .calibration_validation import validate_real_calibration
from .augmentations import verify_physical_augmentation_parity
from .architecture_pilot import run_architecture_ladder
from .cohorts import build_endpoint_cohorts
from .gates import assert_through, evidence_state
from .full_extraction import build_full_specialist_feature_cache
from .frozen_specialists import run_frozen_specialists
from .inventory import run_dicom_inventory, run_label_inventory
from .optimizer_peft_pilot import run_optimizer_peft_pilot, write_decision_ledger
from .provenance_audit import write_licence_provenance
from .runtime import dinov3_checkpoint_smoke, echojepa_checkpoint_smoke, gpu_smoke, vjepa21_checkpoint_smoke, vjepa21_vitb_checkpoint_smoke, write_report
from .shortcut_controls import run_shortcut_controls
from .splits import audit_existing_index, write_audit
from .train_cached import CachedPilotConfig, run_cached_functional_pilot


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2))


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
    splits.add_argument("action", choices=["audit"])

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
    specialists.add_argument("action", choices=["extract", "train"])

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
        report = run_dicom_inventory(
            config["paths"]["record_list"],
            config["paths"]["dicom_root"],
            Path(config["paths"]["private_root"]) / "dicom_inventory.csv",
            evidence_root / "G1" / "dicom_inventory_summary.json",
            workers=args.workers,
            limit=args.limit,
            resume=args.resume,
        )
        report["passed"] = bool(report["complete"] and report["readable"] > 0 and report["spatially_calibrated"] > 0)
        write_report(report, evidence_root / "G1" / "dicom_inventory_summary.json")
        _print(report)
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "splits":
        path = Path(config["paths"]["prior_feature_root"]) / "study_index.csv"
        report = audit_existing_index(path)
        report["passed"] = True
        write_audit(report, evidence_root / "G1" / "split_audit.json")
        _print(report)
        return
    if args.command == "calibration":
        report = validate_real_calibration(
            Path(config["paths"]["private_root"]) / "dicom_inventory.csv",
            config["paths"]["dicom_root"],
            evidence_root / "G1" / "calibration_tests.json",
            max_files=args.max_files,
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
        report = run_architecture_ladder(
            config,
            train_per_target=args.train_per_target,
            validation_per_target=args.validation_per_target,
        )
        _print(report)
        raise SystemExit(0 if report["passed"] else 2)
    if args.command == "tuning":
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
        report = (
            build_full_specialist_feature_cache(config)
            if args.action == "extract"
            else run_frozen_specialists(config)
        )
        _print(report)
        passed = (
            report["passed"]
            if "passed" in report
            else all(item["passed"] for item in report.values())
        )
        raise SystemExit(0 if passed else 2)
    if args.command == "gates" and args.action == "status":
        _print(evidence_state(config))
        return
    if args.command == "gates" and args.action == "assert":
        _print(assert_through(config, args.through))
        return
    if args.command == "pilot":
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
