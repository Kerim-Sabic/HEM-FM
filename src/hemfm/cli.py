Exit code: 0
Wall time: 1.1 seconds
Output:
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
from .splits import audit_existing_index, write_audit
from .staged_final_scalar import run_staged_scalar_target, stage_staged_scalar_cache
from .ted_temporal import run_ted_temporal_training
from .unity_landmarks import run_unity_landmark_training
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
    specialists.add_argument("action", choices=["extract", "train", "fusion"])

    dense_lv = commands.add_parser("dense-lv")
    dense_lv.add_argument("action", choices=["manifest", "smoke", "train"])
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
    temporal.add_argument("action", choices=["smoke", "train"])
    temporal.add_argument("--device", type=int, default=0)
    temporal.add_argument("--epochs", type=int, default=None)
    temporal.add_argument("--frozen-epochs", type=int, default=None)
    temporal.add_argument("--max-train", type=int, default=None)
    temporal.add_argument("--max-validation", type=int, default=None)
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
    staged_final.add_argument("action", choices=["cache", "smoke", "train"])
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
        if args.action == "manifest":
            report = write_camus_acquisition_manifest(config)
        elif args.action == "smoke":
            report = run_camus_dense_pretraining(
                config,
                device=args.device,
                epochs=args.epochs or 2,
                frozen_epochs=args.frozen_epochs if args.frozen_epochs is not None else 1,
                maximum_train=args.max_train or 8,
                maximum_validation=args.max_validation or 4,
                mode="smoke",
                backbone=args.backbone,
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
        if args.action == "cache":
            report = stage_staged_scalar_cache(
                config, workers=args.workers, limit=args.limit
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

