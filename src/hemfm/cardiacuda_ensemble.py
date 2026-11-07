from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .cardiacuda_segmentation import (
    BASELINE_SCOPE,
    BASELINE_VIEW,
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_REPLICATES,
    BOUNDARY_METRIC_NAME,
    BOUNDARY_TOLERANCE_PIXELS,
    CardiacUDCFrameDataset,
    GLOBAL_CLASSES,
    HD95_METRIC_NAME,
    IGNORE_INDEX,
    TRANSFORM_POLICY,
    VIEW_NAMES,
    ViewConditionedUNet,
    _assert_nested_finite,
    _bootstrap_mean_ci,
    _canonical_fingerprint,
    _file_sha256,
    _policy_fingerprint,
    _surface_statistics,
    _atomic_json_write,
    build_manifest,
    masked_logits,
)


ENSEMBLE_SCOPE = "cardiacuda_a4c_three_seed_development_ensemble_audit"
DEFAULT_SEEDS = (1103, 2207, 3301)
CALIBRATION_FRACTION = 0.60
TARGET_CALIBRATION_COVERAGE = 0.80
DEVELOPMENT_FAILURE_DICE = 0.80
RELIABILITY_BINS = 15
RISK_COVERAGES = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
UNCERTAINTY_POLICY = {
    "schema_version": 1,
    "entropy_weight": 0.40,
    "aleatoric_weight": 0.30,
    "epistemic_jensen_shannon_weight": 0.30,
    "normalization": "divide entropy and Jensen-Shannon terms by log(active class count)",
    "abstention": {
        "selection_partition": "development_validation_calibration",
        "target_coverage": TARGET_CALIBRATION_COVERAGE,
        "rule": "retain patient when mean pixel uncertainty <= calibration quantile",
    },
}


def _patient_partition(patient_id: str, manifest_fingerprint: str) -> str:
    if not patient_id or len(manifest_fingerprint) != 64:
        raise ValueError("Patient partition requires a nonblank identity and manifest fingerprint")
    digest = hashlib.sha256(
        f"{manifest_fingerprint}:ensemble-calibration-v1:{patient_id}".encode()
    ).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    return "calibration" if fraction < CALIBRATION_FRACTION else "audit"


def _patient_pseudonym(patient_id: str, manifest_fingerprint: str) -> str:
    return hashlib.sha256(
        f"{manifest_fingerprint}:failure-library-v1:{patient_id}".encode()
    ).hexdigest()[:16]


def _validate_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_manifest_fingerprint: str,
) -> dict[str, Any]:
    if payload.get("schema_version") != 2 or payload.get("scope") != BASELINE_SCOPE:
        raise ValueError("Chamber checkpoint has the wrong schema or scope")
    if payload.get("sealed_test_accessed") is not False:
        raise ValueError("Chamber checkpoint does not attest a sealed TEST boundary")
    if payload.get("seed") != expected_seed:
        raise ValueError("Chamber checkpoint seed ancestry does not match its run")
    if payload.get("manifest_fingerprint") != expected_manifest_fingerprint:
        raise ValueError("Chamber checkpoint manifest ancestry differs from this audit")
    if payload.get("transform_policy_fingerprint") != _policy_fingerprint(TRANSFORM_POLICY):
        raise ValueError("Chamber checkpoint preprocessing policy differs from this audit")
    if tuple(payload.get("global_classes", ())) != GLOBAL_CLASSES:
        raise ValueError("Chamber checkpoint class ontology differs from this audit")
    if tuple(payload.get("view_names", ())) != VIEW_NAMES:
        raise ValueError("Chamber checkpoint view ontology differs from this audit")
    if tuple(payload.get("trained_views", ())) != (BASELINE_VIEW,):
        raise ValueError("Chamber checkpoint is not the exact A4C-only baseline")
    if tuple(payload.get("validation_views", ())) != (BASELINE_VIEW,):
        raise ValueError("Chamber checkpoint validation-view ancestry is incomplete")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Chamber checkpoint configuration is missing")
    if payload.get("config_fingerprint") != _canonical_fingerprint(config):
        raise ValueError("Chamber checkpoint configuration fingerprint is invalid")
    settings = payload.get("settings")
    if not isinstance(settings, Mapping) or config.get("settings") != settings:
        raise ValueError("Chamber checkpoint settings are not bound to its configuration")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Chamber checkpoint model state is missing")
    _assert_nested_finite(state, "ensemble checkpoint")
    return dict(settings)


def _load_models(
    checkpoint_paths: Sequence[Path],
    seeds: Sequence[int],
    *,
    manifest_fingerprint: str,
    device: str,
) -> tuple[list[ViewConditionedUNet], list[dict[str, Any]], dict[str, Any]]:
    import torch

    if len(checkpoint_paths) != len(seeds) or len(checkpoint_paths) < 2:
        raise ValueError("Ensemble audit requires one checkpoint per seed and at least two seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Ensemble seed identities must be unique")
    models: list[ViewConditionedUNet] = []
    evidence: list[dict[str, Any]] = []
    reference_settings: dict[str, Any] | None = None
    for path, seed in zip(checkpoint_paths, seeds, strict=True):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing ensemble checkpoint: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("Chamber checkpoint payload is not a mapping")
        settings = _validate_checkpoint(
            payload,
            expected_seed=int(seed),
            expected_manifest_fingerprint=manifest_fingerprint,
        )
        functional_settings = {
            key: settings[key]
            for key in ("resolution", "base_channels")
            if key in settings
        }
        if set(functional_settings) != {"resolution", "base_channels"}:
            raise ValueError("Chamber checkpoint lacks functional inference settings")
        if reference_settings is None:
            reference_settings = functional_settings
        elif functional_settings != reference_settings:
            raise ValueError("Ensemble checkpoints have incompatible inference settings")
        model = ViewConditionedUNet(base_channels=int(settings["base_channels"]))
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        models.append(model)
        evidence.append(
            {
                "seed": int(seed),
                "filename": path.name,
                "sha256": _file_sha256(path),
                "config_fingerprint": payload["config_fingerprint"],
                "best_validation_macro_dice": float(
                    payload["best_validation"]["macro_dice"]
                ),
                "sealed_test_accessed": False,
            }
        )
    assert reference_settings is not None
    return models, evidence, reference_settings


def _ensemble_prediction(models, images, views, class_valid):
    import torch
    import torch.nn.functional as functional

    member_probabilities = []
    member_aleatoric = []
    for model in models:
        outputs = model(images, views)
        _assert_nested_finite(outputs, "ensemble validation outputs")
        probabilities = functional.softmax(
            masked_logits(outputs["logits"].float(), class_valid.bool()), dim=1
        )
        member_probabilities.append(probabilities)
        member_aleatoric.append(torch.sigmoid(outputs["log_variance"].float().squeeze(1)))
    stacked = torch.stack(member_probabilities, dim=0)
    mean_probability = stacked.mean(dim=0)
    active_count = class_valid.sum(dim=1).float().clamp_min(2.0)
    normalizer = active_count.log()[:, None, None]
    entropy = -(
        mean_probability.clamp_min(1e-8).log() * mean_probability
    ).sum(dim=1) / normalizer
    member_entropy = -(
        stacked.clamp_min(1e-8).log() * stacked
    ).sum(dim=2).mean(dim=0) / normalizer
    epistemic = (entropy - member_entropy).clamp(0, 1)
    aleatoric = torch.stack(member_aleatoric, dim=0).mean(dim=0)
    uncertainty = (
        UNCERTAINTY_POLICY["entropy_weight"] * entropy
        + UNCERTAINTY_POLICY["aleatoric_weight"] * aleatoric
        + UNCERTAINTY_POLICY["epistemic_jensen_shannon_weight"] * epistemic
    ).clamp(0, 1)
    return mean_probability, uncertainty, entropy, aleatoric, epistemic


def _new_patient_totals() -> dict[str, Any]:
    return {
        "uncertainty_sum": 0.0,
        "pixel_count": 0,
        "classes": defaultdict(
            lambda: {
                "intersection": 0,
                "denominator": 0,
                "truth_pixels": 0,
                "surface_distances": [],
                "matched_boundary": 0,
                "boundary_denominator": 0,
            }
        ),
    }


def _patient_rows(totals: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for patient_id, patient in totals.items():
        dice: dict[str, float] = {}
        hd95: dict[str, float] = {}
        boundary: dict[str, float] = {}
        for name, values in patient["classes"].items():
            if int(values["truth_pixels"]) == 0:
                continue
            denominator = int(values["denominator"])
            if denominator <= 0:
                raise ValueError("Positive reference support has an invalid Dice denominator")
            dice[name] = 2.0 * int(values["intersection"]) / denominator
            boundary_denominator = int(values["boundary_denominator"])
            boundary[name] = (
                1.0
                if boundary_denominator == 0
                else int(values["matched_boundary"]) / boundary_denominator
            )
            chunks = values["surface_distances"]
            hd95[name] = (
                float(np.quantile(np.concatenate(chunks), 0.95)) if chunks else 0.0
            )
        if not dice or int(patient["pixel_count"]) <= 0:
            raise ValueError("Patient has no evaluable chamber evidence")
        rows.append(
            {
                "patient_id": patient_id,
                "macro_dice": float(np.mean(list(dice.values()))),
                "dice": dice,
                HD95_METRIC_NAME: hd95,
                BOUNDARY_METRIC_NAME: boundary,
                "mean_uncertainty": float(patient["uncertainty_sum"])
                / int(patient["pixel_count"]),
            }
        )
    return sorted(rows, key=lambda row: row["patient_id"])


def _binary_auroc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    from scipy.stats import rankdata

    y = np.asarray(labels, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or values.shape != y.shape or not np.isfinite(values).all():
        raise ValueError("AUROC requires aligned finite one-dimensional inputs")
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(values, method="average")
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _calibration_threshold(values: Sequence[float], target_coverage: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.ndim != 1 or not len(ordered) or not np.isfinite(ordered).all():
        raise ValueError("Abstention calibration requires finite patient uncertainty")
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("Target coverage must be in (0, 1]")
    index = max(0, min(len(ordered) - 1, math.ceil(target_coverage * len(ordered)) - 1))
    return float(ordered[index])


def _risk_coverage(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (float(row["mean_uncertainty"]), row["patient_id"]))
    if not ordered:
        raise ValueError("Risk-coverage analysis requires patients")
    result = []
    for requested in RISK_COVERAGES:
        retained = max(1, min(len(ordered), math.ceil(requested * len(ordered))))
        selected = ordered[:retained]
        dices = np.asarray([float(row["macro_dice"]) for row in selected])
        result.append(
            {
                "requested_coverage": requested,
                "realized_coverage": retained / len(ordered),
                "retained_patients": retained,
                "mean_macro_dice": float(dices.mean()),
                "selective_risk_one_minus_dice": float((1.0 - dices).mean()),
            }
        )
    return result


def _abstention_summary(
    audit_rows: Sequence[Mapping[str, Any]],
    retained: Sequence[Mapping[str, Any]],
    abstained: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    if not audit_rows or not retained:
        raise ValueError("Abstention audit requires nonempty audit and retained cohorts")
    audit_dices = np.asarray([float(row["macro_dice"]) for row in audit_rows])
    retained_dices = np.asarray([float(row["macro_dice"]) for row in retained])
    before = float((1.0 - audit_dices).mean())
    after = float((1.0 - retained_dices).mean())
    improves = after < before
    return {
        "threshold": float(threshold),
        "threshold_selected_without_audit_labels": True,
        "calibration_target_coverage": TARGET_CALIBRATION_COVERAGE,
        "audit_coverage": len(retained) / len(audit_rows),
        "audit_retained_patients": len(retained),
        "audit_abstained_patients": len(abstained),
        "audit_macro_dice_before": float(audit_dices.mean()),
        "audit_macro_dice_after": float(retained_dices.mean()),
        "audit_selective_risk_before": before,
        "audit_selective_risk_after": after,
        "improves_selective_risk": improves,
        "gate_status": "passed" if improves else "failed",
    }


def _reliability_report(counts, confidence_sums, correct_sums) -> dict[str, Any]:
    counts = np.asarray(counts, dtype=np.int64)
    confidence_sums = np.asarray(confidence_sums, dtype=np.float64)
    correct_sums = np.asarray(correct_sums, dtype=np.float64)
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("Pixel reliability audit received no valid pixels")
    bins = []
    ece = 0.0
    for index in range(len(counts)):
        count = int(counts[index])
        accuracy = float(correct_sums[index] / count) if count else None
        confidence = float(confidence_sums[index] / count) if count else None
        if count:
            ece += count / total * abs(float(accuracy) - float(confidence))
        bins.append(
            {
                "lower": index / len(counts),
                "upper": (index + 1) / len(counts),
                "pixels": count,
                "accuracy": accuracy,
                "mean_confidence": confidence,
            }
        )
    return {"pixels": total, "ece": float(ece), "bins": bins}


def _summarize_rows(rows: Sequence[Mapping[str, Any]], bootstrap_seed: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("Patient summary requires at least one row")
    per_class: dict[str, Any] = {}
    for class_offset, name in enumerate(GLOBAL_CLASSES[1:]):
        available = [row for row in rows if name in row["dice"]]
        if not available:
            continue
        per_class[name] = {}
        for metric_offset, metric in enumerate(
            ("dice", HD95_METRIC_NAME, BOUNDARY_METRIC_NAME)
        ):
            per_class[name][metric] = _bootstrap_mean_ci(
                [float(row[metric][name]) for row in available],
                replicates=BOOTSTRAP_REPLICATES,
                confidence=BOOTSTRAP_CONFIDENCE,
                seed=bootstrap_seed + class_offset * 101 + metric_offset,
            )
    macro = np.asarray([float(row["macro_dice"]) for row in rows])
    return {
        "patients": len(rows),
        "macro_dice": _bootstrap_mean_ci(
            macro,
            replicates=BOOTSTRAP_REPLICATES,
            confidence=BOOTSTRAP_CONFIDENCE,
            seed=bootstrap_seed + 1009,
        ),
        "per_class": per_class,
    }


def run_ensemble_audit(
    dataset_root: Path,
    checkpoint_paths: Sequence[Path],
    output_dir: Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    device: str = "cuda:0",
    validation_domain: str = "Site_R",
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite ensemble audit evidence: {output_dir}")
    manifest = build_manifest(Path(dataset_root), validation_domain=validation_domain)
    if not bool(manifest.development_file_inventory.get("approved")):
        raise ValueError("Ensemble audit is blocked by an unapproved development inventory")
    if any(frame.volume.sealed for frame in manifest.validation_frames):
        raise ValueError("Sealed TEST frames entered the development ensemble audit")
    models, checkpoint_evidence, settings = _load_models(
        checkpoint_paths,
        seeds,
        manifest_fingerprint=manifest.fingerprint,
        device=device,
    )
    dataset = CardiacUDCFrameDataset(
        manifest.validation_frames,
        resolution=int(settings["resolution"]),
        augment=False,
        cache_size=2,
        preload=True,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    patient_totals: dict[str, dict[str, Any]] = defaultdict(_new_patient_totals)
    reliability = {
        name: {
            "counts": np.zeros(RELIABILITY_BINS, dtype=np.int64),
            "confidence": np.zeros(RELIABILITY_BINS, dtype=np.float64),
            "correct": np.zeros(RELIABILITY_BINS, dtype=np.float64),
        }
        for name in ("calibration", "audit")
    }
    common_grid_shape: tuple[int, int] | None = None
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            views = batch["view_id"].to(device)
            class_valid = batch["class_valid"].to(device)
            probabilities, uncertainty, _entropy, _aleatoric, _epistemic = (
                _ensemble_prediction(models, images, views, class_valid)
            )
            if not bool(torch.isfinite(probabilities).all()) or not bool(
                torch.isfinite(uncertainty).all()
            ):
                raise FloatingPointError("Non-finite ensemble prediction")
            prediction = probabilities.argmax(dim=1)
            confidence = probabilities.max(dim=1).values
            grid_shape = (int(targets.shape[-2]), int(targets.shape[-1]))
            if common_grid_shape is None:
                common_grid_shape = grid_shape
            elif common_grid_shape != grid_shape:
                raise ValueError("Ensemble evaluation grid changed within one audit")
            for index in range(images.shape[0]):
                patient_id = str(batch["patient_id"][index])
                partition = _patient_partition(patient_id, manifest.fingerprint)
                target = targets[index].detach().cpu().numpy().astype(np.int16)
                predicted = prediction[index].detach().cpu().numpy().astype(np.int16)
                uncertainty_array = uncertainty[index].detach().cpu().numpy().astype(np.float64)
                confidence_array = confidence[index].detach().cpu().numpy().astype(np.float64)
                valid = target != IGNORE_INDEX
                patient = patient_totals[patient_id]
                patient["uncertainty_sum"] += float(uncertainty_array[valid].sum())
                patient["pixel_count"] += int(valid.sum())
                correct = predicted[valid] == target[valid]
                valid_confidence = confidence_array[valid]
                bin_index = np.minimum(
                    (valid_confidence * RELIABILITY_BINS).astype(np.int64),
                    RELIABILITY_BINS - 1,
                )
                accumulator = reliability[partition]
                accumulator["counts"] += np.bincount(
                    bin_index, minlength=RELIABILITY_BINS
                )
                accumulator["confidence"] += np.bincount(
                    bin_index, weights=valid_confidence, minlength=RELIABILITY_BINS
                )
                accumulator["correct"] += np.bincount(
                    bin_index, weights=correct.astype(np.float64), minlength=RELIABILITY_BINS
                )
                for class_index, name in enumerate(GLOBAL_CLASSES[1:], start=1):
                    if not bool(class_valid[index, class_index]):
                        continue
                    values = patient["classes"][name]
                    predicted_mask = (predicted == class_index) & valid
                    truth_mask = (target == class_index) & valid
                    values["intersection"] += int((predicted_mask & truth_mask).sum())
                    values["denominator"] += int(predicted_mask.sum() + truth_mask.sum())
                    values["truth_pixels"] += int(truth_mask.sum())
                    distances, matched, boundary_denominator = _surface_statistics(
                        predicted_mask,
                        truth_mask,
                        tolerance_pixels=BOUNDARY_TOLERANCE_PIXELS,
                    )
                    if len(distances):
                        values["surface_distances"].append(distances)
                    values["matched_boundary"] += matched
                    values["boundary_denominator"] += boundary_denominator

    rows = _patient_rows(patient_totals)
    calibration_rows = [
        row
        for row in rows
        if _patient_partition(row["patient_id"], manifest.fingerprint) == "calibration"
    ]
    audit_rows = [row for row in rows if row not in calibration_rows]
    if len(calibration_rows) < 20 or len(audit_rows) < 20:
        raise ValueError("Patient-disjoint calibration/audit split is unexpectedly small")
    threshold = _calibration_threshold(
        [row["mean_uncertainty"] for row in calibration_rows],
        TARGET_CALIBRATION_COVERAGE,
    )
    retained = [row for row in audit_rows if float(row["mean_uncertainty"]) <= threshold]
    abstained = [row for row in audit_rows if row not in retained]
    failure_labels = [
        float(row["macro_dice"]) < DEVELOPMENT_FAILURE_DICE for row in audit_rows
    ]
    failure_auroc = _binary_auroc(
        failure_labels, [float(row["mean_uncertainty"]) for row in audit_rows]
    )
    failure_cases = sorted(
        audit_rows,
        key=lambda row: (float(row["macro_dice"]), -float(row["mean_uncertainty"])),
    )[:20]
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_report = {
        "schema_version": 1,
        "scope": ENSEMBLE_SCOPE,
        "manifest_fingerprint": manifest.fingerprint,
        "patient_identifiers": "pseudonymized; no paths or raw identifiers",
        "cases": [
            {
                "patient_pseudonym": _patient_pseudonym(
                    row["patient_id"], manifest.fingerprint
                ),
                "macro_dice": row["macro_dice"],
                "dice": row["dice"],
                HD95_METRIC_NAME: row[HD95_METRIC_NAME],
                BOUNDARY_METRIC_NAME: row[BOUNDARY_METRIC_NAME],
                "mean_uncertainty": row["mean_uncertainty"],
                "abstained": float(row["mean_uncertainty"]) > threshold,
            }
            for row in failure_cases
        ],
        "sealed_test_accessed": False,
    }
    failure_path = output_dir / "failure_cases.json"
    _atomic_json_write(failure_path, failure_report)
    abstention = _abstention_summary(
        audit_rows,
        retained,
        abstained,
        threshold=threshold,
    )
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": ENSEMBLE_SCOPE,
        "capability_statement": (
            "Three-seed single-frame A4C LV/RV/LA/RA development ensemble; "
            "not general-view segmentation, measurement, or clinical evidence"
        ),
        "manifest_fingerprint": manifest.fingerprint,
        "validation_domain": manifest.validation_domain,
        "validation_frames": len(manifest.validation_frames),
        "common_grid_shape": list(common_grid_shape) if common_grid_shape else None,
        "checkpoints": checkpoint_evidence,
        "uncertainty_policy": UNCERTAINTY_POLICY,
        "patient_split": {
            "method": "manifest-bound SHA-256 patient partition",
            "calibration_fraction_target": CALIBRATION_FRACTION,
            "calibration_patients": len(calibration_rows),
            "audit_patients": len(audit_rows),
            "overlap": 0,
        },
        "development_validation_all": _summarize_rows(rows, 7103),
        "development_validation_calibration": _summarize_rows(calibration_rows, 7207),
        "development_validation_audit": _summarize_rows(audit_rows, 7301),
        "pixel_reliability": {
            name: _reliability_report(
                values["counts"], values["confidence"], values["correct"]
            )
            for name, values in reliability.items()
        },
        "failure_detection": {
            "definition": f"patient macro Dice < {DEVELOPMENT_FAILURE_DICE}",
            "status": "development diagnostic only; not a promotion threshold",
            "audit_failures": int(sum(failure_labels)),
            "audit_nonfailures": len(failure_labels) - int(sum(failure_labels)),
            "uncertainty_auroc": failure_auroc,
        },
        "abstention": abstention,
        "risk_coverage": _risk_coverage(audit_rows),
        "failure_cases": {
            "path": failure_path.name,
            "sha256": _file_sha256(failure_path),
            "count": len(failure_cases),
        },
        "calibration_audit_complete": True,
        "uncertainty_promotion_eligible": bool(abstention["improves_selective_risk"]),
        "uncertainty_status": (
            "development abstention gate passed"
            if abstention["improves_selective_risk"]
            else "failed: development abstention worsened selective risk"
        ),
        "near_duplicate_perceptual_audit_complete": False,
        "locked_test_evaluation_complete": False,
        "sealed_test_accessed": False,
        "promotion_eligible": False,
        "promotion_status": (
            "challenger_only; perceptual duplicate audit, external evidence, and locked TEST "
            "gate remain incomplete"
        ),
    }
    _assert_nested_finite(report, "ensemble audit report")
    _atomic_json_write(output_dir / "report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Three-seed CardiacUDC A4C development ensemble audit"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-domain", default="Site_R")
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS)
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    checkpoints = [
        args.run_root / f"cardiacuda_segmentation_seed_{seed}_full" / "best.pt"
        for seed in seeds
    ]
    report = run_ensemble_audit(
        args.root,
        checkpoints,
        args.output_dir,
        seeds=seeds,
        device=args.device,
        validation_domain=args.validation_domain,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
