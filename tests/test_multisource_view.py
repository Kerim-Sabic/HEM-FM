from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hemfm.multisource_view import (
    ABSTENTION_CALIBRATION_ACCURACY_TARGET,
    CLASSES,
    ViewSample,
    _abstention_threshold,
    _deduplicate_paths,
    _metrics,
    _partition_groups,
    _temperature,
)


def test_group_partition_is_deterministic_disjoint_and_class_covered() -> None:
    rows = [
        (f"{label}-group-{index}", label)
        for label in ("A2C", "A4C", "PLAX")
        for index in range(30)
    ]

    first = _partition_groups(rows, source="unit", minimum=5)
    second = _partition_groups(rows, source="unit", minimum=5)

    assert first == second
    assert set(first.values()) == {"calibration", "audit"}
    for partition in ("calibration", "audit"):
        labels = [label for group, label in rows if first[group] == partition]
        assert all(labels.count(label) >= 5 for label in ("A2C", "A4C", "PLAX"))


def _sample(path: Path, *, label: str, partition: str, name: str) -> ViewSample:
    return ViewSample(
        source="unity",
        group=name,
        name=name,
        label=label,
        paths=(path,),
        partition=partition,
    )


def test_exact_duplicate_policy_quarantines_only_same_partition_same_label(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"same pixels")
    second.write_bytes(b"same pixels")

    retained, evidence = _deduplicate_paths(
        [
            _sample(first, label="A4C", partition="train", name="a"),
            _sample(second, label="A4C", partition="train", name="b"),
        ]
    )

    assert len(retained) == 1
    assert len(evidence) == 1
    assert evidence[0]["label"] == "A4C"

    with pytest.raises(ValueError, match="cross development partitions"):
        _deduplicate_paths(
            [
                _sample(first, label="A4C", partition="train", name="a"),
                _sample(second, label="A4C", partition="audit", name="b"),
            ]
        )
    with pytest.raises(ValueError, match="conflicting labels"):
        _deduplicate_paths(
            [
                _sample(first, label="A4C", partition="train", name="a"),
                _sample(second, label="A2C", partition="train", name="b"),
            ]
        )


def test_metrics_are_perfect_for_perfect_predictions_across_sources() -> None:
    targets = np.arange(len(CLASSES), dtype=np.int64)
    logits = np.full((len(CLASSES), len(CLASSES)), -10.0)
    logits[np.arange(len(CLASSES)), targets] = 10.0
    sources = ["unity" if index < 5 else "ev9v" for index in range(len(CLASSES))]

    result = _metrics(logits, targets, sources)

    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert all(value == 1.0 for value in result["per_class_recall"].values())


def test_temperature_and_abstention_are_finite_and_calibration_only() -> None:
    logits = np.asarray(
        [[8.0, 0.0], [7.0, 0.0], [0.0, 6.0], [5.0, 0.0], [0.0, 4.0]],
        dtype=np.float64,
    )
    targets = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)

    temperature = _temperature(logits, targets)
    threshold = _abstention_threshold(logits / temperature, targets)

    assert np.isfinite(temperature)
    assert 0.25 <= temperature <= 4.0
    assert 0.0 <= threshold["threshold"] <= 1.0
    assert 0.0 <= threshold["calibration_coverage"] <= 1.0
    assert threshold["calibration_accuracy_target"] == ABSTENTION_CALIBRATION_ACCURACY_TARGET
    if threshold["calibration_retained_accuracy"] is not None:
        assert (
            threshold["calibration_retained_accuracy"]
            >= ABSTENTION_CALIBRATION_ACCURACY_TARGET
        )


def test_abstention_uses_calibration_safety_margin_not_audit_gate() -> None:
    logits = np.asarray(
        [
            [8.0, 0.0],
            [7.0, 0.0],
            [6.0, 0.0],
            [5.0, 0.0],
            [0.0, 4.0],
            [3.0, 0.0],
        ],
        dtype=np.float64,
    )
    targets = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)

    result = _abstention_threshold(logits, targets)

    assert result["calibration_retained_accuracy"] == 1.0
    assert result["calibration_coverage"] < 1.0
