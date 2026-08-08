import random

import numpy as np

from hemfm.ev9v_view import VIEWS, _classification_metrics, _sample_indices


def test_ev9v_temporal_sampling_is_bounded_and_ordered():
    indices = _sample_indices(133, 8, True, random.Random(7))
    assert len(indices) == 8
    assert np.all(np.diff(indices) >= 0)
    assert indices.min() >= 0
    assert indices.max() < 133


def test_ev9v_uniform_sampling_includes_endpoints():
    assert _sample_indices(20, 4, False, random.Random(1)).tolist() == [0, 6, 13, 19]


def test_ev9v_metrics_are_perfect_for_perfect_logits():
    targets = np.arange(len(VIEWS))
    logits = np.eye(len(VIEWS), dtype=np.float32) * 20.0
    metrics = _classification_metrics(logits, targets)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0

