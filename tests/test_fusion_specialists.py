from __future__ import annotations

import torch

from hemfm.fusion_specialists import _fusion_model


def test_gated_fusion_specialist_shapes_and_normalized_gates() -> None:
    model = _fusion_model((8, 12, 16, 20), stream_width=6, fusion_width=10)
    features = torch.randn(5, 56)

    mean, log_variance, gates = model(features, return_gates=True)

    assert mean.shape == (5,)
    assert log_variance.shape == (5,)
    assert gates.shape == (5, 4)
    assert torch.allclose(gates.sum(dim=1), torch.ones(5), atol=1e-6)
    assert torch.all(log_variance >= -6)
    assert torch.all(log_variance <= 4)


def test_gated_fusion_requires_multiple_streams() -> None:
    try:
        _fusion_model((8,))
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("single-stream fusion should be rejected")

