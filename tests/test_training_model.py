import torch

from hemfm.train_cached import _make_model


def test_attentive_two_view_output_contract():
    model = _make_model(structured_dim=12)
    tokens = torch.randn(2, 2, 32, 1024)
    structured = torch.randn(2, 12)
    mean, log_variance = model(tokens, structured)
    assert mean.shape == (2, 3)
    assert log_variance.shape == (2, 3)
    assert torch.isfinite(mean).all()

