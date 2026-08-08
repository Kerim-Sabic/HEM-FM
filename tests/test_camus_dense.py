from __future__ import annotations

import torch

from hemfm.camus_dense import DenseLVDecoder, DenseLVModel, _boundary_target, _reshape_tokens


def test_dense_lv_model_to_moves_both_components() -> None:
    encoder = torch.nn.Linear(2, 2)
    decoder = torch.nn.Linear(2, 2)
    model = DenseLVModel(encoder, decoder, frames=16)

    assert model.to("cpu") is model
    assert next(model.encoder.parameters()).device.type == "cpu"
    assert next(model.decoder.parameters()).device.type == "cpu"


def test_token_grid_and_dense_decoder_outputs() -> None:
    tokens = torch.randn(2, 8 * 4 * 4, 24)
    grid = _reshape_tokens(tokens, frames=16)
    decoder = DenseLVDecoder(in_channels=24, width=16, frames=16, resolution=32)

    outputs = decoder(grid)

    assert grid.shape == (2, 24, 8, 4, 4)
    assert outputs["segmentation"].shape == (2, 4, 16, 32, 32)
    assert outputs["boundary"].shape == (2, 1, 16, 32, 32)
    assert outputs["log_variance"].shape == (2, 1, 16, 32, 32)


def test_boundary_target_marks_mask_edges() -> None:
    mask = torch.zeros(1, 2, 8, 8, dtype=torch.long)
    mask[:, :, 2:6, 2:6] = 1

    boundary = _boundary_target(mask)

    assert boundary.shape == (1, 1, 2, 8, 8)
    assert boundary.sum() > 0
    assert boundary[:, :, :, 3:5, 3:5].sum() == 0

