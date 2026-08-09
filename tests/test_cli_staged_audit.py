from pathlib import Path
import sys

import pytest

from hemfm import cli


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    passed: bool,
) -> tuple[int, dict[str, object]]:
    config = {"paths": {"evidence_root": str(tmp_path)}}
    called: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(cli, "ensure_run_directories", lambda _: None)

    def audit(
        received_config: dict,
        *,
        target: str,
        device: int,
    ) -> dict[str, object]:
        called.update(
            {"config": received_config, "target": target, "device": device}
        )
        return {"passed": passed, "promotion_eligible": False, "seeds": []}

    monkeypatch.setattr(cli, "run_staged_scalar_uncertainty_audit", audit)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hemfm",
            "staged-final",
            "audit",
            "--target",
            "LVESV",
            "--device",
            "1",
        ],
    )
    with pytest.raises(SystemExit) as raised:
        cli.main()
    return int(raised.value.code), called


def test_staged_audit_cli_routes_target_and_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    code, called = _run_cli(monkeypatch, tmp_path, passed=True)

    assert code == 0
    assert called["target"] == "LVESV"
    assert called["device"] == 1


def test_staged_audit_cli_fails_closed_on_incomplete_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    code, _ = _run_cli(monkeypatch, tmp_path, passed=False)

    assert code == 2
