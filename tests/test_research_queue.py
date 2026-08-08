from pathlib import Path

from scripts.run_research_training_queue import _read, _run_wave, _write


def test_queue_status_round_trip(tmp_path: Path):
    path = tmp_path / "status.json"
    _write(path, {"phase": "running", "locked_test_accessed": False})
    assert _read(path) == {"phase": "running", "locked_test_accessed": False}


def test_run_wave_clears_stale_failure_before_retry(tmp_path: Path, monkeypatch):
    repository = tmp_path / "repository"
    logs = tmp_path / "logs"
    repository.mkdir()
    logs.mkdir()
    status = logs / "job.json"
    queue_status = tmp_path / "queue.json"
    _write(status, {"phase": "failed", "returncode": 1})

    observed = {}

    class CompletedProcess:
        def poll(self):
            return 0

    def fake_popen(*args, **kwargs):
        observed.update(_read(status))
        _write(status, {"phase": "complete", "returncode": 0})
        return CompletedProcess()

    monkeypatch.setattr("scripts.run_research_training_queue.subprocess.Popen", fake_popen)
    _run_wave(repository, logs, queue_status, "retry_wave", [("job", ["example"])])

    assert observed["phase"] == "queued_for_retry"
    assert _read(queue_status)["phase"] == "retry_wave_complete"

