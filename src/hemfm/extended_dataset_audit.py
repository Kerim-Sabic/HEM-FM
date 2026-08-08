from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

from .frozen_specialists import _atomic_json


SOURCES = {
    "ev9v": "https://huggingface.co/datasets/bgx666/EV9V",
    "cardiacuda": "https://www.kaggle.com/datasets/xiaoweixumedicalai/cardiacudc-dataset",
    "hmc_qu": "https://www.kaggle.com/datasets/aysendegerli/hmcqu-dataset",
    "echocp": "https://www.kaggle.com/datasets/xiaoweixumedicalai/echocp",
    "cardiacnet": "https://www.kaggle.com/datasets/xiaoweixumedicalai/abnormcardiacechovideos",
    "echoxflow": "https://huggingface.co/datasets/Ahus-AIM/EchoXFlow",
    "echonet_dynamic": "https://echonet.github.io/dynamic/",
    "echonet_lvh": "https://echonet.github.io/lvh/",
    "echonet_pediatric": "https://echonet.github.io/pediatric/",
    "echoview": "https://physionet.org/content/echoview/0.1/",
    "echorisk": "https://echorisk-miccai.github.io/",
}


def _status(run_root: Path, suffix: str) -> dict[str, Any]:
    path = run_root / "week_training" / f"dataset_download_status_{suffix}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _exam_key(name: str) -> str:
    output = re.sub(r"_\d{3}_\d+x\d+_\d+$", "", name)
    return re.sub(r"_\d+x\d+_\d+$", "", output)


def _ev9v(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["paths"]["ev9v_root"])
    manifest = root / "development_manifest.csv"
    import pandas as pd

    frame = pd.read_csv(manifest)
    train = frame.loc[frame["split"] == "train"]
    validation = frame.loc[frame["split"] == "validation"]
    overlap = {_exam_key(value) for value in train["video_name"]} & {
        _exam_key(value) for value in validation["video_name"]
    }
    status = _status(Path(config["paths"]["run_root"]), "ev9v")
    checks = {
        "development_video_count": len(frame) == 4250,
        "official_split_counts": len(train) == 3683 and len(validation) == 567,
        "exam_proxy_disjoint": not overlap,
        "nine_views_present": bool(frame["view"].nunique() == 9),
        "all_selected_videos_local": bool(
            frame["local_path"].map(lambda value: Path(value).exists()).all()
        ),
        "test_labels_not_downloaded": not (root / "test_labeled.txt").exists(),
        "source_revision_and_hashes_recorded": bool(status.get("source_revision")) and len(status.get("files", [])) == 4,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "source": SOURCES["ev9v"],
        "licence": "CC BY 4.0",
        "role": "nine-view routing and acquisition-quality control",
        "train_videos": len(train),
        "validation_videos": len(validation),
        "exam_proxy_counts": {
            "train": len({_exam_key(value) for value in train["video_name"]}),
            "validation": len({_exam_key(value) for value in validation["video_name"]}),
        },
        "views": dict(sorted(Counter(frame["view"]).items())),
        "promotion": "May improve the routing/view-QC head only after held-out gain; never supplies scalar measurement labels.",
        "locked_test_accessed": False,
    }


def _cardiacuda(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["paths"]["cardiacuda_root"])
    images = [path for path in root.rglob("*_image.nii.gz") if path.is_file()]
    labels = [path for path in root.rglob("*_label.nii.gz") if path.is_file()]
    paired = [path for path in images if Path(str(path).replace("_image.nii.gz", "_label.nii.gz")).exists()]
    folders = Counter(path.parent.name for path in images)
    status = _status(Path(config["paths"]["run_root"]), "cardiacuda")
    checks = {
        "archive_extracted": status.get("phase") == "extracted_complete",
        "a4c_images_present": len(images) == 364,
        "labelled_a4c_pairs_present": len(paired) == 294,
        "unlabelled_target_partition_present": len(images) - len(paired) == 70,
        "six_site_partitions_plus_dense_reference": len(folders) == 7,
        "archive_and_tree_hashes_recorded": len(status.get("tree_manifest_sha256", "")) == 64 and len(status.get("extracted_tree_manifest_sha256", "")) == 64,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "source": SOURCES["cardiacuda"],
        "licence": "Apache 2.0",
        "role": "Philips/Hitachi multi-centre A4C chamber segmentation and domain stress",
        "available_a4c_videos": len(images),
        "labelled_pairs": len(paired),
        "unlabelled_videos": len(images) - len(paired),
        "folder_counts": dict(sorted(folders.items())),
        "scope_note": "The Kaggle package exposes the A4C subset, not all 992 four-view videos described by the paper.",
        "promotion": "Research challenger only; unclear public partition semantics prohibit using it as a locked final test.",
        "locked_test_accessed": False,
    }


def _xlsx_values(path: Path) -> list[list[Any]]:
    namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", namespace):
                shared.append("".join(node.text or "" for node in item.findall(".//m:t", namespace)))
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows: list[list[Any]] = []
    for row in sheet.findall(".//m:sheetData/m:row", namespace):
        values: dict[int, Any] = {}
        for cell in row.findall("m:c", namespace):
            reference = cell.attrib.get("r", "A1")
            letters = re.match(r"[A-Z]+", reference).group(0)
            column = 0
            for letter in letters:
                column = column * 26 + (ord(letter) - 64)
            value_node = cell.find("m:v", namespace)
            if value_node is None:
                value: Any = None
            elif cell.attrib.get("t") == "s":
                value = shared[int(value_node.text)]
            else:
                raw = value_node.text or ""
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
            values[column - 1] = value
        width = max(values, default=-1) + 1
        rows.append([values.get(index) for index in range(width)])
    return rows


def _hmc_workbook(path: Path, mask_column: int | None) -> dict[str, Any]:
    # The A2C workbook ends with a human-readable footnote in column A.  Require
    # the two numeric cycle columns so it cannot be counted as a clinical row.
    rows = [
        row
        for row in _xlsx_values(path)[2:]
        if len(row) >= 9
        and row[0]
        and isinstance(row[7], (int, float))
        and isinstance(row[8], (int, float))
    ]
    consensus = Counter()
    cycle_frames: list[int] = []
    masks = 0
    for row in rows:
        votes = list(row[1:7])
        mi = sum(value == "MI" for value in votes)
        consensus["MI" if mi > 3 else "non-MI" if mi < 3 else "tie"] += 1
        cycle_frames.append(int(float(row[8]) - float(row[7]) + 1))
        if mask_column is not None and len(row) > mask_column and row[mask_column] == "ü":
            masks += 1
    return {
        "records": len(rows),
        "consensus": dict(consensus),
        "cycle_frames": {"minimum": min(cycle_frames), "maximum": max(cycle_frames)},
        "masks_declared": masks,
    }


def _hmc_qu(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["paths"]["hmc_qu_root"])
    a2c = _hmc_workbook(root / "A2C.xlsx", None)
    a4c = _hmc_workbook(root / "A4C.xlsx", 9)
    masks = list((root / "LV Ground-truth Segmentation Masks").glob("*.mat"))
    videos = [path for path in root.rglob("*") if path.suffix.lower() in {".avi", ".mp4", ".dcm"}]
    status = _status(Path(config["paths"]["run_root"]), "hmc_qu")
    checks = {
        "metadata_counts": a2c["records"] == 160 and a4c["records"] == 162,
        "mask_count": len(masks) == 109 and a4c["masks_declared"] == 109,
        "raw_videos_absent_from_package": not videos,
        "download_hash_recorded": len(status.get("tree_manifest_sha256", "")) == 64,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "source": SOURCES["hmc_qu"],
        "licence": "CC BY-NC-SA 3.0 IGO",
        "role": "metadata and mask-shape audit only until the authors provide the raw A2C/A4C cines",
        "A2C": a2c,
        "A4C": a4c,
        "raw_video_files": len(videos),
        "promotion": "Excluded from image training because the Kaggle package contains no source cines.",
        "locked_test_accessed": False,
    }


def _echocp(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["paths"]["echocp_root"])
    images = list(root.glob("*_image.nii.gz"))
    labels = list(root.glob("*_label.nii.gz"))
    patients = {path.name.split("_", 1)[0] for path in images}
    status = _status(Path(config["paths"]["run_root"]), "echocp")
    checks = {
        "thirty_patients": len(patients) == 30,
        "rest_and_valsalva_pairs": len(images) == 60 and len(labels) == 60,
        "archive_extracted_and_hashed": status.get("phase") == "extracted_complete" and len(status.get("extracted_tree_manifest_sha256", "")) == 64,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "source": SOURCES["echocp"],
        "licence": "Apache 2.0",
        "role": "contrast-bubble OOD and abstention/failure detection",
        "patients": len(patients),
        "videos": len(images),
        "promotion": "Never used to optimize the adult scalar heads; small disease-specific cohort is OOD-only.",
        "locked_test_accessed": False,
    }


def _cardiacnet(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["paths"]["cardiacnet_root"])
    # The archive contains container directories whose names also end in
    # ``_image.nii``.  Count only real NIfTI files, not those directories.
    images = [path for path in root.rglob("*_image.nii") if path.is_file()]
    labels = [path for path in root.rglob("*_label.nii") if path.is_file()]
    by_task = Counter("ASD" if "CardiacNet-ASD" in str(path) else "PAH" for path in images)
    by_group = Counter(
        next(
            group
            for group in ("Non-ASD", "ASD", "Non-PAH", "PAH")
            if group in path.parts
        )
        for path in images
    )
    status = _status(Path(config["paths"]["run_root"]), "cardiacnet")
    checks = {
        "images_and_labels_present": len(images) > 0 and len(labels) > 0,
        "asd_and_pah_present": set(by_task) == {"ASD", "PAH"},
        "tree_hash_recorded": len(status.get("tree_manifest_sha256", "")) == 64,
        "download_complete": status.get("phase") == "complete",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "source": SOURCES["cardiacnet"],
        "licence": "Public research release; verify dataset terms before any redistribution",
        "role": "PAH/ASD disease-specific OOD and failure detection",
        "image_files": len(images),
        "label_files": len(labels),
        "images_by_task": dict(sorted(by_task.items())),
        "images_by_group": dict(sorted(by_group.items())),
        "bytes": status.get("downloaded_bytes"),
        "promotion": "OOD-only because disease labels, nested controls, and patient grouping do not match core scalar endpoints.",
        "locked_test_accessed": False,
    }


def _echoxflow_demo(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["paths"]["echoxflow_demo_root"])
    status = _status(Path(config["paths"]["run_root"]), "echoxflow_demo")
    exams = [path for path in (root / "exams").glob("exam_*") if path.is_dir()]
    recordings = [
        path for path in (root / "exams").rglob("recording_*.zarr") if path.is_dir()
    ]
    complete_groups = [
        path for path in recordings if (path / ".zgroup").is_file() and (path / ".zattrs").is_file()
    ]
    checks = {
        "download_complete_and_hashed": status.get("phase") == "complete"
        and len(status.get("tree_manifest_sha256", "")) == 64,
        "single_demo_exam": len(exams) == 1,
        "seventy_nine_recording_groups": len(recordings) == 79,
        "recording_root_metadata_complete": len(complete_groups) == len(recordings),
        "croissant_metadata_present": (root / "croissant.json").is_file(),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "source": "https://huggingface.co/datasets/Ahus-AIM/EchoXFlow-Demo",
        "full_source": SOURCES["echoxflow"],
        "licence": "CC BY-NC-SA 4.0",
        "exam_count": len(exams),
        "recording_groups": len(recordings),
        "downloaded_bytes": status.get("downloaded_bytes"),
        "role": "native beamspace, Doppler, ECG, Zarr, and Croissant format smoke testing only",
        "decision": "The demo has one examination, so it is excluded from performance training and generalization claims. The full 345.5 GB repository cannot fit beside the protected corpus and current checkpoints within the present local disk budget.",
        "locked_test_accessed": False,
    }
def audit_extended_research_datasets(config: dict[str, Any]) -> dict[str, Any]:
    datasets = {
        "EV9V": _ev9v(config),
        "CardiacUDA": _cardiacuda(config),
        "HMC-QU": _hmc_qu(config),
        "EchoCP": _echocp(config),
        "CardiacNet": _cardiacnet(config),
    }
    echoxflow = _echoxflow_demo(config)
    report = {
        "schema_version": 1,
        "passed": all(item["passed"] for item in datasets.values()) and echoxflow["passed"],
        "datasets": datasets,
        "EchoXFlow": echoxflow,
        "controlled_access": {
            "EchoNet-Dynamic": {"source": SOURCES["echonet_dynamic"], "status": "requires individual Stanford registration and research-use agreement", "role": "high-priority A4C EF/EDV/ESV challenger"},
            "EchoNet-LVH": {"source": SOURCES["echonet_lvh"], "status": "requires individual Stanford registration and non-commercial research-use agreement", "role": "high-priority PLAX IVS/LVID/LVPW challenger"},
            "EchoNet-Pediatric": {"source": SOURCES["echonet_pediatric"], "status": "requires individual Stanford registration and non-commercial research-use agreement", "role": "pediatric age/size-shift validation and pediatric LV function challenger; never mixed into the adult deployment head without a separate analysis plan"},
            "ECHOVIEW": {"source": SOURCES["echoview"], "status": "requires PhysioNet credentialed access and DUA; not found in the currently inventoried local staging area", "role": "MIMIC router calibration/audit labels only because the 23-class annotations are machine-generated and clinician review covered a small sample"},
            "EchoRisk": {"source": SOURCES["echorisk"], "status": "requires Synapse registration/team access; hidden challenge test remains sealed", "role": "high-priority multicentre temporal external validation"},
        },
        "selection_policy": "No research dataset is fused into the final model unless a patient/exam-disjoint development comparison improves its intended endpoint without degrading calibrated core endpoints or failure detection.",
        "locked_test_accessed": False,
    }
    _atomic_json(Path(config["paths"]["evidence_root"]) / "G4" / "extended_research_datasets.json", report)
    return report

