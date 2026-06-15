"""Project integrity checks. This script never starts training."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NAMES = {
    0: "with_mask",
    1: "without_mask",
    2: "mask_weared_incorrect",
}
CORE_DIRS = ["configs", "custom_modules", "scripts", "experiments", "dataset", "dataset_v3", "runs"]
CORE_FILES = [
    "configs/yolov8s_cbam.yaml",
    "configs/yolov8s_eca.yaml",
    "custom_modules/cbam.py",
    "custom_modules/eca.py",
    "custom_modules/__init__.py",
    "scripts/train_exp2_v8s.py",
    "scripts/train_exp3_cbam.py",
    "scripts/train_exp4_eca.py",
    "scripts/train_exp5_dataaug.py",
    "scripts/train_exp6_dataset_v3.py",
    "experiments/experiment_record.xlsx",
    "mask_original.yaml",
    "mask_v3.yaml",
]


def report(ok: bool, label: str, detail: str = "") -> bool:
    suffix = f" - {detail}" if detail else ""
    print(f"[{'PASS' if ok else 'FAIL'}] {label}{suffix}")
    return ok


def main() -> int:
    print("=== Mask Detection Project Integrity Check ===")
    print(f"Python executable: {sys.executable}")
    print(f"Project root: {ROOT}")
    passed = []

    try:
        import torch

        print(f"Torch version: {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"CUDA available: {cuda_ok}")
        print(f"GPU name: {torch.cuda.get_device_name(0) if cuda_ok else 'N/A'}")
        passed.append(report(True, "Import torch"))
    except Exception as exc:
        passed.append(report(False, "Import torch", repr(exc)))

    for directory in CORE_DIRS:
        passed.append(report((ROOT / directory).is_dir(), f"Directory: {directory}"))
    for filename in CORE_FILES:
        passed.append(report((ROOT / filename).is_file(), f"File: {filename}"))

    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from custom_modules.cbam import CBAM
        from custom_modules.eca import ECA

        passed.append(report(True, "Import CBAM and ECA"))
        import ultralytics.nn.tasks as tasks

        tasks.CBAM = CBAM
        tasks.ECA = ECA
        registered = tasks.CBAM is CBAM and tasks.ECA is ECA
        passed.append(report(registered, "Dynamic registration in ultralytics.nn.tasks"))
    except Exception as exc:
        passed.append(report(False, "Custom module import/registration", repr(exc)))

    cbam_yaml = (ROOT / "configs/yolov8s_cbam.yaml").read_text(encoding="utf-8")
    eca_yaml = (ROOT / "configs/yolov8s_eca.yaml").read_text(encoding="utf-8")
    passed.append(report("CBAM" in cbam_yaml, "CBAM referenced by model YAML"))
    passed.append(report("ECA" in eca_yaml, "ECA referenced by model YAML"))

    try:
        import yaml

        original_data = yaml.safe_load((ROOT / "mask_original.yaml").read_text(encoding="utf-8"))
        v3_data = yaml.safe_load((ROOT / "mask_v3.yaml").read_text(encoding="utf-8"))
        original_names = {int(k): v for k, v in original_data.get("names", {}).items()}
        v3_names = {int(k): v for k, v in v3_data.get("names", {}).items()}
        passed.append(
            report(original_names == EXPECTED_NAMES, "Class order in mask_original.yaml", str(original_names))
        )
        passed.append(report(v3_names == EXPECTED_NAMES, "Class order in mask_v3.yaml", str(v3_names)))
        passed.append(
            report(original_names == v3_names, "Class order is consistent across dataset YAML files")
        )
        passed.append(
            report(
                original_data.get("path") == "dataset",
                "mask_original.yaml points to dataset",
                str(original_data.get("path")),
            )
        )
        passed.append(
            report(
                v3_data.get("path") == "dataset_v3",
                "mask_v3.yaml points to dataset_v3",
                str(v3_data.get("path")),
            )
        )
    except Exception as exc:
        passed.append(report(False, "Parse dataset YAML files", repr(exc)))

    print(f"\nSummary: {sum(passed)}/{len(passed)} checks passed")
    print("No training was started.")
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
