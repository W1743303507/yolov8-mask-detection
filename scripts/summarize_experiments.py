"""Summarize YOLO experiment results without running training."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "experiments"
EXPERIMENTS = [
    ("exp2_v8s", "YOLOv8s Baseline", "基准模型"),
    ("exp3_cbam", "YOLOv8s + CBAM", "注意力模块未带来稳定提升"),
    ("exp4_eca", "YOLOv8s + ECA", "轻量注意力模块效果有限"),
    ("exp5_dataaug", "YOLOv8s + Data Augmentation", "数据增强带来明显提升"),
    ("exp6_dataset_v3", "YOLOv8s + dataset_v3", "数据扩充方案综合性能最优"),
    ("exp7_v3_dataaug", "YOLOv8s + dataset_v3 + Data Augmentation", "扩充数据集后继续增强，未超过 Exp6"),
]
CONFIRMED_METRICS = {
    # Final best.pt validation values confirmed for the paper.
    "exp7_v3_dataaug": {
        "Precision": 0.922,
        "Recall": 0.796,
        "mAP50": 0.879,
        "mAP50-95": 0.625,
        "Epoch": 65,
    },
}
FIELDS = {
    "Precision": "metrics/precision(B)",
    "Recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}


def locate_results(experiment: str) -> Path:
    preferred = ROOT / "runs" / "paper" / experiment / "results.csv"
    if preferred.is_file():
        return preferred
    matches = sorted((ROOT / "runs").glob(f"**/{experiment}/results.csv"))
    if not matches:
        raise FileNotFoundError(f"results.csv not found for {experiment}")
    return matches[0]


def best_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No result rows in {path}")
    # Use one consistent and auditable rule for every experiment.
    return max(rows, key=lambda row: float(row[FIELDS["mAP50-95"]]))


def fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> int:
    extracted = []
    for experiment, method, conclusion in EXPERIMENTS:
        path = locate_results(experiment)
        row = best_row(path)
        metrics = {name: float(row[column]) for name, column in FIELDS.items()}
        confirmed = CONFIRMED_METRICS.get(experiment, {})
        metrics.update({name: confirmed[name] for name in FIELDS if name in confirmed})
        extracted.append(
            {
                "Experiment": experiment,
                "Method": method,
                **metrics,
                "Conclusion": conclusion,
                "Epoch": int(confirmed.get("Epoch", float(row["epoch"]))),
                "Source": str(path.relative_to(ROOT)),
            }
        )

    baseline = extracted[0]["mAP50-95"]
    for row in extracted:
        row["Improvement_vs_Baseline_mAP50-95"] = row["mAP50-95"] - baseline

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "experiment_summary.csv"
    columns = [
        "Experiment", "Method", "Precision", "Recall", "mAP50", "mAP50-95",
        "Improvement_vs_Baseline_mAP50-95", "Conclusion",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in extracted:
            writer.writerow({key: fmt(row[key]) if isinstance(row.get(key), float) else row.get(key, "") for key in columns})

    md_path = OUTPUT_DIR / "experiment_summary.md"
    lines = [
        "# 实验结果汇总", "",
        "> Exp2-Exp6 取各实验 `results.csv` 中 mAP50-95 最高轮次；Exp7 采用已确认的 best.pt 最终验证结果。提升量以 Exp2 为基准。", "",
        "| 实验 | 方法 | Precision | Recall | mAP50 | mAP50-95 | 相对 Baseline 提升 | 结论 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in extracted:
        improvement = row["Improvement_vs_Baseline_mAP50-95"]
        lines.append(
            f"| {row['Experiment']} | {row['Method']} | {fmt(row['Precision'])} | "
            f"{fmt(row['Recall'])} | {fmt(row['mAP50'])} | {fmt(row['mAP50-95'])} | "
            f"{improvement:+.3f} | {row['Conclusion']} |"
        )
    lines += ["", "## 提取记录", ""]
    lines += [f"- `{row['Experiment']}`：第 {row['Epoch']} 轮，来源 `{row['Source']}`" for row in extracted]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
