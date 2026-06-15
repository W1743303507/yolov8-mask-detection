# 实验结果汇总

> Exp2-Exp6 取各实验 `results.csv` 中 mAP50-95 最高轮次；Exp7 采用已确认的 best.pt 最终验证结果。提升量以 Exp2 为基准。

| 实验 | 方法 | Precision | Recall | mAP50 | mAP50-95 | 相对 Baseline 提升 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| exp2_v8s | YOLOv8s Baseline | 0.886 | 0.812 | 0.864 | 0.605 | +0.000 | 基准模型 |
| exp3_cbam | YOLOv8s + CBAM | 0.906 | 0.779 | 0.837 | 0.569 | -0.036 | 注意力模块未带来稳定提升 |
| exp4_eca | YOLOv8s + ECA | 0.905 | 0.740 | 0.817 | 0.562 | -0.044 | 轻量注意力模块效果有限 |
| exp5_dataaug | YOLOv8s + Data Augmentation | 0.962 | 0.770 | 0.873 | 0.614 | +0.009 | 数据增强带来明显提升 |
| exp6_dataset_v3 | YOLOv8s + dataset_v3 | 0.915 | 0.817 | 0.896 | 0.630 | +0.025 | 数据扩充方案综合性能最优 |
| exp7_v3_dataaug | YOLOv8s + dataset_v3 + Data Augmentation | 0.922 | 0.796 | 0.879 | 0.625 | +0.020 | 扩充数据集后继续增强，未超过 Exp6 |

## 提取记录

- `exp2_v8s`：第 37 轮，来源 `runs\detect\runs\paper\exp2_v8s\results.csv`
- `exp3_cbam`：第 119 轮，来源 `runs\detect\runs\paper\exp3_cbam\results.csv`
- `exp4_eca`：第 137 轮，来源 `runs\detect\runs\paper\exp4_eca\results.csv`
- `exp5_dataaug`：第 112 轮，来源 `runs\detect\runs\paper\exp5_dataaug\results.csv`
- `exp6_dataset_v3`：第 96 轮，来源 `runs\detect\runs\paper\exp6_dataset_v3\results.csv`
- `exp7_v3_dataaug`：第 65 轮，来源 `runs\detect\runs\paper\exp7_v3_dataaug\results.csv`
