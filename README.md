# 基于 YOLOv8 的口罩佩戴检测系统

## 1. 项目简介

本项目基于 YOLOv8s 实现三分类口罩佩戴检测，用于识别正确佩戴口罩、未佩戴口罩和错误佩戴口罩三种状态。项目重点比较注意力模块改进与数据质量改进的实际效果。

## 2. 环境配置

- 操作系统：Windows 10
- GPU：NVIDIA RTX 3060 Laptop GPU 6GB
- CUDA：12.1
- Python：3.12.3
- PyTorch：2.5.1+cu121
- Ultralytics：8.4.48
- 虚拟环境：`D:\mask-detection\yolov8-env`

项目虚拟环境解释器的绝对路径为：

```text
D:\mask-detection\yolov8-env\Scripts\python.exe
```

如果命令行虽然显示 `(yolov8-env)`，但 `where python` 或 `sys.executable` 仍指向全局 Python，说明环境激活不稳定。此时不要依赖 `activate`，应直接使用上述项目虚拟环境解释器的绝对路径运行脚本。

## 3. 数据集说明

项目保留 `dataset`、`dataset_v2` 和 `dataset_v3`。Exp2 至 Exp5 的原始训练均基于原始数据集 `dataset`，Exp6 和 Exp7 基于扩充后的 `dataset_v3`，其中 Exp7 进一步加入数据增强。为提高后续复现稳定性，当前将数据配置拆分为 `mask_original.yaml` 和 `mask_v3.yaml`：前者固定指向 `dataset`，后者固定指向 `dataset_v3`。两个配置文件使用相同的训练集、验证集相对目录和类别顺序。

## 4. 类别定义

类别编号和顺序不可更改：

```text
0: with_mask
1: without_mask
2: mask_weared_incorrect
```

## 5. 项目目录结构

```text
mask-detection/
├── configs/                 # CBAM、ECA 模型结构配置
├── custom_modules/          # 自定义注意力模块
├── dataset*/                # 各版本数据集
├── docs/                    # 中文说明与实验分析
├── experiments/             # 实验记录及汇总结果
├── runs/                    # Ultralytics 训练输出
├── scripts/                 # 训练、检查和汇总脚本
├── app.py                   # Streamlit 演示页面
├── realtime_camera.py       # 本地摄像头实时检测
├── run_app.bat              # 网页一键启动脚本
├── run_camera.bat           # 摄像头一键启动脚本
├── mask_original.yaml       # Exp2-Exp5 原始数据集配置
└── mask_v3.yaml             # Exp6 扩充数据集配置
```

## 6. 模型方案

基础方案采用 YOLOv8s。结构改进实验分别在检测头特征层后加入 CBAM 和 ECA，并通过运行时注册到 `ultralytics.nn.tasks`，不修改 Ultralytics 官方源码。数据改进实验采用增强策略和 dataset_v3 数据扩充。

## 7. 实验设计

实验包括 Exp2 Baseline、Exp3 CBAM、Exp4 ECA、Exp5 数据增强、Exp6 dataset_v3 和 Exp7 dataset_v3 + 数据增强。Exp2 至 Exp5 使用原始数据集 `dataset`，Exp6 和 Exp7 使用扩充后的 `dataset_v3`。主要训练参数为 300 epochs、640 输入尺寸、batch 8、SGD 优化器。Exp7 在 dataset_v3 基础上使用 `mixup=0.15` 和 `copy_paste=0.2`。

## 8. 实验结果

| 实验 | 方法 | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| Exp2 | YOLOv8s Baseline | 0.886 | 0.812 | 0.864 | 0.605 |
| Exp3 | YOLOv8s + CBAM | 0.906 | 0.779 | 0.837 | 0.569 |
| Exp4 | YOLOv8s + ECA | 0.905 | 0.740 | 0.817 | 0.562 |
| Exp5 | YOLOv8s + 数据增强 | 0.962 | 0.770 | 0.873 | 0.614 |
| Exp6 | YOLOv8s + dataset_v3 | 0.915 | 0.817 | 0.896 | 0.630 |
| Exp7 | YOLOv8s + dataset_v3 + 数据增强 | 0.922 | 0.796 | 0.879 | 0.625 |

注：Exp2 至 Exp6 采用各实验 `results.csv` 中 mAP50-95 最高轮次的数据；Exp7 采用已确认的 best.pt 最终验证结果，与自动汇总文件保持一致。

## 9. 最优方案

最终方案仍为 **YOLOv8s + dataset_v3（Exp6）**。Exp7 的 Precision 为 0.922，略高于 Exp6 的 0.915，但 Recall、mAP50 和 mAP50-95 均低于 Exp6。因此，综合检测能力更强的 Exp6 仍作为最终模型，Streamlit 页面也继续默认加载 Exp6 的 `best.pt`。

## 10. 如何运行训练

激活虚拟环境后，根据实验执行对应脚本：

```powershell
python scripts\train_exp6_dataset_v3.py
```

Exp2 至 Exp5 的训练脚本固定使用 `mask_original.yaml`，Exp6 固定使用 `mask_v3.yaml`，无需通过修改同一个 YAML 切换数据集。CBAM/ECA 脚本已在加载模型前执行动态注册。训练属于耗时操作，应在确认实验配置后手动启动。

## 11. 如何运行演示系统

演示系统支持以下功能：

- 单张图片检测
- 批量图片检测与 CSV 汇总下载
- 上传视频抽帧检测
- 本地摄像头实时检测

首次使用前安装页面依赖：

```powershell
python -m pip install -r requirements_app.txt
```

### 11.1 网页命令行启动

```powershell
D:\mask-detection\yolov8-env\Scripts\python.exe -m streamlit run D:\mask-detection\app.py --server.address 127.0.0.1 --server.port 8502 --server.headless true
```

浏览器手动打开：

```text
http://127.0.0.1:8502
```

### 11.2 网页一键启动

```text
双击 run_app.bat
```

### 11.3 摄像头实时检测

命令行启动：

```powershell
D:\mask-detection\yolov8-env\Scripts\python.exe D:\mask-detection\realtime_camera.py
```

或在项目根目录一键启动：

```text
双击 run_camera.bat
```

摄像头窗口中按 `q` 键退出。摄像头脚本默认使用 CPU，不放入 Streamlit 页面，以减少浏览器权限和 WebRTC 环境对演示稳定性的影响。

页面默认优先加载 `runs/paper/exp6_dataset_v3/weights/best.pt`，并兼容 `runs/detect/runs/paper/exp6_dataset_v3/weights/best.pt`。推理设备默认选择 CPU；确认 CUDA 工作正常后，可在侧边栏切换为 GPU `0`。

## 12. 如何运行完整性检查

```powershell
python scripts\check_project_integrity.py
python scripts\summarize_experiments.py
```

前者只执行环境、文件、类别和模块注册检查，不会启动训练；后者生成 CSV 和 Markdown 实验汇总。

## 13. 注意事项

- 不得修改 Ultralytics 官方源码或 `site-packages`。
- 不得改变类别编号与顺序。
- 不得删除历史数据集和 `runs` 结果。
- CBAM、ECA 必须继续使用动态注册。
- 原始数据集与扩充数据集已分别使用 `mask_original.yaml` 和 `mask_v3.yaml`，复现时不得互换。
- 如果提示符显示 `(yolov8-env)`，但 `where python` 仍指向全局 Python，说明环境激活不稳定；应使用 `D:\mask-detection\yolov8-env\Scripts\python.exe` 的绝对路径运行项目脚本。
- 如果 GPU 推理卡住，请在网页侧边栏选择 CPU。
- 如果 Streamlit 自动打开空白页，请手动访问 `http://127.0.0.1:8502`。
- 模型权重必须位于页面支持的两个 `best.pt` 路径之一。
- 视频检测可能较慢，建议先使用短视频和较大的抽帧间隔测试。
