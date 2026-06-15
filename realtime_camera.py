"""Local webcam inference for the mask detection model."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WEIGHT_CANDIDATES = [
    ROOT / "runs/paper/exp6_dataset_v3/weights/best.pt",
    ROOT / "runs/detect/runs/paper/exp6_dataset_v3/weights/best.pt",
]


def find_weight() -> Path | None:
    return next((path for path in WEIGHT_CANDIDATES if path.is_file()), None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 本地摄像头口罩检测")
    parser.add_argument(
        "--camera",
        type=int,
        default=-1,
        help="摄像头索引，默认 -1 表示自动尝试 0-4。",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "0"],
        help="推理设备，默认 cpu。",
    )
    return parser.parse_args()


def open_camera(cv2, requested_index: int):
    indices = [requested_index] if requested_index >= 0 else list(range(5))
    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("默认后端", cv2.CAP_ANY),
    ]

    print("正在查找可用摄像头...")
    for index in indices:
        for backend_name, backend in backends:
            capture = cv2.VideoCapture(index, backend)
            if not capture.isOpened():
                print(f"  失败：索引 {index}，后端 {backend_name}，无法打开")
                capture.release()
                continue

            frame_ok = False
            for _ in range(10):
                ok, frame = capture.read()
                if ok and frame is not None and frame.size > 0:
                    frame_ok = True
                    break

            if frame_ok:
                print(f"  成功：索引 {index}，后端 {backend_name}")
                return capture, index, backend_name

            print(f"  失败：索引 {index}，后端 {backend_name}，已打开但无法读取画面")
            capture.release()

    return None, None, None


def print_camera_help() -> None:
    print("\n错误：未找到可用的本地摄像头。请依次检查：")
    print("1. 关闭 Windows 相机、微信、QQ、Teams、浏览器会议页面等可能占用摄像头的程序。")
    print("2. 打开 Windows 设置 -> 隐私 -> 相机，启用相机访问和允许桌面应用访问相机。")
    print("3. 在设备管理器中确认摄像头已启用且驱动正常。")
    print("4. 如果使用外接摄像头，请重新插拔并更换 USB 接口。")
    print("5. 可指定索引重试，例如：python realtime_camera.py --camera 1")


def main() -> int:
    args = parse_args()
    weight_path = find_weight()
    if weight_path is None:
        print("错误：未找到模型权重，请确认 best.pt 是否存在。")
        print("已检查以下路径：")
        for candidate in WEIGHT_CANDIDATES:
            print(f"  {candidate}")
        return 1

    try:
        import cv2
        from ultralytics import YOLO
    except Exception as exc:
        print(f"错误：摄像头检测依赖导入失败：{exc}")
        return 1

    print(f"OpenCV 版本：{cv2.__version__}")
    capture, camera_index, backend_name = open_camera(cv2, args.camera)
    if capture is None:
        print_camera_help()
        return 1

    try:
        print(f"正在加载模型：{weight_path}")
        model = YOLO(str(weight_path))
    except Exception as exc:
        print(f"错误：模型加载失败：{exc}")
        capture.release()
        return 1

    print(
        f"摄像头检测已启动：索引 {camera_index}，后端 {backend_name}，"
        f"推理设备 {args.device}。按 q 键退出。"
    )
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("错误：无法读取摄像头画面。")
                break

            try:
                results = model.predict(
                    source=frame,
                    conf=0.25,
                    imgsz=640,
                    device=args.device,
                    verbose=False,
                )
                display_frame = results[0].plot() if results else frame
            except Exception as exc:
                print(f"错误：摄像头推理失败：{exc}")
                display_frame = frame

            cv2.imshow("YOLOv8 Mask Detection - Press q to quit", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print("摄像头检测已退出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
