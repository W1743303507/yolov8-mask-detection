import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ultralytics import YOLO
from custom_modules.cbam import CBAM

import ultralytics.nn.tasks as tasks

tasks.CBAM = CBAM


if __name__ == "__main__":

    model = YOLO("configs/yolov8s_cbam.yaml")

    model.train(
        data="mask_original.yaml",

        epochs=300,
        patience=50,

        imgsz=640,
        batch=8,

        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        mosaic=1.0,

        workers=4,

        project="runs/paper",
        name="exp3_cbam"
    )
