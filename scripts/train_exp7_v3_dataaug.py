from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolov8s.pt")

    model.train(
        data="mask_v3.yaml",

        epochs=300,
        patience=50,

        imgsz=640,
        batch=8,

        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.2,

        translate=0.1,
        scale=0.5,
        fliplr=0.5,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        workers=4,

        project="runs/paper",
        name="exp7_v3_dataaug"
    )