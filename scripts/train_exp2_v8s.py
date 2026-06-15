from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolov8s.pt")

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
        name="exp2_v8s"
    )
