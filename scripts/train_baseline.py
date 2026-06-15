from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolov8n.pt")

    model.train(
        data="mask.yaml",

        epochs=100,
        batch=8,
        imgsz=640,

        workers=4,

        seed=42,

        patience=30,

        mosaic=0.0,

        project="runs/baseline",

        name="exp1_baseline",

        pretrained=True
    )