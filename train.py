from ultralytics import YOLO

model = YOLO("yolov8s-seg.pt")

model.train(
    data="foodseg.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=2,
    patience=25
)