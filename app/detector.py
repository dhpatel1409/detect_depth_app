from ultralytics import YOLO

_yolo = None

def load_yolo():
    global _yolo
    if _yolo is None:
        _yolo = YOLO('yolov8n.pt')
    return _yolo

def detect_objects(raw_img):
    yolo = load_yolo()
    results = yolo(raw_img, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = yolo.names[cls_id]
        detections.append({
            "class": cls_name,
            "confidence": conf,
            "bbox": [float(x1), float(y1), float(x2), float(y2)]
        })
    return detections