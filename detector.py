"""
detector.py
-----------
Nesne tespiti ve takip modülü.

Performans düzeltmeleri:
  • imgsz=640  → 960'tan düşürüldü, hız ~2x artar, doğruluk kabul edilebilir
  • Frame skip  → Her frame değil, her N frame'de bir inference yapar
                  Aradaki frame'lerde son sonucu tekrar kullanır
                  Bu sayede UI akıcı kalır, YOLO CPU/GPU'yu boğmaz
"""

from ultralytics import YOLO

model = YOLO("yolov8s.pt")

SECURITY_CLASSES = [0, 24, 26, 28]

# Kaç frame'de bir inference yapılsın?
# 1 = her frame (yavaş), 2 = her 2 frame'de bir (dengeli), 3 = her 3'te bir (hızlı)
INFERENCE_EVERY_N = 2

_frame_counter = 0
_last_detections: list = []


def detect_and_track(frame):
    """
    Video frame'i üzerinde YOLOv8 track() çalıştırır.
    INFERENCE_EVERY_N frame'de bir gerçek inference yapar,
    aradaki frame'lerde son sonucu döner → UI akıcı kalır.
    """
    global _frame_counter, _last_detections

    _frame_counter += 1

    if _frame_counter % INFERENCE_EVERY_N != 0:
        # Bu frame'i atla, son sonucu kullan
        return _last_detections

    results = model.track(
        frame,
        persist=True,
        verbose=False,
        imgsz=640,        # 960 → 640: ~2x hız kazanımı
        conf=0.25,
        iou=0.60,
        classes=SECURITY_CLASSES,
        tracker="botsort.yaml",
    )

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls      = int(box.cls[0])
            label    = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            detections.append((track_id, label, x1, y1, x2, y2))

    _last_detections = detections
    return detections


def reset_counter():
    """Video yeniden başladığında sayacı sıfırla."""
    global _frame_counter, _last_detections
    _frame_counter = 0
    _last_detections = []