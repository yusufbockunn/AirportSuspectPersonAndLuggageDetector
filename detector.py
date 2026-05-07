"""
detector.py
-----------
Nesne tespiti ve takip modülü.

Model & parametre seçimleri:
  • yolov8s.pt        → Small model: uzak/küçük nesnelerde çok daha iyi doğruluk
  • imgsz=960         → Yüksek çözünürlük: uzak/küçük nesneler için maksimum detay
  • conf=0.25         → Düşük eşik: arka plandaki uzak kişi ve valizleri de yakalar
  • iou=0.60          → Yüksek IOU: kalabalık/örtüşen kişileri ayrı ayrı tespit eder
  • tracker=botsort   → Stabil ID ataması, az flickering
  • classes=[0,24,26,28] → Sadece: person, backpack, handbag, suitcase

Not: Threaded pipeline sayesinde inference süresi UI'ı etkilemez.
"""

from ultralytics import YOLO

# Small model — Nano'ya göre daha iyi küçük nesne tespiti
model = YOLO("yolov8s.pt")

# Güvenlik uygulaması için ilgili COCO sınıf ID'leri
SECURITY_CLASSES = [0, 24, 26, 28]


def detect_and_track(frame):
    """
    Video frame'i üzerinde YOLOv8 track() çalıştırır.
    persist=True sayesinde ID'ler kareler arasında korunur.

    Dönüş:
        list[tuple]: (track_id, label, x1, y1, x2, y2)
    """
    results = model.track(
        frame,
        persist=True,
        verbose=False,
        imgsz=960,
        conf=0.25,
        iou=0.60,
        classes=SECURITY_CLASSES,
        tracker="botsort.yaml",
    )
    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls     = int(box.cls[0])
            label   = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            detections.append((track_id, label, x1, y1, x2, y2))

    return detections