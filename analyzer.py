"""
analyzer.py
-----------
Tespit edilen nesneleri analiz ederek güvenlik uyarıları üretir.

Düzeltmeler:
  • Şüpheli kişi tespiti artık piksel genişliğine değil,
    video zamanına (ekranda kalma süresi) dayalı.
    Eski yöntem slowmo'da yanlış alarm üretiyordu çünkü
    yavaş frame'lerde kişi daha uzun süre "büyük" görünüyordu.
  • SUSPICIOUS_MIN_SECONDS: Bir kişi bu kadar saniye ekranda
    kalırsa ve belirli bir büyüklükteyse şüpheli sayılır.
"""

import cv2
import numpy as np

_first_seen: dict[int, float] = {}
_last_seen:  dict[int, float] = {}

GRACE_PERIOD_SECONDS      = 3.0
UNATTENDED_THRESHOLD_SECONDS = 5

# Şüpheli kişi: hem belirli süre ekranda kalmış hem de yeterince büyük olmalı
SUSPICIOUS_MIN_SECONDS = 8       # en az 8 saniye ekranda
SUSPICIOUS_WIDTH_PX    = 200     # ve en az 200px genişlikte

COLOR_RATIO_THRESHOLD = 0.15

HSV_RANGES: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
    "white":      [(np.array([0, 0, 180]),    np.array([180, 50, 255]))],
    "black":      [(np.array([0, 0, 0]),      np.array([180, 255, 40]))],
    "navy":       [(np.array([100, 50, 20]),  np.array([130, 255, 90]))],
    "red":        [(np.array([0, 100, 80]),   np.array([10, 255, 255])),
                   (np.array([170, 100, 80]), np.array([180, 255, 255]))],
    "blue":       [(np.array([100, 80, 60]),  np.array([130, 255, 255]))],
    "green":      [(np.array([35, 30, 30]),   np.array([85, 255, 220]))],
    "yellow":     [(np.array([20, 80, 80]),   np.array([35, 255, 255]))],
    "orange":     [(np.array([10, 100, 100]), np.array([22, 255, 255]))],
    "purple":     [(np.array([130, 50, 50]),  np.array([160, 255, 255]))],
    "gray":       [(np.array([0, 0, 60]),     np.array([180, 40, 180]))],
    "dark_green": [(np.array([35, 30, 20]),   np.array([85, 255, 80]))],
    "beige":      [(np.array([15, 15, 140]),  np.array([35, 80, 220]))],
    "brown":      [(np.array([5, 30, 20]),    np.array([25, 180, 140]))],
}


def analyze(detections: list, video_time: float) -> list:
    """
    Parametreler:
        detections : [(track_id, label, x1, y1, x2, y2), ...]
        video_time : cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    Dönüş:
        alerts: [(alert_type, x1, y1, x2, y2), ...]
    """
    alerts    = []
    active_ids = {d[0] for d in detections}

    for track_id, label, x1, y1, x2, y2 in detections:
        if track_id not in _first_seen:
            _first_seen[track_id] = video_time
        _last_seen[track_id] = video_time

        elapsed = video_time - _first_seen[track_id]

        # Sahipsiz çanta tespiti (zaman tabanlı — değişmedi)
        if label == "backpack":
            if elapsed > UNATTENDED_THRESHOLD_SECONDS:
                alerts.append(("unattended_bag", x1, y1, x2, y2))

        # Şüpheli kişi: artık hem süre hem boyut koşulu
        # (sadece piksel genişliği = slowmo'da yanlış alarm)
        if label == "person":
            width = x2 - x1
            if elapsed >= SUSPICIOUS_MIN_SECONDS and width > SUSPICIOUS_WIDTH_PX:
                alerts.append(("suspicious_person", x1, y1, x2, y2))

    # Grace period temizliği
    stale_ids = set(_first_seen.keys()) - active_ids
    for sid in stale_ids:
        last = _last_seen.get(sid, 0)
        if (video_time - last) > GRACE_PERIOD_SECONDS:
            del _first_seen[sid]
            _last_seen.pop(sid, None)

    return alerts


def get_elapsed_time(track_id: int, video_time: float) -> float:
    if track_id in _first_seen:
        return video_time - _first_seen[track_id]
    return 0.0


def get_all_elapsed_times(detections: list, video_time: float) -> dict[int, float]:
    result = {}
    for track_id, *_ in detections:
        result[track_id] = get_elapsed_time(track_id, video_time)
    return result


def check_color_in_roi(frame: np.ndarray,
                       x1: int, y1: int, x2: int, y2: int,
                       color_name: str,
                       label: str = "person") -> bool:
    if color_name not in HSV_RANGES:
        return False

    h, w = frame.shape[:2]
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(w, x2);  y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return False

    _bag_labels = {"backpack", "suitcase", "handbag"}
    if label == "person":
        mid_y = y1 + (y2 - y1) // 2
        roi = frame[y1:mid_y, x1:x2]
    elif label in _bag_labels:
        bw, bh   = x2 - x1, y2 - y1
        margin_x = int(bw * 0.10)
        margin_y = int(bh * 0.10)
        roi = frame[y1 + margin_y : y2 - margin_y,
                    x1 + margin_x : x2 - margin_x]
    else:
        roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in HSV_RANGES[color_name]:
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    total_pixels   = combined_mask.size
    matched_pixels = cv2.countNonZero(combined_mask)
    ratio          = matched_pixels / total_pixels if total_pixels > 0 else 0.0

    return ratio >= COLOR_RATIO_THRESHOLD


def reset():
    _first_seen.clear()
    _last_seen.clear()