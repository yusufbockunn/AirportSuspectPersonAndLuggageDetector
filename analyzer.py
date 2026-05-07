"""
analyzer.py
-----------
Tespit edilen nesneleri analiz ederek güvenlik uyarıları üretir.

İki ana işlev:
  - analyze()           : Sahipsiz çanta ve şüpheli kişi tespiti (track_id tabanlı)
  - check_color_in_roi(): Bounding box içinde HSV renk analizi (komut sistemi için)
"""

import cv2
import time
import numpy as np

# Track ID → ilk görülme zamanı
_first_seen: dict[int, float] = {}

# Bir çanta kaç saniye hareketsiz kalırsa uyarı tetiklensin?
UNATTENDED_THRESHOLD_SECONDS = 5

# Şüpheli kişi için minimum piksel genişliği (kameraya yakınlık proxy'si)
SUSPICIOUS_WIDTH_PX = 200

# ROI'deki hedef renk piksel oranı eşiği (%15)
COLOR_RATIO_THRESHOLD = 0.15


# ── HSV Renk Aralıkları ─────────────────────────────────────────────────────
# Her renk: (lower_hsv, upper_hsv) — birden fazla aralık olabilir (kırmızı gibi)
HSV_RANGES: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
    "white": [
        (np.array([0, 0, 180]),    np.array([180, 50, 255])),
    ],
    "black": [
        (np.array([0, 0, 0]),      np.array([180, 80, 60])),
    ],
    "red": [
        (np.array([0, 100, 80]),   np.array([10, 255, 255])),
        (np.array([170, 100, 80]), np.array([180, 255, 255])),
    ],
    "blue": [
        (np.array([100, 80, 60]),  np.array([130, 255, 255])),
    ],
    "green": [
        (np.array([35, 60, 60]),   np.array([85, 255, 255])),
    ],
    "yellow": [
        (np.array([20, 80, 80]),   np.array([35, 255, 255])),
    ],
    "orange": [
        (np.array([10, 100, 100]), np.array([22, 255, 255])),
    ],
    "purple": [
        (np.array([130, 50, 50]),  np.array([160, 255, 255])),
    ],
    "gray": [
        (np.array([0, 0, 60]),     np.array([180, 40, 180])),
    ],
}


def analyze(detections: list) -> list:
    """
    Parametreler:
        detections: detect_and_track() çıktısı →
                    [(track_id, label, x1, y1, x2, y2), ...]

    Dönüş:
        alerts: [(alert_type, x1, y1, x2, y2), ...]
    """
    alerts = []
    now    = time.time()

    for track_id, label, x1, y1, x2, y2 in detections:
        # ----- Sahipsiz Çanta Tespiti -----
        if label == "backpack":
            if track_id not in _first_seen:
                _first_seen[track_id] = now          # İlk görülme anını kaydet

            elapsed = now - _first_seen[track_id]
            if elapsed > UNATTENDED_THRESHOLD_SECONDS:
                alerts.append(("unattended_bag", x1, y1, x2, y2))

        # ----- Şüpheli Kişi Tespiti -----
        # Kameraya çok yakın / büyük görünen kişi → potansiyel tehdit
        if label == "person":
            width = x2 - x1
            if width > SUSPICIOUS_WIDTH_PX:
                alerts.append(("suspicious_person", x1, y1, x2, y2))

    # Artık görünmeyen track ID'leri temizle (bellek sızıntısı önlemi)
    active_ids = {d[0] for d in detections}
    stale_ids  = set(_first_seen.keys()) - active_ids
    for sid in stale_ids:
        del _first_seen[sid]

    return alerts


def check_color_in_roi(frame: np.ndarray,
                       x1: int, y1: int, x2: int, y2: int,
                       color_name: str) -> bool:
    """
    Verilen bounding box'ın ÜST YARISINI (tişört bölgesi) kırparak
    HSV renk aralığına göre hedef rengin varlığını kontrol eder.

    Parametreler:
        frame      : Tam BGR frame
        x1,y1,x2,y2: Bounding box koordinatları
        color_name : HSV_RANGES tablosundaki renk adı (ör. "white", "red")

    Dönüş:
        True  → Hedef renk ROI alanının %15'inden fazlasını kaplıyor
        False → Yetersiz renk eşleşmesi veya geçersiz renk adı
    """
    if color_name not in HSV_RANGES:
        return False

    h, w = frame.shape[:2]

    # Bounding box sınırlarını kontrol et
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return False

    # Üst yarıyı kırp (tişört / üst giysi bölgesi)
    mid_y = y1 + (y2 - y1) // 2
    roi = frame[y1:mid_y, x1:x2]

    if roi.size == 0:
        return False

    # BGR → HSV dönüşümü
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Tüm aralıkları birleştirerek maske oluştur
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in HSV_RANGES[color_name]:
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Piksel oranını hesapla
    total_pixels   = combined_mask.size
    matched_pixels = cv2.countNonZero(combined_mask)
    ratio          = matched_pixels / total_pixels if total_pixels > 0 else 0.0

    return ratio >= COLOR_RATIO_THRESHOLD


def reset():
    """Gözetim modu yeniden başlatıldığında state'i temizle."""
    _first_seen.clear()