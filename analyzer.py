"""
analyzer.py
-----------
Tespit edilen nesneleri analiz ederek güvenlik uyarıları üretir.

İki ana işlev:
  - analyze()           : Sahipsiz çanta ve şüpheli kişi tespiti (track_id tabanlı)
  - check_color_in_roi(): Bounding box içinde HSV renk analizi (komut sistemi için)
  - get_elapsed_time()  : Belirli track_id'nin ekranda kalma süresini döndürür

Zaman Kaynağı:
  Tüm zamanlama video'nun gerçek zaman damgasına (CAP_PROP_POS_MSEC) dayanır.
  time.time() kullanılmaz — çünkü FPS düşüklüğü veya frame atlama nedeniyle
  sistem saati ile video zaman çizelgesi arasında büyük sapma oluşur.

Grace Period mekanizması:
  YOLO tracking bazen nesneleri birkaç frame boyunca kaybedip aynı ID ile
  geri getirir (flickering). Bu durumda zamanlayıcının sıfırlanmasını önlemek
  için _last_seen sözlüğü kullanılır. Bir nesne kaybolduğunda _first_seen
  kaydı GRACE_PERIOD_SECONDS süresince korunur.
"""

import cv2
import numpy as np

# Track ID → ilk görülme zamanı (video saniyesi)
_first_seen: dict[int, float] = {}

# Track ID → son görülme zamanı (video saniyesi, grace period için)
_last_seen: dict[int, float] = {}

# Bir nesne kaybolduktan sonra kaç saniye bellekte tutulsun? (video süresi)
GRACE_PERIOD_SECONDS = 3.0

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
        (np.array([0, 0, 0]),      np.array([180, 255, 40])),
    ],
    "navy": [
        (np.array([100, 50, 20]),  np.array([130, 255, 90])),
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


def analyze(detections: list, video_time: float) -> list:
    """
    Parametreler:
        detections : detect_and_track() çıktısı →
                     [(track_id, label, x1, y1, x2, y2), ...]
        video_time : Videonun gerçek zaman damgası (saniye).
                     cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    Dönüş:
        alerts: [(alert_type, x1, y1, x2, y2), ...]
    """
    alerts = []

    # Aktif ID'leri topla
    active_ids = {d[0] for d in detections}

    for track_id, label, x1, y1, x2, y2 in detections:
        # Her nesne için ilk görülme zamanını kaydet
        if track_id not in _first_seen:
            _first_seen[track_id] = video_time

        # Son görülme zamanını güncelle
        _last_seen[track_id] = video_time

        elapsed = video_time - _first_seen[track_id]

        # ----- Sahipsiz Çanta Tespiti -----
        if label == "backpack":
            if elapsed > UNATTENDED_THRESHOLD_SECONDS:
                alerts.append(("unattended_bag", x1, y1, x2, y2))

        # ----- Şüpheli Kişi Tespiti -----
        # Kameraya çok yakın / büyük görünen kişi → potansiyel tehdit
        if label == "person":
            width = x2 - x1
            if width > SUSPICIOUS_WIDTH_PX:
                alerts.append(("suspicious_person", x1, y1, x2, y2))

    # Grace Period ile temizlik:
    # Artık görünmeyen ama grace period'u dolmuş track ID'leri temizle
    stale_ids = set(_first_seen.keys()) - active_ids
    for sid in stale_ids:
        last = _last_seen.get(sid, 0)
        if (video_time - last) > GRACE_PERIOD_SECONDS:
            del _first_seen[sid]
            _last_seen.pop(sid, None)
        # Grace period içindeyse → silme, zamanlayıcıyı koru

    return alerts


def get_elapsed_time(track_id: int, video_time: float) -> float:
    """
    Belirli bir track_id'nin ekranda görülme süresini (video saniyesi) döndürür.
    track_id bilinmiyorsa 0.0 döner.

    Parametreler:
        track_id   : Takip ID'si
        video_time : Videonun şu anki zaman damgası (saniye)
    """
    if track_id in _first_seen:
        return video_time - _first_seen[track_id]
    return 0.0


def get_all_elapsed_times(detections: list, video_time: float) -> dict[int, float]:
    """
    Tüm tespit edilen nesnelerin ekranda kalma sürelerini döndürür.
    draw modülünün analyzer'a bağımlı olmasını engellemek için kullanılır.

    Dönüş:
        {track_id: elapsed_seconds, ...}
    """
    result = {}
    for track_id, *_ in detections:
        result[track_id] = get_elapsed_time(track_id, video_time)
    return result


def check_color_in_roi(frame: np.ndarray,
                       x1: int, y1: int, x2: int, y2: int,
                       color_name: str,
                       label: str = "person") -> bool:
    """
    Bounding box içindeki renk varlığını kontrol eder.

    ROI seçimi label'a göre değişir:
      - "person"  → Üst yarı (tişört bölgesi)
      - Diğer     → Merkezi %80 (kenar gürültüsünü atlar)

    Parametreler:
        frame      : Tam BGR frame
        x1,y1,x2,y2: Bounding box koordinatları
        color_name : HSV_RANGES tablosundaki renk adı (ör. "white", "red")
        label      : YOLO etiketi — ROI kırpma stratejisini belirler

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

    # Label'a göre ROI seçimi
    _bag_labels = {"backpack", "suitcase", "handbag"}
    if label == "person":
        # Üst yarıyı kırp (tişört / üst giysi bölgesi)
        mid_y = y1 + (y2 - y1) // 2
        roi = frame[y1:mid_y, x1:x2]
    elif label in _bag_labels:
        # Merkezi %80 — kenar gürültüsünü atla
        bw, bh = x2 - x1, y2 - y1
        margin_x = int(bw * 0.10)
        margin_y = int(bh * 0.10)
        roi = frame[y1 + margin_y : y2 - margin_y,
                     x1 + margin_x : x2 - margin_x]
    else:
        # Diğer nesneler → tam bounding box
        roi = frame[y1:y2, x1:x2]

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
    _last_seen.clear()