"""
utils/draw.py
-------------
OpenCV ile frame üzerine bounding box, etiket ve mod göstergesi çizimi.

Gözetim modu çizimi:
  draw_surveillance() → track_id + alert renkleri + komut bilgisi + ekranda kalma süresi

Highlight sistemi:
  NLP komutunun TÜM koşullarını sağlayan nesneler highlight_ids ile işaretlenir.
  Bu nesneler Cyan renkli, kalın çizgili "lock-on" bounding box ile çizilir.
"""

import cv2
import numpy as np

# Renk paleti (BGR)
COLOR_DEFAULT       = (180, 180, 180)   # Gri  – normal nesne
COLOR_UNATTENDED    = (0, 60, 255)      # Kırmızı – sahipsiz çanta
COLOR_SUSPICIOUS    = (0, 140, 255)     # Turuncu – şüpheli kişi
COLOR_HIGHLIGHT     = (255, 255, 0)     # Cyan – komut eşleşmesi (lock-on)
COLOR_TRACKED       = (50, 220, 50)     # Yeşil – takip edilen nesne
COLOR_ALERT_OVERLAY = (0, 0, 200)       # Kırmızı yarı-saydam dolgu

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.55
THICKNESS   = 2
THICK_HL    = 4     # Highlight bounding box kalınlığı


def _format_time(seconds: float) -> str:
    """Saniyeyi MM:SS formatına dönüştürür."""
    minutes = int(seconds) // 60
    secs    = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def _draw_box(frame, x1, y1, x2, y2, color, label_text,
              time_text="", highlight=False):
    """Tek bir bounding box + etiket + süre çizer."""
    thickness = THICK_HL if highlight else THICKNESS
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Etiket + süre birleşik metni
    full_label = f"{label_text}  {time_text}" if time_text else label_text

    # Arka plan şerit (highlight: tamamen dolgulu, daha büyük font)
    scale = 0.60 if highlight else FONT_SCALE
    (tw, th), _ = cv2.getTextSize(full_label, FONT, scale, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)

    cv2.putText(
        frame, full_label,
        (x1 + 3, y1 - 5),
        FONT, scale,
        (0, 0, 0) if highlight else (255, 255, 255),
        2 if highlight else 1,
        cv2.LINE_AA,
    )


def draw_surveillance(frame, detections, alerts, mode="all",
                      command_info="", highlight_ids=None,
                      elapsed_times=None):
    """
    Gözetim modu çizimi.

    detections    : [(track_id, label, x1, y1, x2, y2)]
    alerts        : [(alert_type, x1, y1, x2, y2)]
    mode          : "all" | "person" | "bag"
    command_info  : Aktif komut bilgisi (frame üzerine yazılır)
    highlight_ids : set of track_id — NLP komut koşullarını sağlayan nesneler
    elapsed_times : dict {track_id: float} — her nesnenin video süre bilgisi
    """
    frame = frame.copy()
    if highlight_ids is None:
        highlight_ids = set()
    if elapsed_times is None:
        elapsed_times = {}

    # Uyarı koordinatlarını hızlı lookup için set'e al
    alert_map = {(a[1], a[2], a[3], a[4]): a[0] for a in alerts}

    # Çanta etiketleri (bag filtresi için)
    bag_labels = {"backpack", "suitcase", "handbag"}

    for track_id, label, x1, y1, x2, y2 in detections:
        # ── Mod filtresi ─────────────────────────────────────────
        if mode == "person" and label != "person":
            continue
        if mode == "bag" and label not in bag_labels:
            continue

        coords = (x1, y1, x2, y2)
        alert_type = alert_map.get(coords)
        is_highlighted = track_id in highlight_ids

        # Ekranda kalma süresi
        elapsed = elapsed_times.get(track_id, 0.0)
        time_text = f"[{_format_time(elapsed)}]" if elapsed > 0 else ""

        # ── Renk ve etiket belirleme ─────────────────────────────
        if is_highlighted:
            # Komut eşleşmesi → Cyan lock-on
            color      = COLOR_HIGHLIGHT
            label_text = f"HEDEF: {label} [ID:{track_id}]"
            # Yarı-saydam cyan dolgu
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_HIGHLIGHT, -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        elif alert_type == "unattended_bag":
            color      = COLOR_UNATTENDED
            label_text = f"UYARI: Sahipsiz Canta [ID:{track_id}]"
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_ALERT_OVERLAY, -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        elif alert_type == "suspicious_person":
            color      = COLOR_SUSPICIOUS
            label_text = f"SUPHE: Kisi [ID:{track_id}]"

        else:
            color      = COLOR_DEFAULT
            label_text = f"{label} [ID:{track_id}]"

        _draw_box(frame, x1, y1, x2, y2, color, label_text,
                  time_text, highlight=is_highlighted)

    # Sol üst köşeye mod göstergesi
    mode_labels = {
        "all": "Hepsi", "person": "Kisiler", "bag": "Cantalar",
    }
    mode_text = f"MOD: Gozetim | Filtre: {mode_labels.get(mode, mode)}"
    cv2.putText(
        frame, mode_text,
        (10, 25), FONT, 0.6, (200, 255, 100), 2, cv2.LINE_AA,
    )

    # Aktif komut bilgisi
    if command_info:
        cv2.putText(
            frame, f"KOMUT: {command_info}",
            (10, 50), FONT, 0.55, (255, 200, 0), 1, cv2.LINE_AA,
        )

    return frame