"""
utils/draw.py
-------------
OpenCV ile frame üzerine bounding box, etiket ve mod göstergesi çizimi.

Gözetim modu çizimi:
  draw_surveillance() → track_id + alert renkleri + komut bilgisi
"""

import cv2
import numpy as np

# Renk paleti (BGR)
COLOR_DEFAULT       = (180, 180, 180)   # Gri  – normal nesne
COLOR_UNATTENDED    = (0, 60, 255)      # Kırmızı – sahipsiz çanta
COLOR_SUSPICIOUS    = (0, 140, 255)     # Turuncu – şüpheli kişi
COLOR_COLOR_MATCH   = (255, 200, 0)     # Cyan – renk eşleşmesi
COLOR_TRACKED       = (50, 220, 50)     # Yeşil – takip edilen nesne
COLOR_ALERT_OVERLAY = (0, 0, 200)       # Kırmızı yarı-saydam dolgu

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.55
THICKNESS   = 2


def _draw_box(frame, x1, y1, x2, y2, color, label_text):
    """Tek bir bounding box + etiket çizer (iç yardımcı fonksiyon)."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)

    # Arka plan şerit
    (tw, th), _ = cv2.getTextSize(label_text, FONT, FONT_SCALE, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)

    cv2.putText(
        frame, label_text,
        (x1 + 2, y1 - 4),
        FONT, FONT_SCALE,
        (255, 255, 255), 1, cv2.LINE_AA,
    )


def draw_surveillance(frame, detections, alerts, mode="all",
                      command_info="", color_matches=None):
    """
    Gözetim modu çizimi.

    detections    : [(track_id, label, x1, y1, x2, y2)]
    alerts        : [(alert_type, x1, y1, x2, y2)]
    mode          : "all" | "unattended_bag" | "suspicious_person" | "person"
    command_info  : Aktif komut bilgisi (frame üzerine yazılır)
    color_matches : set of (x1, y1, x2, y2) — renk eşleşen bbox'lar
    """
    frame = frame.copy()
    if color_matches is None:
        color_matches = set()

    # Uyarı koordinatlarını hızlı lookup için set'e al
    alert_map = {(a[1], a[2], a[3], a[4]): a[0] for a in alerts}

    for track_id, label, x1, y1, x2, y2 in detections:
        # Mod filtresi
        if mode == "unattended_bag" and label != "backpack":
            continue
        if mode == "suspicious_person" and label != "person":
            continue
        if mode == "person" and label != "person":
            continue

        coords = (x1, y1, x2, y2)
        alert_type = alert_map.get(coords)

        # Renk eşleşmesi varsa özel vurgulama
        if coords in color_matches:
            color      = COLOR_COLOR_MATCH
            label_text = f"ESLESME: {label} [ID:{track_id}]"
            # Cyan yarı-saydam dolgu
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_COLOR_MATCH, -1)
            cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

        elif alert_type == "unattended_bag":
            color      = COLOR_UNATTENDED
            label_text = f"UYARI: Sahipsiz Canta [ID:{track_id}]"
            # Yarı saydam dolgu
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_ALERT_OVERLAY, -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        elif alert_type == "suspicious_person":
            color      = COLOR_SUSPICIOUS
            label_text = f"SUPHE: Kisi [ID:{track_id}]"

        else:
            color      = COLOR_DEFAULT
            label_text = f"{label} [ID:{track_id}]"

        _draw_box(frame, x1, y1, x2, y2, color, label_text)

    # Sol üst köşeye mod göstergesi
    mode_text = f"MOD: Gozetim | Filtre: {mode}"
    cv2.putText(
        frame, mode_text,
        (10, 25), FONT, 0.6, (200, 255, 100), 2, cv2.LINE_AA,
    )

    # Aktif komut bilgisi (varsa ikinci satıra)
    if command_info:
        cv2.putText(
            frame, f"KOMUT: {command_info}",
            (10, 50), FONT, 0.55, (255, 200, 0), 1, cv2.LINE_AA,
        )

    return frame