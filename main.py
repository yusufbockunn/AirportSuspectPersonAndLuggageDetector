"""
main.py
-------
Tek Pencere Mimarisi — v8  (Multi-Condition Intersection Filtering)

Mimari:
  • Arka plan thread'i: VideoCapture okuma + YOLO inference + analiz + çizim
  • collections.deque(maxlen=2): İşlenmiş frame'ler thread-safe kuyrukta bekler
  • Main thread (root.after): Sadece kuyruktan frame alır ve Canvas'a yazar
  • UI hiçbir zaman ağır iş yapmaz → donma/takılma olmaz

Özellikler:
  • Çoklu koşul (target + color + time) KESİŞİM filtresi
  • NLP komutları radio butonları otomatik günceller
  • Video zaman damgası tabanlı süre hesaplama (CAP_PROP_POS_MSEC)
  • Eşleşen nesneler Cyan "lock-on" highlight ile vurgulanır
  • Persistent log dosyasına detaylı eşleşme kaydı
"""

import cv2
import time
import threading
import collections
import customtkinter as ctk

from detector       import detect_and_track
from analyzer       import (analyze, check_color_in_roi,
                            reset as reset_analyzer,
                            get_elapsed_time, get_all_elapsed_times)
from prompt_parser  import parse_prompt
from utils.draw     import draw_surveillance
from ui             import App

DEFAULT_VIDEO = "luggageVideo.mp4"

# UI tarafı ne sıklıkla kuyruğu kontrol eder (ms)
UI_POLL_MS = 30

# Çanta etiketi seti (bag filtresi için)
_BAG_LABELS = {"backpack", "suitcase", "handbag"}


def _format_time(seconds: float) -> str:
    """Saniyeyi MM:SS formatına dönüştürür (log mesajları için)."""
    minutes = int(seconds) // 60
    secs    = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


# ─────────────────────────────────────────────────────────────────────────────
# Gözetim Döngüsü (Threaded Pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class SurveillanceLoop:
    """
    İki parçalı pipeline:
      1. _worker_loop()  → Arka plan thread'i: frame oku → YOLO → analiz → çiz
      2. _ui_poll()      → Main thread (root.after): kuyruktan al → Canvas'a yaz
    """

    def __init__(self, root: ctk.CTk, app: App, video_path: str):
        self.root = root
        self.app  = app

        # ── Thread kontrol ─────────────────────────────────────────────
        self._running     = True
        self._cap: cv2.VideoCapture | None = None
        self._cap_lock    = threading.Lock()

        # ── Thread-safe kuyruk ─────────────────────────────────────────
        self._frame_queue: collections.deque = collections.deque(maxlen=2)

        # ── Aktif komut (main thread'den yazılır, worker'dan okunur) ───
        self._active_command: dict | None = None

        # ── FPS sayacı ─────────────────────────────────────────────────
        self._frame_times: list[float] = []

        # ── Uyarı log spam kontrolü ───────────────────────────────────
        self._last_alert_time: float = 0.0

        # ── Detaylı log throttle (video zamanı bazlı) ─────────────────
        self._last_detail_log: float = 0.0
        self._DETAIL_LOG_INTERVAL = 3.0

        self._open_video(video_path)
        reset_analyzer()

    # ── Video açma / değiştirme ────────────────────────────────────────────
    def _open_video(self, path: str):
        with self._cap_lock:
            if self._cap and self._cap.isOpened():
                self._cap.release()
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                self.app.log(f"[HATA] Video açılamadı: {path}", tag="error")
                self.app.set_status(f"[HATA] Video açılamadı: {path}")
            else:
                short = path.split('/')[-1].split(chr(92))[-1]
                self.app.log(f"Video başlatıldı: {short}", tag="ok")

    def load_new_video(self, path: str):
        """Yeni video dosyası yükle."""
        reset_analyzer()
        self._frame_times.clear()
        self._frame_queue.clear()
        self._last_detail_log = 0.0
        self._open_video(path)

    def set_command(self, cmd_dict: dict):
        """Aktif komutu ayarla + radio butonunu senkronize et."""
        self._active_command = cmd_dict
        self._last_detail_log = 0.0
        self.app.set_filter_mode(cmd_dict["base_filter"])

    def clear_command(self):
        """Aktif komutu temizle."""
        self._active_command = None

    # ── Başlat ─────────────────────────────────────────────────────────────
    def start(self):
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self.root.after(UI_POLL_MS, self._ui_poll)

    # ── Durdur ─────────────────────────────────────────────────────────────
    def stop(self):
        self._running = False
        with self._cap_lock:
            if self._cap:
                self._cap.release()
                self._cap = None

    # ─────────────────────────────────────────────────────────────────────
    # WORKER THREAD
    # ─────────────────────────────────────────────────────────────────────
    def _worker_loop(self):
        while self._running:
          try:
            # ── Frame oku + Video zaman damgası ─────────────────────
            with self._cap_lock:
                if self._cap is None or not self._cap.isOpened():
                    time.sleep(0.05)
                    continue

                ret, frame = self._cap.read()
                if not ret:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    reset_analyzer()
                    self._last_detail_log = 0.0
                    time.sleep(0.01)
                    continue

                video_time_sec = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # ── Tespit & Takip ─────────────────────────────────────
            detections = detect_and_track(frame)

            # ── Analiz (video zamanıyla) ───────────────────────────
            alerts = analyze(detections, video_time_sec)

            # ── Tüm nesnelerin süre bilgileri ──────────────────────
            elapsed_times = get_all_elapsed_times(detections, video_time_sec)

            # ── Uyarılar (throttled) ───────────────────────────────
            alert_msgs = []
            now_sys = time.time()
            if alerts and (now_sys - self._last_alert_time) > 1.0:
                self._last_alert_time = now_sys
                for alert_type, *_ in alerts:
                    if alert_type == "unattended_bag":
                        alert_msgs.append(("Tespit: Sahipsiz çanta!", "warn"))
                    elif alert_type == "suspicious_person":
                        alert_msgs.append(("Tespit: Şüpheli kişi!", "warn"))

            # ── Aktif modu oku ─────────────────────────────────────
            mode          = self.app.get_filter_mode()
            command_info  = ""
            highlight_ids = set()
            file_log_msg  = None

            # ──────────────────────────────────────────────────────
            # MULTI-CONDITION INTERSECTION FILTERING
            # ──────────────────────────────────────────────────────
            cmd = self._active_command
            if cmd is not None:
                command_info = cmd["raw"]
                target    = cmd["target"]
                color     = cmd["color"]
                time_sec  = cmd["time_sec"]
                motionless = cmd.get("motionless", False)

                # Her detection için TÜM koşulları kontrol et
                for track_id, label, x1, y1, x2, y2 in detections:
                    # Koşul 1: Hedef nesne filtresi
                    if target is not None:
                        if target == "bag":
                            # Meta-hedef: tüm çanta türleri
                            if label not in _BAG_LABELS:
                                continue
                        elif label != target:
                            continue

                    # Koşul 2: Süre eşiği (time_sec veya motionless+time)
                    if time_sec is not None:
                        elapsed = elapsed_times.get(track_id, 0.0)
                        if elapsed < time_sec:
                            continue

                    # Koşul 3: Renk eşleşmesi (label-aware ROI)
                    if color is not None:
                        if not check_color_in_roi(
                                frame, x1, y1, x2, y2, color,
                                label=label):
                            continue

                    # Tüm koşullar sağlandı → highlight
                    highlight_ids.add(track_id)

                # ── Detaylı persistent log (throttled) ─────────────
                if (highlight_ids and
                        (video_time_sec - self._last_detail_log)
                        >= self._DETAIL_LOG_INTERVAL):
                    self._last_detail_log = video_time_sec
                    parts = []
                    for tid, lbl, *_ in detections:
                        if tid in highlight_ids:
                            et = elapsed_times.get(tid, 0.0)
                            parts.append(
                                f"{lbl.capitalize()} [ID:{tid}] "
                                f"{_format_time(et)}")
                    conds = []
                    if target:
                        conds.append(f"target={target}")
                    if color:
                        conds.append(f"color={color}")
                    if time_sec:
                        conds.append(f"time>{time_sec:.0f}s")
                    if motionless:
                        conds.append("motionless")
                    file_log_msg = (
                        f"Query [{', '.join(conds)}] → "
                        f"{len(highlight_ids)} match: "
                        f"{', '.join(parts)}")

            # ── Strict mode filtresi (çizim öncesi) ─────────────
            # Radio butonuna göre SADECE eşleşen etiketler kalır
            if mode == "person":
                detections = [d for d in detections if d[1] == "person"]
            elif mode == "bag":
                detections = [d for d in detections if d[1] in _BAG_LABELS]
            # mode == "all" → filtreleme yok

            # elapsed_times'ı filtrelenmiş listeye göre güncelle
            if mode != "all":
                elapsed_times = {
                    d[0]: elapsed_times.get(d[0], 0.0) for d in detections}

            # ── Çizim ─────────────────────────────────────────────
            rendered = draw_surveillance(
                frame, detections, alerts, mode,
                command_info=command_info,
                highlight_ids=highlight_ids,
                elapsed_times=elapsed_times,
            )

            # ── Durum bilgisi ──────────────────────────────────────
            n_obj = len(detections)
            n_hl  = len(highlight_ids)
            n_al  = len(alerts)

            if n_hl:
                status = f"🎯  {n_hl} eşleşme  |  {n_obj} nesne"
            elif n_al:
                status = f"⚠  {n_al} uyarı  |  {n_obj} nesne"
            else:
                status = f"✔  Normal  |  {n_obj} nesne"

            # ── Kuyruğa koy ───────────────────────────────────────
            self._frame_queue.append({
                "rendered":     rendered,
                "status":       status,
                "alert_msgs":   alert_msgs,
                "file_log_msg": file_log_msg,
            })

            time.sleep(0.01)

          except Exception as e:
            import traceback
            print(f"[WORKER HATA] {e}")
            traceback.print_exc()
            time.sleep(0.5)

    # ─────────────────────────────────────────────────────────────────────
    # UI POLLER — Main thread
    # ─────────────────────────────────────────────────────────────────────
    def _ui_poll(self):
        if not self._running:
            return

        data = None
        while self._frame_queue:
            data = self._frame_queue.popleft()

        if data is not None:
            self.app.update_video_frame(data["rendered"])
            self.app.set_status(data["status"])

            for msg, tag in data["alert_msgs"]:
                self.app.log(msg, tag=tag)

            file_log_msg = data.get("file_log_msg")
            if file_log_msg:
                self.app.log_to_file_only(file_log_msg, tag="data")

            now = time.time()
            self._frame_times.append(now)
            self._frame_times = [t for t in self._frame_times if now - t < 1.0]
            self.app.set_fps(len(self._frame_times))

        self.root.after(UI_POLL_MS, self._ui_poll)


# ─────────────────────────────────────────────────────────────────────────────
# Başlatma
# ─────────────────────────────────────────────────────────────────────────────

def main():
    root = ctk.CTk()
    app  = App(root)

    loop = SurveillanceLoop(root, app, DEFAULT_VIDEO)
    loop.start()

    # ── Komut callback ─────────────────────────────────────────────────
    def on_command(prompt: str):
        cmd = parse_prompt(prompt)
        target     = cmd["target"]
        color      = cmd["color"]
        time_sec   = cmd["time_sec"]
        motionless = cmd.get("motionless", False)
        base       = cmd["base_filter"]

        # Log detayları
        detail_parts = [f"Filtre: {base}"]
        if target:
            detail_parts.append(f"Hedef: {target}")
        if color:
            detail_parts.append(f"Renk: {color}")
        if time_sec is not None:
            detail_parts.append(f"Süre: >{time_sec:.0f}s")
        if motionless:
            detail_parts.append("Hareketsiz")
        app.log(f"  → {' | '.join(detail_parts)}", tag="cmd")

        # Reset komutu
        if (base == "all" and target is None
                and color is None and time_sec is None):
            loop.clear_command()
            app.set_filter_mode("all")
            app.log("Komut temizlendi: Tüm nesneler gösteriliyor.", tag="ok")
            return

        # Komutu aktifle + radio butonunu otomatik güncelle
        loop.set_command(cmd)
        app.log(f"Komut aktif: {cmd['raw']}", tag="ok")

    app.set_command_callback(on_command)
    app.set_load_video_callback(loop.load_new_video)
    app.set_filter_change_callback(loop.clear_command)   # Radio tıklama → komutu temizle

    # Başlangıç logları
    app.log("Sistem başlatıldı. Varsayılan video: luggageVideo.mp4", tag="ok")
    app.log("Filtre: Hepsi (tüm nesneler)", tag="info")
    app.log("Komut girişi hazır (TR/EN desteklenir).", tag="info")

    def on_close():
        loop.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()