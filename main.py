"""
main.py
-------
Tek Pencere Mimarisi — v6  (Threaded Pipeline)

Mimari:
  • Arka plan thread'i: VideoCapture okuma + YOLO inference + analiz + çizim
  • collections.deque(maxlen=2): İşlenmiş frame'ler thread-safe kuyrukta bekler
  • Main thread (root.after): Sadece kuyruktan frame alır ve Canvas'a yazar
  • UI hiçbir zaman ağır iş yapmaz → donma/takılma olmaz

Özellikler:
  • Metin Komut Sistemi (Text-to-Action)
  • HSV renk analizi ile giysi rengi filtresi
  • Radio buton filtreleri + metin komut override
"""

import cv2
import time
import threading
import collections
import customtkinter as ctk

from detector       import detect_and_track
from analyzer       import analyze, check_color_in_roi, reset as reset_analyzer
from prompt_parser  import parse_prompt
from utils.draw     import draw_surveillance
from ui             import App

DEFAULT_VIDEO = "luggageVideo.mp4"

# UI tarafı ne sıklıkla kuyruğu kontrol eder (ms)
# Bu değer sadece Canvas güncelleme gecikmesidir — inference'ı etkilemez.
UI_POLL_MS = 30


# ─────────────────────────────────────────────────────────────────────────────
# Gözetim Döngüsü (Threaded Pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class SurveillanceLoop:
    """
    İki parçalı pipeline:
      1. _worker_loop()  → Arka plan thread'i: frame oku → YOLO → analiz → çiz
      2. _ui_poll()      → Main thread (root.after): kuyruktan al → Canvas'a yaz

    Thread-safe iletişim: collections.deque(maxlen=2)
    """

    def __init__(self, root: ctk.CTk, app: App, video_path: str):
        self.root = root
        self.app  = app

        # ── Thread kontrol ─────────────────────────────────────────────
        self._running     = True
        self._cap: cv2.VideoCapture | None = None
        self._cap_lock    = threading.Lock()       # VideoCapture erişim kilidi

        # ── Thread-safe kuyruk ─────────────────────────────────────────
        # maxlen=2: Sadece en güncel 2 frame tutulur, eski frame'ler atılır
        self._frame_queue: collections.deque = collections.deque(maxlen=2)

        # ── Aktif komut (main thread'den yazılır, worker'dan okunur) ───
        self._active_command: dict | None = None

        # ── FPS sayacı ─────────────────────────────────────────────────
        self._frame_times: list[float] = []

        # ── Uyarı log spam kontrolü ───────────────────────────────────
        self._last_alert_time: float = 0.0

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
        """Yeni video dosyası yükle (Main Thread'den çağrılır)."""
        reset_analyzer()
        self._frame_times.clear()
        self._frame_queue.clear()
        self._open_video(path)

    def set_command(self, cmd_dict: dict):
        """Aktif komutu ayarla (Main Thread'den çağrılır)."""
        self._active_command = cmd_dict
        action = cmd_dict["action"]
        mode_map = {
            "filter_all":        "all",
            "filter_bag":        "unattended_bag",
            "filter_suspicious": "suspicious_person",
            "filter_person":     "person",
        }
        if action in mode_map:
            self.app.set_filter_mode(mode_map[action])

    def clear_command(self):
        """Aktif komutu temizle."""
        self._active_command = None

    # ── Başlat ─────────────────────────────────────────────────────────────
    def start(self):
        """Worker thread'i ve UI poller'ı başlat."""
        # Arka plan thread'i
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        # UI poller (main thread)
        self.root.after(UI_POLL_MS, self._ui_poll)

    # ── Durdur ─────────────────────────────────────────────────────────────
    def stop(self):
        self._running = False
        with self._cap_lock:
            if self._cap:
                self._cap.release()
                self._cap = None

    # ─────────────────────────────────────────────────────────────────────
    # WORKER THREAD — Ağır iş burada yapılır
    # ─────────────────────────────────────────────────────────────────────
    def _worker_loop(self):
        """
        Arka plan thread'i:
          1. Frame oku (VideoCapture)
          2. YOLO inference + tracking
          3. Analiz (sahipsiz çanta, şüpheli kişi)
          4. Komut bazlı filtreleme + renk analizi
          5. Çizim (draw_surveillance)
          6. Sonucu kuyruğa koy
        """
        while self._running:
            # ── Frame oku ────────────────────────────────────────────
            with self._cap_lock:
                if self._cap is None or not self._cap.isOpened():
                    time.sleep(0.05)
                    continue
                ret, frame = self._cap.read()
                if not ret:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    reset_analyzer()
                    time.sleep(0.01)
                    continue

            # ── Tespit & Takip ───────────────────────────────────────
            detections = detect_and_track(frame)

            # ── Analiz ───────────────────────────────────────────────
            alerts = analyze(detections)

            # ── Uyarılar (throttled — saniyede max 1 log) ────────────
            alert_msgs = []
            now = time.time()
            if alerts and (now - self._last_alert_time) > 1.0:
                self._last_alert_time = now
                for alert_type, *_ in alerts:
                    if alert_type == "unattended_bag":
                        alert_msgs.append(("Tespit: Sahipsiz çanta!", "warn"))
                    elif alert_type == "suspicious_person":
                        alert_msgs.append(("Tespit: Şüpheli kişi!", "warn"))

            # ── Komut bazlı işleme ───────────────────────────────────
            mode          = self.app.get_filter_mode()
            command_info  = ""
            color_matches = set()

            cmd = self._active_command
            if cmd is not None:
                action = cmd["action"]
                target = cmd["target"]
                color  = cmd["color"]
                command_info = cmd["raw"]

                if action == "find_color" and color:
                    mode = "all"
                    for track_id, label, x1, y1, x2, y2 in detections:
                        if label == target:
                            if check_color_in_roi(frame, x1, y1, x2, y2, color):
                                color_matches.add((x1, y1, x2, y2))

                elif action == "track":
                    detections = [d for d in detections if d[1] == target]
                    mode = "all"

                elif action == "filter_suspicious":
                    mode = "suspicious_person"
                elif action == "filter_bag":
                    mode = "unattended_bag"
                elif action == "filter_person":
                    mode = "person"
                elif action == "filter_all":
                    mode = "all"

            # ── Çizim ────────────────────────────────────────────────
            rendered = draw_surveillance(
                frame, detections, alerts, mode,
                command_info=command_info,
                color_matches=color_matches,
            )

            # ── Durum bilgisi ────────────────────────────────────────
            n_obj   = len(detections)
            n_alert = len(alerts)
            n_match = len(color_matches)

            if n_match:
                status = f"🎯  {n_match} eşleşme  |  {n_obj} nesne"
            elif n_alert:
                status = f"⚠  {n_alert} uyarı  |  {n_obj} nesne"
            else:
                status = f"✔  Normal  |  {n_obj} nesne"

            # ── Kuyruğa koy ──────────────────────────────────────────
            # deque(maxlen=2) → eski frame'ler otomatik atılır
            self._frame_queue.append({
                "rendered":   rendered,
                "status":     status,
                "alert_msgs": alert_msgs,
            })

            # CPU'yu %100 çalıştırmamak için kısa bekleme
            time.sleep(0.01)

    # ─────────────────────────────────────────────────────────────────────
    # UI POLLER — Main thread (hafif, sadece kuyruktan oku + Canvas güncelle)
    # ─────────────────────────────────────────────────────────────────────
    def _ui_poll(self):
        """
        root.after() ile çağrılır — Main Thread'de çalışır.
        Kuyruktan en güncel frame'i alır ve Canvas'a yazar.
        AĞIR İŞ YAPMAZ.
        """
        if not self._running:
            return

        # Kuyruktaki en güncel frame'i al (varsa)
        data = None
        while self._frame_queue:
            data = self._frame_queue.popleft()

        if data is not None:
            # Canvas güncelle
            self.app.update_video_frame(data["rendered"])

            # Durum çubuğu
            self.app.set_status(data["status"])

            # Uyarı logları (throttled)
            for msg, tag in data["alert_msgs"]:
                self.app.log(msg, tag=tag)

            # FPS
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

    # Gözetim döngüsü
    loop = SurveillanceLoop(root, app, DEFAULT_VIDEO)
    loop.start()

    # Komut callback
    def on_command(prompt: str):
        cmd = parse_prompt(prompt)
        action = cmd["action"]
        target = cmd["target"]
        color  = cmd["color"]

        detail_parts = [f"Aksiyon: {action}", f"Hedef: {target}"]
        if color:
            detail_parts.append(f"Renk: {color}")
        app.log(f"  → {' | '.join(detail_parts)}", tag="cmd")

        if action == "filter_all":
            loop.clear_command()
            app.set_filter_mode("all")
            app.log("Komut: Tüm nesneler gösteriliyor.", tag="ok")
        else:
            loop.set_command(cmd)
            app.log(f"Komut aktif: {cmd['raw']}", tag="ok")

    app.set_command_callback(on_command)
    app.set_load_video_callback(loop.load_new_video)

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