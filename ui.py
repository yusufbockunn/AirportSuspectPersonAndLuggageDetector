"""
ui.py
-----
CustomTkinter tabanlı modern arayüz — v6

Düzeltmeler (v6):
  • _trigger_command: placeholder kontrolü güvenilir hale getirildi
  • _clear_placeholder / _restore_placeholder: state yönetimi düzeltildi
  • _on_enter_key: return "break" her durumda uygulanıyor (çift gönderim önlendi)
  • set_filter_mode: artık log yazmıyor (main.py zaten log yazıyor, spam önlendi)
  • Komut kutusu her gönderimden sonra kesinlikle temizlenip placeholder konuluyor
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from datetime import datetime
import threading
import os
from pathlib import Path

# Speech Recognition (opsiyonel)
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

# ── CustomTkinter global ayarlar ────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ── Renk sabitleri ──────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#21262d"
ACCENT  = "#1f6feb"
TEXT    = "#c9d1d9"
MUTED   = "#8b949e"
GREEN   = "#3fb950"
YELLOW  = "#d29922"
RED     = "#f85149"
CYAN    = "#58a6ff"
DARK_BG = "#010409"

FONT = "Segoe UI"

PLACEHOLDER_TEXT = "Örn: beyaz tişörtlü kişiyi bul"


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


class App:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("Güvenlik Analiz Sistemi  |  YOLOv8 + OpenCV")
        self.root.configure(fg_color=BG)
        self.root.minsize(1200, 700)

        # ── Durum değişkenleri ─────────────────────────────────────────────
        self._mode        = tk.StringVar(value="all")
        self._status_text = tk.StringVar(value="Sistem hazır.")
        self._fps_text    = tk.StringVar(value="")
        self._video_label = tk.StringVar(value="luggageVideo.mp4")

        # Callback'ler
        self._on_command_cb       = None
        self._on_load_video_cb    = None
        self._on_filter_change_cb = None

        # Canvas görsel referansı
        self._current_imgtk  = None
        self._canvas_img_id  = None
        self._canvas_text_id = None

        # Placeholder durumu
        self._placeholder_active = False

        self._build_ui()
        self.root.bind("<Configure>", self._on_resize)

        # Persistent logging
        self._log_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "logs"
        self._log_dir.mkdir(exist_ok=True)
        self._log_file_lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────────
    # Ana UI yapısı
    # ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        header = ctk.CTkFrame(self.root, height=50, fg_color=PANEL, corner_radius=0)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        ctk.CTkLabel(header, text="⬡  GÜVENLİK ANALİZ SİSTEMİ",
                     text_color=GREEN,
                     font=(FONT, 15, "bold")).pack(side="left", padx=20, pady=12)
        ctk.CTkLabel(header, text="YOLOv8s  +  OpenCV  |  Video Analiz",
                     text_color=MUTED,
                     font=(FONT, 11)).pack(side="left", padx=6)
        ctk.CTkLabel(header, textvariable=self._fps_text,
                     text_color=GREEN,
                     font=(FONT, 11, "bold")).pack(side="right", padx=20)

        ctk.CTkFrame(self.root, height=1, fg_color=BORDER, corner_radius=0).pack(fill="x")

        content = ctk.CTkFrame(self.root, fg_color=BG, corner_radius=0)
        content.pack(fill="both", expand=True)
        self._build_left_panel(content)
        self._build_right_panel(content)

        ctk.CTkFrame(self.root, height=1, fg_color=BORDER, corner_radius=0).pack(fill="x")
        status_bar = ctk.CTkFrame(self.root, height=30, fg_color=BG, corner_radius=0)
        status_bar.pack(fill="x")
        status_bar.pack_propagate(False)
        ctk.CTkLabel(status_bar, textvariable=self._status_text,
                     text_color=YELLOW,
                     font=(FONT, 10)).pack(side="left", padx=14, pady=4)

    # ─────────────────────────────────────────────────────────────────────
    # Sol Panel
    # ─────────────────────────────────────────────────────────────────────
    def _build_left_panel(self, parent):
        left = ctk.CTkFrame(parent, width=300, fg_color=PANEL, corner_radius=0)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        # Video Seç
        self._section(left, "📁  VİDEO KAYNAĞI")
        video_row = ctk.CTkFrame(left, fg_color="transparent")
        video_row.pack(fill="x", padx=16, pady=(0, 6))

        ctk.CTkLabel(video_row, textvariable=self._video_label,
                     text_color=MUTED, font=(FONT, 9),
                     anchor="w").pack(side="left", fill="x", expand=True)
        ctk.CTkButton(video_row, text="📂 Video Seç",
                      width=100, height=28,
                      fg_color="#2d333b", hover_color=ACCENT,
                      text_color=TEXT, font=(FONT, 10, "bold"),
                      corner_radius=6,
                      command=self._load_video).pack(side="right")

        self._divider(left)

        # Gözetim Filtresi
        self._section(left, "📹  GÖZETİM FİLTRESİ")
        for label_text, value in [
            ("Hepsini Göster",  "all"),
            ("Sadece Kişiler",  "person"),
            ("Sadece Çantalar", "bag"),
        ]:
            ctk.CTkRadioButton(
                left, text=label_text,
                variable=self._mode, value=value,
                command=self._on_filter_change,
                fg_color=ACCENT, border_color=MUTED,
                hover_color=GREEN, text_color=TEXT,
                font=(FONT, 11),
            ).pack(anchor="w", padx=22, pady=3)

        self._divider(left)

        # Komut Girişi
        self._section(left, "🎯  KOMUT GİRİŞİ")
        ctk.CTkLabel(left, text="Komut girin (TR / EN):",
                     text_color=MUTED,
                     font=(FONT, 10)).pack(anchor="w", padx=18, pady=(0, 4))

        self._cmd_text = ctk.CTkTextbox(
            left, height=80,
            fg_color=DARK_BG, text_color=TEXT,
            font=(FONT, 11), corner_radius=8,
            border_width=1, border_color=BORDER,
        )
        self._cmd_text.pack(padx=16, pady=(0, 8), fill="x")
        self._set_placeholder()
        self._cmd_text.bind("<FocusIn>",  self._clear_placeholder)
        self._cmd_text.bind("<FocusOut>", self._restore_placeholder)
        self._cmd_text.bind("<Return>",   self._on_enter_key)

        btn_row = ctk.CTkFrame(left, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 12))

        ctk.CTkButton(
            btn_row, text="⚡ Komutu Uygula",
            fg_color=ACCENT, hover_color=GREEN,
            text_color="#ffffff", font=(FONT, 11, "bold"),
            height=34, corner_radius=8,
            command=self._trigger_command,
        ).pack(side="left", fill="x", expand=True, padx=(0, 6))

        self._mic_btn = ctk.CTkButton(
            btn_row, text="🎤 Ses",
            width=80, height=34,
            fg_color="#2d333b", hover_color=ACCENT,
            text_color=TEXT, font=(FONT, 11, "bold"),
            corner_radius=8,
            command=self._start_voice_input,
        )
        self._mic_btn.pack(side="right")

        self._divider(left)

        # Sistem Logları
        self._section(left, "📋  SİSTEM LOGLARI")

        self._log_box = ctk.CTkTextbox(
            left,
            fg_color=DARK_BG, text_color=MUTED,
            font=("Consolas", 9), corner_radius=8,
            border_width=1, border_color=BORDER,
            state="disabled",
        )
        self._log_box.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        self._log_box.tag_config("error", foreground=RED)
        self._log_box.tag_config("warn",  foreground=YELLOW)
        self._log_box.tag_config("ok",    foreground=GREEN)
        self._log_box.tag_config("info",  foreground=MUTED)
        self._log_box.tag_config("cmd",   foreground=CYAN)

    # ─────────────────────────────────────────────────────────────────────
    # Sağ Panel
    # ─────────────────────────────────────────────────────────────────────
    def _build_right_panel(self, parent):
        right = ctk.CTkFrame(parent, fg_color=BG, corner_radius=0)
        right.pack(side="left", fill="both", expand=True)

        self._canvas = tk.Canvas(right, bg="#000000", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, padx=10, pady=10)

        self._canvas_text_id = self._canvas.create_text(
            400, 260, text="Video yükleniyor…",
            fill=MUTED, font=(FONT, 14),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Yardımcı widget oluşturucular
    # ─────────────────────────────────────────────────────────────────────
    def _section(self, parent, title):
        ctk.CTkLabel(parent, text=title,
                     text_color=ACCENT,
                     font=(FONT, 11, "bold")).pack(anchor="w", padx=16, pady=(14, 5))

    def _divider(self, parent):
        ctk.CTkFrame(parent, height=1, fg_color=BORDER,
                     corner_radius=0).pack(fill="x", padx=12, pady=8)

    # ─────────────────────────────────────────────────────────────────────
    # Placeholder yönetimi
    # ─────────────────────────────────────────────────────────────────────
    def _set_placeholder(self):
        """Kutuyu temizle ve placeholder metnini gri renkte yaz."""
        self._cmd_text.configure(state="normal")
        self._cmd_text.delete("1.0", "end")
        self._cmd_text.insert("1.0", PLACEHOLDER_TEXT)
        self._cmd_text.configure(text_color=MUTED)
        self._placeholder_active = True

    def _clear_placeholder(self, _=None):
        """Odaklanıldığında placeholder'ı temizle."""
        if self._placeholder_active:
            self._cmd_text.delete("1.0", "end")
            self._cmd_text.configure(text_color=TEXT)
            self._placeholder_active = False

    def _restore_placeholder(self, _=None):
        """
        Odak kaybolduğunda kutu boşsa placeholder'ı geri koy.
        _trigger_command zaten placeholder set ediyor; bu sadece
        kullanıcı kutuyu boş bırakıp başka yere tıklarsa devreye girer.
        """
        # _trigger_command çağrısı placeholder'ı zaten set etti,
        # ikinci kez set etmemek için içerik kontrolü yap
        content = self._cmd_text.get("1.0", "end").strip()
        if not content or content == PLACEHOLDER_TEXT:
            if not self._placeholder_active:
                self._set_placeholder()

    def _get_real_text(self) -> str:
        """
        Komut kutusundaki gerçek kullanıcı metnini döndürür.
        Flag'e değil, doğrudan içeriğe bakarak placeholder tespiti yapar.
        Böylece focus event'i tetiklenmese bile doğru çalışır.
        """
        content = self._cmd_text.get("1.0", "end").strip()
        if not content or content == PLACEHOLDER_TEXT:
            return ""
        return content

    # ─────────────────────────────────────────────────────────────────────
    # Resize
    # ─────────────────────────────────────────────────────────────────────
    def _on_resize(self, _=None):
        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        if w > 1 and h > 1 and self._canvas_text_id:
            self._canvas.coords(self._canvas_text_id, w // 2, h // 2)

    # ─────────────────────────────────────────────────────────────────────
    # Event handler'lar
    # ─────────────────────────────────────────────────────────────────────
    def _on_filter_change(self):
        """Radio buton tıklandığında: aktif NLP komutunu temizle + log."""
        mode = self._mode.get()
        mode_labels = {
            "all":    "Hepsi (tüm nesneler)",
            "person": "Sadece Kişiler",
            "bag":    "Sadece Çantalar",
        }
        label = mode_labels.get(mode, mode)

        # Komut kutusunu temizle
        self._set_placeholder()

        if self._on_filter_change_cb:
            self._on_filter_change_cb()

        self.log(f"Filtre aktif: {label}", tag="info")

    def _load_video(self):
        path = filedialog.askopenfilename(
            title="Video Dosyası Seç",
            filetypes=[
                ("Video Dosyaları", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                ("Tüm Dosyalar",    "*.*"),
            ],
        )
        if not path:
            return
        short_name = path.split("/")[-1].split("\\")[-1]
        self._video_label.set(short_name)
        self.log(f"Video yüklendi: {short_name}", tag="ok")
        if self._on_load_video_cb:
            self._on_load_video_cb(path)
        else:
            self.log("[HATA] Video yükleme callback'i bağlanmadı.", tag="error")

    def _on_enter_key(self, event):
        """
        Enter → komutu gönder (her durumda).
        Shift+Enter → normal satır atlama.
        Her iki durumda da "break" dönerek TextBox'ın varsayılan
        davranışını engelliyoruz; Shift+Enter için manuel newline ekliyoruz.
        """
        if event.state & 0x1:  # Shift basılı
            self._cmd_text.insert("insert", "\n")
        else:
            self._trigger_command()
        return "break"  # Her durumda TextBox'ın kendi Enter işlemini engelle

    def _trigger_command(self):
        """Komut kutusundaki metni al ve callback'e gönder."""
        # İçeriği flag'e bakmadan direkt oku (_get_real_text placeholder ile karşılaştırır)
        prompt = self._get_real_text()

        if not prompt:
            self._set_placeholder()
            return

        # Kutuyu ÖNCE temizle, sonra callback çağır
        # (callback içinde başka UI işlemleri olursa placeholder state temiz olsun)
        self._set_placeholder()

        if self._on_command_cb:
            self._on_command_cb(prompt)
        else:
            self.log("[HATA] Komut callback'i bağlanmadı.", tag="error")

    # ─────────────────────────────────────────────────────────────────────
    # Ses Komutu (Speech-to-Text)
    # ─────────────────────────────────────────────────────────────────────
    def _start_voice_input(self):
        if not SR_AVAILABLE:
            self.log("[HATA] speech_recognition yüklü değil.\n"
                     "       pip install SpeechRecognition pyaudio", tag="error")
            return

        self._mic_btn.configure(
            text="🔴 Dinleniyor...",
            fg_color=RED,
            state="disabled",
        )
        self.log("Mikrofon açıldı — konuşmaya başlayın…", tag="cmd")

        thread = threading.Thread(target=self._voice_listen, daemon=True)
        thread.start()

    def _voice_listen(self):
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

            text = recognizer.recognize_google(audio, language="tr-TR")
            self.root.after(0, lambda: self._on_voice_result(text))

        except sr.WaitTimeoutError:
            self.root.after(0, lambda: self._on_voice_error(
                "Ses algılanamadı (zaman aşımı). Tekrar deneyin."))
        except sr.UnknownValueError:
            self.root.after(0, lambda: self._on_voice_error(
                "Ses anlaşılamadı, lütfen tekrar deneyin."))
        except sr.RequestError as e:
            self.root.after(0, lambda: self._on_voice_error(
                f"Google API hatası: {e}"))
        except Exception as e:
            self.root.after(0, lambda: self._on_voice_error(
                f"Mikrofon hatası: {e}"))

    def _on_voice_result(self, text: str):
        self._reset_mic_button()
        self.log(f"Ses komutu algılandı: \"{text}\"", tag="ok")

        # Metni kutuya yaz ve gönder
        self._placeholder_active = False
        self._cmd_text.configure(text_color=TEXT)
        self._cmd_text.delete("1.0", "end")
        self._cmd_text.insert("1.0", text)
        self._trigger_command()

    def _on_voice_error(self, msg: str):
        self._reset_mic_button()
        self.log(f"[SES] {msg}", tag="error")

    def _reset_mic_button(self):
        self._mic_btn.configure(
            text="🎤 Ses",
            fg_color="#2d333b",
            state="normal",
        )

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────
    def set_command_callback(self, fn):
        self._on_command_cb = fn

    def set_load_video_callback(self, fn):
        self._on_load_video_cb = fn

    def set_filter_change_callback(self, fn):
        self._on_filter_change_cb = fn

    def get_filter_mode(self) -> str:
        return self._mode.get()

    def set_filter_mode(self, mode: str):
        """Programatik mod değişikliği (NLP komutu ile). Log yazmaz."""
        self._mode.set(mode)

    def set_status(self, msg: str):
        self._status_text.set(msg)

    def set_fps(self, fps: float):
        self._fps_text.set(f"FPS: {fps:.1f}")

    def log(self, msg: str, tag: str = "info"):
        timestamp = _ts()

        self._log_box.configure(state="normal")
        self._log_box.insert("end", f"{timestamp} {msg}\n", tag)
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

        self._write_log_to_file(timestamp, msg, tag)

    def log_to_file_only(self, msg: str, tag: str = "data"):
        timestamp = _ts()
        self._write_log_to_file(timestamp, msg, tag)

    def update_video_frame(self, bgr_image: np.ndarray):
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        ih, iw = bgr_image.shape[:2]
        scale  = min(cw / iw, ch / ih)
        nw, nh = int(iw * scale), int(ih * scale)

        resized = cv2.resize(bgr_image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil     = Image.fromarray(rgb)
        imgtk   = ImageTk.PhotoImage(image=pil)

        self._current_imgtk = imgtk

        if self._canvas_text_id:
            self._canvas.delete(self._canvas_text_id)
            self._canvas_text_id = None

        cx, cy = cw // 2, ch // 2
        if self._canvas_img_id is None:
            self._canvas_img_id = self._canvas.create_image(
                cx, cy, anchor="center", image=imgtk)
        else:
            self._canvas.coords(self._canvas_img_id, cx, cy)
            self._canvas.itemconfig(self._canvas_img_id, image=imgtk)

    # ─────────────────────────────────────────────────────────────────────
    # Persistent Logging
    # ─────────────────────────────────────────────────────────────────────
    def _write_log_to_file(self, timestamp: str, msg: str, tag: str):
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            log_path = self._log_dir / f"system_log_{today}.txt"
            line = f"{timestamp} [{tag.upper()}] {msg}\n"

            with self._log_file_lock:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception:
            pass