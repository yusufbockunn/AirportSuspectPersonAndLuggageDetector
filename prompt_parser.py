"""
prompt_parser.py
----------------
Doğal dil komutlarını (Türkçe / İngilizce) ayrıştırarak
gözetim sistemine çoklu koşul (multi-condition) komutları döndürür.

Döndürülen sözlük yapısı:
    {
        "target"      : str | None,   # "person", "bag" (meta), veya spesifik YOLO etiketi
        "color"       : str | None,   # HSV renk adı (white, red, blue, navy…) veya None
        "time_sec"    : float | None, # Saniye cinsinden süre eşiği veya None
        "motionless"  : bool,         # Hareketsiz/sahipsiz bağlamı var mı?
        "base_filter" : str,          # all | person | bag
        "raw"         : str,          # Orijinal kullanıcı metni
    }

Ayrıştırma mantığı:
    Tüm koşullar aynı anda çıkarılır (intersection), birbirini dışlamaz.
    Örn: "3 saniyeden fazla görünen kırmızı tişörtlü kişi"
        → target="person", color="red", time_sec=3, base_filter="person"
"""

import re

# ── Renk eşleme tablosu (TR → EN) ────────────────────────────────────────
_COLOR_MAP: dict[str, str] = {
    # Türkçe
    "beyaz":   "white",
    "siyah":   "black",
    "kırmızı": "red",
    "kirmizi": "red",
    "mavi":    "blue",
    "yeşil":   "green",
    "yesil":   "green",
    "sarı":    "yellow",
    "sari":    "yellow",
    "turuncu": "orange",
    "mor":     "purple",
    "gri":     "gray",
    "lacivert":    "navy",
    "koyu mavi":   "navy",
    "bej":         "beige",        # yeni
    "kahve":       "brown",        # yeni
    "kahverengi":  "brown",        # yeni
    "koyu yeşil":  "dark_green",   # yeni
    "koyu yesil":  "dark_green",   # yeni

    # İngilizce
    "white":      "white",
    "black":      "black",
    "red":        "red",
    "blue":       "blue",
    "green":      "green",
    "yellow":     "yellow",
    "orange":     "orange",
    "purple":     "purple",
    "gray":       "gray",
    "grey":       "gray",
    "navy":       "navy",
    "dark blue":  "navy",
    "beige":      "beige",         # yeni
    "brown":      "brown",         # yeni
    "dark green": "dark_green",    # yeni
}

# ── Nesne etiket eşlemeleri (TR → YOLO label) ────────────────────────────
_TARGET_MAP: dict[str, str] = {
    # Türkçe
    "kişi":    "person",
    "kisi":    "person",
    "insan":   "person",
    "adam":    "person",
    "kadın":   "person",
    "kadin":   "person",
    "çocuk":   "person",
    "cocuk":   "person",
    "çanta":   "bag",
    "canta":   "bag",
    "valiz":   "bag",
    "bavul":   "bag",
    "top":     "sports ball",
    "araba":   "car",
    "otobüs":  "bus",
    "otobus":  "bus",
    "bisiklet":"bicycle",
    "köpek":   "dog",
    "kopek":   "dog",
    "kedi":    "cat",
    # İngilizce
    "person":  "person",
    "people":  "person",
    "man":     "person",
    "woman":   "person",
    "child":   "person",
    "backpack":"backpack",
    "bag":     "bag",
    "suitcase":"suitcase",
    "luggage": "bag",
    "handbag": "handbag",
    "ball":    "sports ball",
    "car":     "car",
    "bus":     "bus",
    "bicycle": "bicycle",
    "dog":     "dog",
    "cat":     "cat",
}

# Çanta türleri (bag filtresi için)
_BAG_LABELS = {"backpack", "suitcase", "handbag"}

# ── Zaman birimi regex desenleri ──────────────────────────────────────────
_TIME_PATTERN = re.compile(
    r'(\d+)\s*'
    r'(dakika|dk|saniye|sn|dakikalık|dakikalik|saniyelik'
    r'|minutes?|mins?|seconds?|secs?)',
    re.IGNORECASE,
)


def _parse_time_condition(text: str) -> float | None:
    """Metinden zaman koşulunu çıkarır ve saniye cinsinden döndürür."""
    match = _TIME_PATTERN.search(text)
    if not match:
        return None

    value = int(match.group(1))
    unit  = match.group(2).lower()

    if unit in ("dakika", "dk", "dakikalık", "dakikalik",
                "minute", "minutes", "min", "mins"):
        return float(value * 60)
    if unit in ("saniye", "sn", "saniyelik",
                "second", "seconds", "sec", "secs"):
        return float(value)
    return None


def _detect_color(text: str) -> str | None:
    """Metinden renk çıkarır."""
    for keyword, en_color in _COLOR_MAP.items():
        if keyword in text:
            return en_color
    return None


def _detect_target(text: str) -> str | None:
    """Metinden hedef nesne çıkarır."""
    for keyword, label in _TARGET_MAP.items():
        if keyword in text:
            return label
    return None


def _is_suspicious(text: str) -> bool:
    """Metin şüpheli / tehdit bağlamı içeriyor mu?"""
    return _any_in(text, [
        "şüpheli", "supheli", "suspicious", "tehlikeli",
        "tehdit", "threat", "şüphe", "suphe",
    ])


def _is_unattended(text: str) -> bool:
    """Metin sahipsiz / terk edilmiş bağlamı içeriyor mu?"""
    return _any_in(text, [
        "sahipsiz", "terk edilmiş", "terk edilmis",
        "unattended", "abandoned", "bırakılmış", "birakilmis",
    ])


def _is_motionless(text: str) -> bool:
    """Metin hareketsiz / sabit / bekleyen bağlamı içeriyor mu?"""
    return _any_in(text, [
        "hareketsiz", "sabit", "bekleyen", "kaldı", "kaldi",
        "duran", "yerinde", "kıpırdamayan", "kipirdamayan",
        "motionless", "stationary", "still", "idle",
    ])


def _has_time_context(text: str) -> bool:
    """Metin zaman bağlamı içeriyor mu? (süre ile filtreleme niyeti)"""
    # Hareketsiz / sahipsiz bağlamları da dolaylı olarak zaman koşulu ima eder
    if _is_motionless(text) or _is_unattended(text):
        return True
    return _any_in(text, [
        "boyunca", "süre", "sure", "kalma", "kalan",
        "ekranda", "görünen", "gorunen", "bulunan",
        "fazla", "uzun", "aşan", "asan", "geçen", "gecen",
        "here for", "on screen", "for more than",
        "longer than", "at least", "visible for",
        "more than", "over",
    ])


def parse_prompt(prompt: str) -> dict:
    """
    Kullanıcı metnini ayrıştırarak çoklu koşul sözlüğü döndürür.

    Tüm koşullar eşzamanlı çıkarılır — birbirini dışlamaz.
    main.py bu koşulların KESİŞİMİNİ (intersection) uygular.
    """
    raw   = prompt.strip()
    lower = raw.lower()

    result = {
        "target":      None,
        "color":       None,
        "time_sec":    None,
        "motionless":  False,
        "base_filter": "all",
        "raw":         raw,
    }

    # ── "hepsini göster" / "show all" → reset ────────────────────────────
    if _any_in(lower, ["hepsini göster", "hepsini goster", "tümünü göster",
                        "tumunu goster", "show all", "reset", "sıfırla",
                        "sifirla", "temizle", "clear"]):
        return result  # base_filter="all", her şey None → reset

    # ── Koşulları eşzamanlı çıkar ────────────────────────────────────────
    result["color"]  = _detect_color(lower)
    result["target"] = _detect_target(lower)

    # Zaman koşulu (sadece bağlam varsa aktifleşir)
    time_val = _parse_time_condition(lower)
    if time_val is not None and _has_time_context(lower):
        result["time_sec"] = time_val

    # ── Motionless tespiti ────────────────────────────────────────────────
    if _is_motionless(lower) or _is_unattended(lower):
        result["motionless"] = True

    # ── base_filter belirleme (sadece 3 mod: all / person / bag) ──────
    target = result["target"]
    is_bag = (target == "bag") or (target in _BAG_LABELS) if target else False

    if _is_suspicious(lower) or _is_unattended(lower):
        # Şüpheli/sahipsiz bağlamı: hedefi belirle ama base_filter 3 modda kalır
        if is_bag or _is_unattended(lower):
            result["base_filter"] = "bag"
            if target is None:
                result["target"] = "bag"
        else:
            result["base_filter"] = "person"
            if target is None:
                result["target"] = "person"

    elif is_bag:
        result["base_filter"] = "bag"

    elif target == "person":
        result["base_filter"] = "person"

    elif target is not None:
        # Spesifik nesne (car, dog vs.) → all modunda ama target filtreli
        result["base_filter"] = "all"

    # Hedef yoksa ama renk veya süre varsa → kişi varsay
    if result["target"] is None and (result["color"] or result["time_sec"]):
        result["target"] = "person"
        if result["base_filter"] == "all":
            result["base_filter"] = "person"

    # ── Bul / takip / göster bağlamı — hedef varsa etkinleştir ───────────
    if result["target"] is None:
        if _any_in(lower, ["takip", "izle", "track", "follow",
                            "bul", "ara", "find", "search",
                            "göster", "goster", "mark",
                            "işaretle", "isaretle"]):
            result["target"] = "person"  # Varsayılan hedef

    return result


def _any_in(text: str, keywords: list[str]) -> bool:
    """Keyword listesinden herhangi biri metinde geçiyor mu?"""
    return any(kw in text for kw in keywords)