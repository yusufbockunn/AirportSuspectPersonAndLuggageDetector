"""
prompt_parser.py
----------------
Doğal dil komutlarını (Türkçe / İngilizce) ayrıştırarak
gözetim sistemine aksiyon komutları döndürür.

Döndürülen sözlük yapısı:
    {
        "action" : str,          # filter_all | filter_bag | filter_suspicious
                                 # filter_person | find_color | track
        "target" : str,          # Hedef nesne etiketi  (person, backpack, ball …)
        "color"  : str | None,   # HSV renk adı (white, red, blue …) veya None
        "raw"    : str,          # Orijinal kullanıcı metni
    }
"""

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
    # İngilizce
    "white":   "white",
    "black":   "black",
    "red":     "red",
    "blue":    "blue",
    "green":   "green",
    "yellow":  "yellow",
    "orange":  "orange",
    "purple":  "purple",
    "gray":    "gray",
    "grey":    "gray",
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
    "çanta":   "backpack",
    "canta":   "backpack",
    "valiz":   "suitcase",
    "bavul":   "suitcase",
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
    "bag":     "backpack",
    "suitcase":"suitcase",
    "ball":    "sports ball",
    "car":     "car",
    "bus":     "bus",
    "bicycle": "bicycle",
    "dog":     "dog",
    "cat":     "cat",
}


def parse_prompt(prompt: str) -> dict:
    """
    Kullanıcı metnini ayrıştırarak aksiyon sözlüğü döndürür.

    Öncelik sırası:
        1. Renk + nesne komutu  →  find_color
        2. Şüpheli kişi komutu  →  filter_suspicious
        3. Sahipsiz çanta komutu →  filter_bag
        4. Takip komutu          →  track
        5. Nesne filtresi        →  filter_person (veya diğer hedefler)
        6. Hiçbiri               →  filter_all
    """
    raw   = prompt.strip()
    lower = raw.lower()

    result = {
        "action": "filter_all",
        "target": "person",
        "color":  None,
        "raw":    raw,
    }

    # ── 0. "hepsini göster" / "show all" tarzı komutlar ──────────────────
    if _any_in(lower, ["hepsini göster", "hepsini goster", "tümünü göster",
                        "tumunu goster", "show all", "all"]):
        result["action"] = "filter_all"
        return result

    # ── 1. Renk ayrıştırma ───────────────────────────────────────────────
    detected_color = None
    for tr_color, en_color in _COLOR_MAP.items():
        if tr_color in lower:
            detected_color = en_color
            break

    # ── 2. Hedef nesne ayrıştırma ────────────────────────────────────────
    detected_target = None
    for keyword, label in _TARGET_MAP.items():
        if keyword in lower:
            detected_target = label
            break

    # ── 3. Aksiyon belirleme ─────────────────────────────────────────────

    # Renk + nesne → find_color
    if detected_color and detected_target:
        result["action"] = "find_color"
        result["target"] = detected_target
        result["color"]  = detected_color
        return result

    # Sadece renk (nesne belirtilmemiş → kişi varsayılır)
    if detected_color:
        result["action"] = "find_color"
        result["target"] = "person"
        result["color"]  = detected_color
        return result

    # Şüpheli kişi
    if _any_in(lower, ["şüpheli", "supheli", "suspicious", "tehlikeli",
                        "tehdit", "threat"]):
        result["action"] = "filter_suspicious"
        result["target"] = "person"
        return result

    # Sahipsiz çanta
    if _any_in(lower, ["sahipsiz", "terk edilmiş", "terk edilmis",
                        "unattended", "abandoned"]):
        result["action"] = "filter_bag"
        result["target"] = "backpack"
        return result

    # Takip komutu
    if _any_in(lower, ["takip", "izle", "track", "follow"]):
        result["action"] = "track"
        if detected_target:
            result["target"] = detected_target
        return result

    # Bul / ara komutu
    if _any_in(lower, ["bul", "ara", "find", "search", "göster", "goster",
                        "mark", "işaretle", "isaretle"]):
        if detected_target:
            result["action"] = "track"
            result["target"] = detected_target
        return result

    # Sadece nesne adı yazılmışsa → filtre
    if detected_target:
        if detected_target == "person":
            result["action"] = "filter_person"
        elif detected_target == "backpack":
            result["action"] = "filter_bag"
        else:
            result["action"] = "track"
        result["target"] = detected_target
        return result

    return result


def _any_in(text: str, keywords: list[str]) -> bool:
    """Keyword listesinden herhangi biri metinde geçiyor mu?"""
    return any(kw in text for kw in keywords)