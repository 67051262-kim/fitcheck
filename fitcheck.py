"""
Fit Check core logic
--------------------
ไม่ผูกกับ YOLO / FastAPI เลย รับเป็น list ของ detection แล้วคืนผลวิเคราะห์
จึงเทสต์ได้โดยไม่ต้องโหลดโมเดล (ดู test_logic.py)

detection = {"cls": "baby_tee", "conf": 0.87, "box": [x1, y1, x2, y2],   # box เป็นสัดส่วน 0-1
             "rgb": [r, g, b]}                                            # สีหลักของชิ้น (ไม่บังคับ)
"""
from __future__ import annotations

import colorsys
import re
from collections import defaultdict
from itertools import product

# ---------------------------------------------------------------------------
# 1) คลาสทั้ง 16 และหมวด
# ---------------------------------------------------------------------------
TOPS = ["Tube_top", "baby_tee", "long_sleeve", "puff_sleeve_top", "short_sleeve_top", "tank-top"]
BOTTOMS = [
    "baggy_pants", "long_pants", "long_skirt", "low_rise_jeans",
    "low_rise_pants", "mini_pants", "mini_skirt",
]
ONE_PIECES = ["long_dress", "mini_dress"]
OUTERWEAR = ["suit_jacket"]

CATEGORY: dict[str, str] = {}
for _n in TOPS:
    CATEGORY[_n] = "top"
for _n in BOTTOMS:
    CATEGORY[_n] = "bottom"
for _n in ONE_PIECES:
    CATEGORY[_n] = "one_piece"
for _n in OUTERWEAR:
    CATEGORY[_n] = "outer"

# ---------------------------------------------------------------------------
# 2) Mapping สไตล์  (แก้ตรงนี้ได้เลย ไม่ต้องเทรนโมเดลใหม่)
#    น้ำหนักของชิ้นในแต่ละสไตล์:  3 = เอกลักษณ์ (เห็นแล้วนึกถึงสไตล์นี้ทันที)
#                               2 = พบบ่อยในสไตล์นี้
#                               1 = ใส่ได้ แต่ไม่ได้บ่งบอกสไตล์
#    (ใส่เป็น list ธรรมดาก็ได้ จะถือว่าทุกชิ้นน้ำหนัก 1)
# ---------------------------------------------------------------------------
STYLES: dict[str, dict[str, int]] = {
    "Cleangirl":  {"tank-top": 3, "long_pants": 2, "short_sleeve_top": 2, "long_dress": 2,
                   "long_skirt": 1, "mini_dress": 1, "Tube_top": 1, "mini_pants": 1, "low_rise_pants": 1},
    "Formal":     {"suit_jacket": 3, "long_pants": 2, "long_dress": 2, "long_skirt": 2, "long_sleeve": 2},
    "Hyperpop":   {"mini_skirt": 3, "mini_pants": 3, "Tube_top": 2, "baby_tee": 2,
                   "tank-top": 1, "long_pants": 1, "long_sleeve": 1, "baggy_pants": 1, "mini_dress": 1,
                   "low_rise_pants": 1, "short_sleeve_top": 1},
    "Softgirl":   {"puff_sleeve_top": 3, "mini_dress": 3, "long_skirt": 2, "mini_skirt": 2,
                   "short_sleeve_top": 1, "tank-top": 1, "long_sleeve": 1, "long_dress": 1, "mini_pants": 1,
                   "baggy_pants": 1},
    "Streetwear": {"baggy_pants": 3, "short_sleeve_top": 2, "long_sleeve": 2,
                   "tank-top": 1, "baby_tee": 1, "Tube_top": 1, "low_rise_jeans": 1, "mini_dress": 1},
    "Y2K":        {"low_rise_jeans": 3, "low_rise_pants": 3, "baby_tee": 2, "Tube_top": 2, "mini_skirt": 2,
                   "tank-top": 1, "mini_pants": 1, "long_sleeve": 1},
}

# ---------------------------------------------------------------------------
# 2b) โทนสี  (แก้ได้เหมือน STYLES  น้ำหนัก 1-3 แบบเดียวกัน)
# ---------------------------------------------------------------------------
COLOR_GROUPS: dict[str, str] = {        # ชื่อในโค้ด -> ชื่อที่แสดง
    "white":      "ขาว",
    "cream":      "ครีม/เบจ",
    "grey":       "เทา",
    "black":      "ดำ",
    "brown":      "น้ำตาล",
    "navy":       "กรมท่า",
    "olive":      "เขียวขี้ม้า",
    "pastel":     "พาสเทล",
    "vivid":      "สด/นีออน",
    "denim":      "ยีนส์",
    "lightdenim": "ยีนส์อ่อน",
}
STYLE_COLORS: dict[str, dict[str, int]] = {
    "Cleangirl":  {"cream": 2, "white": 1, "grey": 1, "brown": 1},
    "Formal":     {"black": 2, "navy": 2, "grey": 1, "white": 1, "cream": 1, "brown": 1},
    "Hyperpop":   {"vivid": 3},
    "Softgirl":   {"pastel": 2, "white": 1, "cream": 1},
    "Streetwear": {"black": 2, "olive": 2, "grey": 1, "white": 1, "denim": 1, "brown": 1},
    "Y2K":        {"lightdenim": 2, "pastel": 1, "vivid": 1, "denim": 1},
}
ITEM_WEIGHT = 0.6        # น้ำหนักของชิ้นเสื้อผ้า
COLOR_WEIGHT = 0.4       # น้ำหนักของสี (ตั้ง 0 = ไม่ใช้สี)
SIGNATURE = 3            # น้ำหนักที่นับเป็น "ชิ้นเอกลักษณ์" (★)

# ค่าที่ปรับได้
TIE_MARGIN = 10.0        # สองสไตล์ห่างกันน้อยกว่านี้ (เปอร์เซ็นต์) = ผสม
DUP_IOU = 0.6            # กรอบหมวดเดียวกันที่ซ้อนกันเกินนี้ถือว่าเป็นชิ้นเดียวกัน
DUP_INSIDE = 0.7         # กรอบเล็กที่อยู่ในกรอบใหญ่หมวดเดียวกันเกินนี้ (เช่น กรอบแค่แขนเสื้อ) ก็ถือว่าชิ้นเดียวกัน
MIXMATCH_TOP_N = 12
MAX_PER_SIGNATURE = 2    # ชุดที่ใช้คลาสเหมือนกันเป๊ะ แสดงได้กี่อัน

# ตารางที่คำนวณจาก STYLES / STYLE_COLORS (สร้างใหม่ด้วย configure())
_W: dict[str, dict[str, float]] = {}        # style -> {class: น้ำหนัก}
_CW: dict[str, dict[str, float]] = {}       # style -> {tone: น้ำหนัก}
_STYLE_SET: dict[str, set] = {}
_COLOR_SET: dict[str, set] = {}


def _as_weights(v) -> dict[str, float]:
    if isinstance(v, dict):
        return {k: float(w) for k, w in v.items() if w}
    out: dict[str, float] = {}
    for k in v:                              # list -> น้ำหนัก 1 (ซ้ำก็นับครั้งเดียว)
        out[k] = 1.0
    return out


def configure(styles=None, style_colors=None) -> None:
    """เปลี่ยน mapping ตอนรัน (ใช้ในเทสต์) แล้วสร้างตารางน้ำหนักใหม่
    รับได้ทั้ง {style: {class: น้ำหนัก}} และ {style: [class, ...]}"""
    global STYLES, STYLE_COLORS
    if styles is not None:
        STYLES = styles
    if style_colors is not None:
        STYLE_COLORS = style_colors
    _W.clear(); _CW.clear(); _STYLE_SET.clear(); _COLOR_SET.clear()
    for st, items in STYLES.items():
        _W[st] = _as_weights(items)
        _STYLE_SET[st] = set(_W[st])
        _CW[st] = _as_weights(STYLE_COLORS.get(st, {}))
        _COLOR_SET[st] = set(_CW[st])


def _items_of(st: str) -> list[str]:
    """ชิ้นของสไตล์ เรียงจากน้ำหนักมากไปน้อย"""
    return sorted(_W.get(st, {}), key=lambda c: -_W[st][c])


def check_mapping() -> list[str]:
    """คืนรายการปัญหาของ mapping (ว่าง = ใช้ได้)"""
    problems = []
    for st, items in STYLES.items():
        if isinstance(items, list):
            dup = sorted({c for c in items if items.count(c) > 1})
            if dup:
                problems.append(f"{st}: มีชิ้นซ้ำ {dup}")
        bad = [c for c in _W.get(st, {}) if c not in CATEGORY]
        if bad:
            problems.append(f"{st}: ชื่อคลาสไม่ถูกต้อง {bad}")
        bad_w = [c for c, w in (items.items() if isinstance(items, dict) else []) if w not in (1, 2, 3)]
        if bad_w:
            problems.append(f"{st}: น้ำหนักควรเป็น 1, 2 หรือ 3 {bad_w}")
        if not _CW.get(st):
            problems.append(f"{st}: ยังไม่ได้กำหนดโทนสีใน STYLE_COLORS")
        bad_g = [g for g in _CW.get(st, {}) if g not in COLOR_GROUPS]
        if bad_g:
            problems.append(f"{st}: ชื่อโทนสีไม่ถูกต้อง {bad_g}")
        if _W.get(st) and max(_W[st].values()) < SIGNATURE:
            problems.append(f"{st}: ไม่มีชิ้นเอกลักษณ์ (น้ำหนัก {SIGNATURE}) แยกจากสไตล์อื่นได้ยาก")
    return problems


configure()


# ---------------------------------------------------------------------------
# ชื่อคลาส: ทนต่อตัวพิมพ์เล็ก/ใหญ่ และ - _ ช่องว่าง  (tank-top == Tank_Top)
# ---------------------------------------------------------------------------
def _canon(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


_CANON = {_canon(n): n for n in CATEGORY}


def normalize_class(name: str) -> str | None:
    """คืนชื่อคลาสมาตรฐาน หรือ None ถ้าไม่รู้จัก"""
    return _CANON.get(_canon(name))


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def _area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inside(a: list[float], b: list[float]) -> float:
    """สัดส่วนของกรอบที่เล็กกว่า ที่อยู่ในอีกกรอบ"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    small = min(_area(a), _area(b))
    return inter / small if small > 0 else 0.0


def _same_piece(k: dict, d: dict) -> bool:
    if k["category"] != d["category"]:
        return False
    if iou(k["box"], d["box"]) > DUP_IOU:
        return True
    # ชิ้นที่ผู้ใช้เพิ่มเองใช้กรอบเต็มรูป จึงไม่เช็กแบบกรอบซ้อน
    return not (k.get("manual") or d.get("manual")) and inside(k["box"], d["box"]) > DUP_INSIDE


def prepare_detections(raw: list[dict]) -> list[dict]:
    """เติม category และตัดกรอบซ้ำในหมวดเดียวกัน (เช่น baby_tee กับ short_sleeve_top บนเสื้อตัวเดียว)
    เสื้อคลุมซ้อนเสื้อข้างใน ไม่ถือเป็นซ้ำ เพราะอยู่คนละหมวด"""
    dets = []
    for d in raw:
        cls = normalize_class(d["cls"])
        if cls is None:
            continue
        rec = {**d, "cls": cls, "category": CATEGORY[cls]}
        if d.get("rgb") is not None and not d.get("color"):
            rec["color"] = color_info(d["rgb"])
        dets.append(rec)

    dets.sort(key=lambda d: -d["conf"])
    kept: list[dict] = []
    for d in dets:
        k = next((k for k in kept if _same_piece(k, d)), None)
        if k is None:
            kept.append(d)
        elif _area(d["box"]) > _area(k["box"]) * 1.2:
            # ชื่อคลาสใช้ของกรอบที่มั่นใจกว่า แต่กรอบ/สีใช้ของกรอบที่ใหญ่กว่า (ได้ทั้งตัวเสื้อ ไม่ใช่แค่แขน)
            k["box"] = d["box"]
            for f in ("rgb", "color"):
                if d.get(f) is not None:
                    k[f] = d[f]
    return kept


# ---------------------------------------------------------------------------
# สี: จัดสี RGB เป็นกลุ่มโทน
# ---------------------------------------------------------------------------
def classify_color(rgb) -> str:
    """RGB (0-255) -> หนึ่งในโทนของ COLOR_GROUPS"""
    r, g, b = (max(0, min(255, int(c))) / 255 for c in rgb)
    h, sat, val = colorsys.rgb_to_hsv(r, g, b)
    hue = h * 360
    if val < 0.22:
        return "black"
    if 195 <= hue <= 235 and 0.15 <= sat <= 0.55 and val >= 0.4:
        return "lightdenim" if val > 0.78 else "denim"           # ฟ้าอมเทาแบบยีนส์
    if 200 <= hue <= 255 and sat > 0.3 and val < 0.55:
        return "navy"
    if sat < 0.12:
        if val > 0.86:
            return "white"
        return "grey" if val >= 0.3 else "black"
    if 15 <= hue <= 55 and sat < 0.45:
        return "cream" if val >= 0.6 else "brown"                # ครีม เบจ / น้ำตาล
    if 60 <= hue <= 110 and sat < 0.6 and val < 0.6:
        return "olive"
    if sat >= 0.6 and val >= 0.55:
        return "vivid"
    if val >= 0.7:
        return "pastel"
    if val < 0.45:
        return "brown" if hue <= 55 or hue >= 340 else "black"
    return "vivid" if sat >= 0.45 else "grey"


def color_info(rgb) -> dict:
    r, g, b = (max(0, min(255, int(c))) for c in rgb)
    group = classify_color((r, g, b))
    return {"rgb": [r, g, b], "hex": f"#{r:02x}{g:02x}{b:02x}", "group": group, "label": COLOR_GROUPS[group]}


# ---------------------------------------------------------------------------
# 3) Look check
# ---------------------------------------------------------------------------
def look_check(dets: list[dict]) -> dict:
    cats = {d["category"] for d in dets}
    has_top, has_bottom = "top" in cats, "bottom" in cats
    has_one, has_outer = "one_piece" in cats, "outer" in cats

    if not cats:
        return {"status": "none", "missing": [], "message": "ตรวจไม่พบเสื้อผ้า", "warning": None}

    if has_one or (has_top and has_bottom):
        warning = None
        if has_one and (has_top or has_bottom):
            warning = "เจอเดรสพร้อมเสื้อหรือกางเกงในรูปเดียวกัน อาจตรวจซ้ำ ลองดูกรอบอีกครั้ง"
        return {"status": "complete", "missing": [], "message": "ชุดครบ", "warning": warning}

    if has_top:
        return {"status": "incomplete", "missing": ["bottom"], "message": "ขาดท่อนล่าง", "warning": None}
    if has_bottom:
        return {"status": "incomplete", "missing": ["top"], "message": "ขาดท่อนบน", "warning": None}
    # เหลือแต่เสื้อคลุม
    return {"status": "incomplete", "missing": ["top", "bottom"],
            "message": "ขาดทั้งท่อนบนและท่อนล่าง", "warning": None}


# ---------------------------------------------------------------------------
# 4) Style scoring
# ---------------------------------------------------------------------------
def _distribution(raw: dict[str, float]) -> dict[str, float]:
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()} if total else {}


def score_styles(items: list[tuple]) -> dict:
    """items = [(class, conf), ...] หรือ [(class, conf, color_group), ...]
    -> เปอร์เซ็นต์ต่อสไตล์ + สไตล์ที่ชนะ (อาจเป็นผสม 2 สไตล์)

    คะแนน = ITEM_WEIGHT x สัดส่วนจากชิ้นเสื้อผ้า + COLOR_WEIGHT x สัดส่วนจากสี
    ถ้าไม่มีข้อมูลสีเลย ใช้ชิ้นเสื้อผ้าอย่างเดียว"""
    item_raw = {style: 0.0 for style in STYLES}
    color_raw = {style: 0.0 for style in STYLES}
    for it in items:
        cls, conf = it[0], it[1]
        group = it[2] if len(it) > 2 else None
        for style in STYLES:
            item_raw[style] += conf * _W[style].get(cls, 0)
            if group:
                color_raw[style] += conf * _CW[style].get(group, 0)

    item_d, color_d = _distribution(item_raw), _distribution(color_raw)
    if not item_d and not color_d:
        return {"scores": {s: 0.0 for s in STYLES}, "styles": [], "label": "ระบุสไตล์ไม่ได้"}
    if not item_d:          # ชิ้นไม่อยู่ในสไตล์ไหนเลย -> ไม่ตัดสินจากสีอย่างเดียว
        return {"scores": {s: 0.0 for s in STYLES}, "styles": [], "label": "ระบุสไตล์ไม่ได้"}
    if color_d and COLOR_WEIGHT > 0:
        wi, wc = ITEM_WEIGHT / (ITEM_WEIGHT + COLOR_WEIGHT), COLOR_WEIGHT / (ITEM_WEIGHT + COLOR_WEIGHT)
        mix = {s: wi * item_d.get(s, 0) + wc * color_d.get(s, 0) for s in STYLES}
    else:
        mix = item_d

    pct = {s: round(mix.get(s, 0) * 100, 1) for s in STYLES}
    ranked = sorted(pct.items(), key=lambda x: -x[1])
    (s1, p1), (s2, p2) = ranked[0], ranked[1]
    if p1 - p2 < TIE_MARGIN:
        styles = [s1, s2]
        label = f"{s1} × {s2}"
    else:
        styles = [s1]
        label = s1
    return {"scores": pct, "styles": styles, "label": label}


def style_colors() -> dict[str, list[dict]]:
    """โทนสีของแต่ละสไตล์ สำหรับหน้าเว็บ"""
    return {s: [{"group": g, "label": COLOR_GROUPS.get(g, g), "weight": w}
                for g, w in sorted(_CW.get(s, {}).items(), key=lambda x: -x[1])] for s in STYLES}


def style_catalog() -> dict[str, list[dict]]:
    """รายชื่อไอเทมของแต่ละสไตล์ เรียงตามน้ำหนัก สำหรับหน้าแนะนำสไตล์
    unique=True คือชิ้นเอกลักษณ์ (น้ำหนัก >= SIGNATURE) หน้าเว็บแสดงเป็น ★"""
    return {
        style: [{"cls": c, "category": CATEGORY[c], "weight": _W[style][c], "unique": _W[style][c] >= SIGNATURE}
                for c in _items_of(style) if c in CATEGORY]
        for style, items in STYLES.items()
    }


def _public(d: dict) -> dict:
    return {
        "cls": d["cls"],
        "category": d["category"],
        "conf": round(d["conf"], 3),
        "box": [round(v, 4) for v in d["box"]],
        **({"color": d["color"]} if d.get("color") else {}),
        **({"manual": True} if d.get("manual") else {}),
    }


def analyze_outfit(raw_dets: list[dict], index: int, filename: str = "", style_override: str | None = None) -> dict:
    """style_override = สไตล์ที่ผู้ใช้ระบุเอง (ใช้แทนผลที่คำนวณได้ แต่ยังเก็บคะแนนไว้ให้ดู)"""
    dets = prepare_detections(raw_dets)

    # สไตล์: ใช้ conf สูงสุดต่อคลาส
    found: dict[str, tuple] = {}
    for d in dets:                       # dets เรียงตาม conf มากไปน้อยแล้ว
        if d["cls"] not in found:
            g = d["color"]["group"] if d.get("color") else None
            found[d["cls"]] = (d["cls"], d["conf"], g) if g else (d["cls"], d["conf"])

    style = score_styles(list(found.values()))
    if style_override in STYLES:
        style = {**style, "styles": [style_override], "label": style_override, "manual": True}
    return {
        "index": index,
        "filename": filename,
        "detections": [_public(d) for d in dets],
        "look": look_check(dets),
        "style": style,
    }


# ---------------------------------------------------------------------------
# 5) Mix & match ข้ามชุด
# ---------------------------------------------------------------------------
def _verdict(score: int) -> str:
    if score >= 85:
        return "เข้ากันมาก"
    if score >= 60:
        return "เข้ากัน"
    if score >= 40:
        return "ลองดูได้"
    return "ไม่ค่อยเข้า"


def _evaluate(items: list[dict]) -> dict:
    """ให้คะแนนชุดที่ประกอบขึ้นใหม่

    style  = score_styles ของไอเทมทั้งชุด (ถ่วงน้ำหนักด้วย conf และสีถ้ามี)
    score  = (สัดส่วนชิ้นที่อยู่ในสไตล์ที่ชนะ ถ่วงกับ สัดส่วนชิ้นที่สีเข้ากับสไตล์ที่ชนะ) x ค่าความมั่นใจเฉลี่ย
             ถ้าผสมสองสไตล์ นับรวมทั้งสอง / ถ้าไม่มีข้อมูลสี ใช้ชิ้นอย่างเดียว
    """
    def tup(i):
        g = i["color"]["group"] if i.get("color") else None
        return (i["cls"], i["conf"], g) if g else (i["cls"], i["conf"])

    style = score_styles([tup(i) for i in items])
    if not style["styles"]:
        score = 0
    else:
        members = set().union(*(_STYLE_SET[s] for s in style["styles"]))
        coverage = sum(1 for i in items if i["cls"] in members) / len(items)
        colored = [i for i in items if i.get("color")]
        if colored and COLOR_WEIGHT > 0:
            tones = set().union(*(_COLOR_SET[s] for s in style["styles"]))
            color_fit = sum(1 for i in colored if i["color"]["group"] in tones) / len(colored)
            wi = ITEM_WEIGHT / (ITEM_WEIGHT + COLOR_WEIGHT)
            coverage = wi * coverage + (1 - wi) * color_fit
        mean_conf = sum(i["conf"] for i in items) / len(items)
        score = round(100 * coverage * (0.8 + 0.2 * mean_conf))
    return {"items": items, "style": style, "score": score, "verdict": _verdict(score)}


# ---------------------------------------------------------------------------
# 6) เป้าหมาย (เรฟ) : รูปเรฟ หรือสไตล์ที่ผู้ใช้เลือก
# ---------------------------------------------------------------------------
def target_from_ref(ref_outfit: dict) -> dict | None:
    """ใช้ผลตรวจรูปเรฟเป็นเป้าหมาย  (None ถ้าเรฟไม่มีชิ้นเสื้อผ้าที่รู้จัก)"""
    if not ref_outfit["style"]["styles"]:
        return None
    pieces = list(dict.fromkeys(d["cls"] for d in ref_outfit["detections"]))
    return {
        "source": "ref",
        "label": ref_outfit["style"]["label"],
        "styles": ref_outfit["style"]["styles"],
        "scores": ({st: (100.0 if st in ref_outfit["style"]["styles"] else 0.0) for st in STYLES}
                   if ref_outfit["style"].get("manual") else ref_outfit["style"]["scores"]),
        "pieces": pieces,
        "ref": ref_outfit,
    }


def target_from_style(style: str) -> dict | None:
    if style not in STYLES:
        return None
    return {
        "source": "style",
        "label": style,
        "styles": [style],
        "scores": {s: (100.0 if s == style else 0.0) for s in STYLES},
        "pieces": _items_of(style),
        "ref": None,
    }


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(a[k] * b.get(k, 0.0) for k in a)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def match_target(items: list[dict], style: dict, target: dict) -> int:
    """ความใกล้เรฟ 0-100 = 60% สไตล์ที่คล้ายกัน + 40% ชิ้นที่ตรงกับเรฟ
    ถ้าเป็นรูปเรฟ ชิ้นที่ตรงกับในเรฟเป๊ะได้เต็ม ชิ้นที่อยู่ในสไตล์ของเรฟได้ครึ่ง"""
    style_sim = _cosine(style["scores"], target["scores"])
    exact = set(target["pieces"])
    near = set().union(*(_STYLE_SET[s] for s in target["styles"])) if target["styles"] else set()
    piece_pts = [1.0 if i["cls"] in exact else 0.5 if i["cls"] in near else 0.0 for i in items]
    piece_sim = sum(piece_pts) / len(piece_pts) if piece_pts else 0.0
    return round(100 * (0.6 * style_sim + 0.4 * piece_sim))


# ---------------------------------------------------------------------------
# 7) Mix & match
# ---------------------------------------------------------------------------
def _pool(outfits: list[dict]) -> dict[str, list[dict]]:
    pool: dict[str, list[dict]] = {"top": [], "bottom": [], "one_piece": [], "outer": []}
    for o in outfits:
        for k, d in enumerate(o["detections"]):
            pool[d["category"]].append({**d, "outfit": o["index"], "id": f"{o['index']}-{k}"})
    return pool


def mix_and_match(outfits: list[dict], top_n: int = MIXMATCH_TOP_N, target: dict | None = None) -> list[dict]:
    """รวมทุกชิ้นที่ตรวจเจอจากทุกรูปเป็น 'ตู้เสื้อผ้า' แล้วจัดอันดับชุด

    - เก็บทุกชิ้น ไม่จำกัดหมวดละชิ้นต่อรูป และไม่สนว่าชิ้นมาจากรูปไหน
    - รูปแบบที่ลอง:  ท่อนบน + ท่อนล่าง  /  ชุดชิ้นเดียว   (+ เสื้อคลุมชิ้นไหนก็ได้ หรือไม่ใส่)
    - ไม่มีเป้าหมาย: เรียงตามคะแนนความเข้ากัน (score)
      มีเป้าหมาย (เรฟ/สไตล์ที่เลือก): เรียงตามความใกล้เรฟ (match) ก่อน แล้วค่อย score
    - ใส่เสื้อคลุมเฉพาะเมื่อทำให้อันดับดีขึ้น
    - ชุดที่ใช้คลาสเหมือนกันเป๊ะ แสดงได้ไม่เกิน MAX_PER_SIGNATURE อัน
    """
    pool = _pool(outfits)
    bases = [list(p) for p in product(pool["top"], pool["bottom"])] + [[op] for op in pool["one_piece"]]
    outer_options: list[dict | None] = [None, *pool["outer"]]

    def rank(c: dict) -> tuple:
        return (c.get("match", 0), c["score"])

    scored = []
    for base in bases:
        candidates = []
        for outer in outer_options:
            c = _evaluate(base + ([outer] if outer else []))
            if target:
                c["match"] = match_target(c["items"], c["style"], target)
            candidates.append(c)
        scored.append(max(candidates, key=lambda c: (*rank(c), -len(c["items"]))))

    scored.sort(key=lambda c: (*(-v for v in rank(c)), -sum(i["conf"] for i in c["items"]) / len(c["items"])))

    picked, seen = [], defaultdict(int)
    for c in scored:
        sig = frozenset(i["cls"] for i in c["items"])
        if seen[sig] >= MAX_PER_SIGNATURE:
            continue
        seen[sig] += 1
        picked.append(c)
        if len(picked) == top_n:
            break

    out = []
    for c in picked:
        row = {
            "items": [{**_public(i), "outfit": i["outfit"]} for i in c["items"]],
            "style": c["style"],
            "score": c["score"],
            "verdict": c["verdict"],
        }
        if target:
            row["match"] = c["match"]
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# 8) แนะนำชิ้นที่ควรหามาเพิ่ม
# ---------------------------------------------------------------------------
def _wanted(target: dict) -> list[str]:
    """ชิ้นที่อยากได้ เรียงตามความสำคัญ: ชิ้นในรูปเรฟก่อน แล้วชิ้นเอกลักษณ์ของสไตล์ แล้วชิ้นอื่นของสไตล์"""
    out = list(target["pieces"]) if target["source"] == "ref" else []
    style_items = [c for s in target["styles"] for c in _W[s]]
    style_items.sort(key=lambda c: -max(_W[s].get(c, 0) for s in target["styles"]))   # น้ำหนักมากก่อน
    out += style_items
    return list(dict.fromkeys(out))


def suggest_pieces(outfits: list[dict], target: dict | None, limit: int = 3) -> list[dict]:
    """บอกว่าควรหาชิ้นไหนเพิ่ม

    - หมวดที่ยังไม่มีเลย (เช่น อัปมาแต่เสื้อ) -> แนะนำชิ้นในหมวดนั้นที่เข้ากับเป้าหมาย
      ถ้าไม่มีเป้าหมาย ใช้สไตล์ของเสื้อผ้าที่ผู้ใช้มีเป็นเป้าหมายแทน
    - มีเป้าหมาย และหมวดนั้นมีของแล้วแต่ไม่มีชิ้นไหนตรงสไตล์เป้าหมาย -> แนะนำเช่นกัน
    """
    pool = _pool(outfits)
    owned = {cat: {i["cls"] for i in items} for cat, items in pool.items()}
    if not any(owned.values()):
        return []

    goal = target
    if goal is None:
        all_items = [(i["cls"], i["conf"]) for items in pool.values() for i in items]
        st = score_styles(all_items)
        goal = target_from_style(st["styles"][0]) if st["styles"] else None
        if goal is None:
            return []
        goal = {**goal, "styles": st["styles"]}
    wanted = _wanted(goal)
    goal_items = set().union(*(_STYLE_SET[s] for s in goal["styles"])) | set(goal["pieces"])
    has_one_piece = bool(owned["one_piece"])

    out = []
    for cat in ("top", "bottom"):
        mine = owned[cat]
        if not mine:
            if has_one_piece and target is None:
                continue                                   # มีเดรสแล้ว ไม่ขาดอะไร
            reason = "missing"
        elif target is not None and not (mine & goal_items):
            reason = "off_target"
        else:
            continue
        picks = [c for c in wanted if CATEGORY[c] == cat and c not in mine][:limit]
        if picks:
            out.append({"category": cat, "reason": reason, "classes": picks})

    # ไม่มีทั้งบนและล่าง แต่เป้าหมายมีเดรส -> เสนอเดรสเป็นอีกทาง
    if not owned["top"] and not owned["bottom"] and not has_one_piece:
        dress = [c for c in wanted if CATEGORY[c] == "one_piece"][:limit]
        if dress:
            out.append({"category": "one_piece", "reason": "missing", "classes": dress})
    return out


def analyze_closet(all_raw: list[list[dict]], filenames: list[str] | None = None,
                   ref_raw: list[dict] | None = None, target_style: str | None = None,
                   style_overrides: dict | None = None) -> dict:
    """all_raw[i] = raw detections ของรูปที่ i
    ref_raw = detections ของรูปเรฟ (ถ้ามี)  /  target_style = สไตล์ที่ผู้ใช้เลือก (ถ้ามี)
    ถ้าส่งมาทั้งคู่ ใช้รูปเรฟก่อน
    style_overrides = สไตล์ที่ผู้ใช้ระบุเอง {"0": "Y2K", "ref": "Softgirl"}"""
    filenames = filenames or [""] * len(all_raw)
    ov = {str(k): v for k, v in (style_overrides or {}).items()}
    outfits = [analyze_outfit(raw, i, filenames[i], ov.get(str(i))) for i, raw in enumerate(all_raw)]

    target, note, ref = None, None, None
    if ref_raw is not None:
        ref = analyze_outfit(ref_raw, -1, "ref", ov.get("ref"))
        target = target_from_ref(ref)
        if target is None:
            note = "Dobby หาเสื้อผ้าในรูปเรฟไม่เจอ จึงจัดลุคแบบปกติแทน กด \"แก้ให้ถูก\" ที่รูปเรฟเพื่อบอกเองได้"
            if target_style:
                target = target_from_style(target_style)
    elif target_style:
        target = target_from_style(target_style)
        if target is None:
            note = f"ไม่รู้จักสไตล์ {target_style} จึงจัดลุคแบบปกติแทน"

    public_target = None
    if target:
        public_target = {k: v for k, v in target.items() if k != "ref"}
        if target["ref"]:
            public_target["detections"] = target["ref"]["detections"]

    return {
        "outfits": outfits,
        "target": public_target,
        "target_note": note,
        "ref": ({k: v for k, v in ref.items() if k != "look"} if ref else None),   # ผลตรวจรูปเรฟ (ไว้แก้เองได้)
        "mixmatch": mix_and_match(outfits, target=target),
        "suggestions": suggest_pieces(outfits, target),
    }
