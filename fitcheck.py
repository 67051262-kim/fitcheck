"""
Fit Check core logic
--------------------
ไม่ผูกกับ YOLO / FastAPI เลย รับเป็น list ของ detection แล้วคืนผลวิเคราะห์
จึงเทสต์ได้โดยไม่ต้องโหลดโมเดล (ดู test_logic.py)

detection = {"cls": "baby_tee", "conf": 0.87, "box": [x1, y1, x2, y2]}   # box เป็นสัดส่วน 0-1
"""
from __future__ import annotations

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
# ---------------------------------------------------------------------------
STYLES: dict[str, list[str]] = {
    "Cleangirl":  ["tank-top", "long_pants", "suit_jacket", "short_sleeve_top"],
    "Formal":     ["suit_jacket", "long_pants", "long_dress", "long_skirt"],
    "Hyperpop":   ["Tube_top", "mini_skirt", "baby_tee", "mini_pants"],
    "Softgirl":   ["puff_sleeve_top", "long_skirt", "mini_dress", "baby_tee"],
    "Streetwear": ["baggy_pants", "short_sleeve_top", "tank-top", "long_sleeve"],
    "Y2K":        ["low_rise_jeans", "low_rise_pants", "baby_tee", "Tube_top", "mini_skirt"],
}

# ค่าที่ปรับได้
TIE_MARGIN = 10.0        # สองสไตล์ห่างกันน้อยกว่านี้ (เปอร์เซ็นต์) = ผสม
DUP_IOU = 0.6            # กรอบหมวดเดียวกันที่ซ้อนกันเกินนี้ถือว่าเป็นชิ้นเดียวกัน
MIXMATCH_TOP_N = 12
MAX_PER_SIGNATURE = 2    # ชุดที่ใช้คลาสเหมือนกันเป๊ะ แสดงได้กี่อัน

# น้ำหนักไอเทม = 1 / จำนวนสไตล์ที่ไอเทมนั้นอยู่
_STYLE_COUNT: dict[str, int] = defaultdict(int)
for _items in STYLES.values():
    for _i in _items:
        _STYLE_COUNT[_i] += 1
_STYLE_SET = {s: set(v) for s, v in STYLES.items()}


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


def prepare_detections(raw: list[dict]) -> list[dict]:
    """เติม category และตัดกรอบซ้ำในหมวดเดียวกัน (เช่น baby_tee กับ short_sleeve_top บนเสื้อตัวเดียว)
    เสื้อคลุมซ้อนเสื้อข้างใน ไม่ถือเป็นซ้ำ เพราะอยู่คนละหมวด"""
    dets = []
    for d in raw:
        cls = normalize_class(d["cls"])
        if cls is None:
            continue
        dets.append({**d, "cls": cls, "category": CATEGORY[cls]})

    dets.sort(key=lambda d: -d["conf"])
    kept: list[dict] = []
    for d in dets:
        if any(k["category"] == d["category"] and iou(k["box"], d["box"]) > DUP_IOU for k in kept):
            continue
        kept.append(d)
    return kept


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
def score_styles(items: list[tuple[str, float]]) -> dict:
    """items = [(class, conf), ...]  ->  เปอร์เซ็นต์ต่อสไตล์ + สไตล์ที่ชนะ (อาจเป็นผสม 2 สไตล์)"""
    raw = {
        style: sum(conf / _STYLE_COUNT[cls] for cls, conf in items if cls in members)
        for style, members in _STYLE_SET.items()
    }
    total = sum(raw.values())
    if total == 0:
        return {"scores": {s: 0.0 for s in STYLES}, "styles": [], "label": "ระบุสไตล์ไม่ได้"}

    pct = {s: round(v / total * 100, 1) for s, v in raw.items()}
    ranked = sorted(pct.items(), key=lambda x: -x[1])
    (s1, p1), (s2, p2) = ranked[0], ranked[1]
    if p1 - p2 < TIE_MARGIN:
        styles = [s1, s2]
        label = f"{s1} × {s2}"
    else:
        styles = [s1]
        label = s1
    return {"scores": pct, "styles": styles, "label": label}


def style_catalog() -> dict[str, list[dict]]:
    """รายชื่อไอเทมของแต่ละสไตล์ สำหรับหน้าแนะนำสไตล์
    unique=True คือไอเทมที่อยู่ในสไตล์นี้สไตล์เดียว (น้ำหนักสูงสุดตอนให้คะแนน)"""
    return {
        style: [{"cls": c, "category": CATEGORY[c], "unique": _STYLE_COUNT[c] == 1} for c in items]
        for style, items in STYLES.items()
    }


def _public(d: dict) -> dict:
    return {
        "cls": d["cls"],
        "category": d["category"],
        "conf": round(d["conf"], 3),
        "box": [round(v, 4) for v in d["box"]],
    }


def analyze_outfit(raw_dets: list[dict], index: int, filename: str = "") -> dict:
    dets = prepare_detections(raw_dets)

    # สไตล์: ใช้ conf สูงสุดต่อคลาส
    found: dict[str, float] = {}
    for d in dets:
        found[d["cls"]] = max(found.get(d["cls"], 0.0), d["conf"])

    return {
        "index": index,
        "filename": filename,
        "detections": [_public(d) for d in dets],
        "look": look_check(dets),
        "style": score_styles(list(found.items())),
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

    style  = score_styles ของไอเทมทั้งชุด (ถ่วงน้ำหนักด้วย conf)
    score  = สัดส่วนไอเทมที่อยู่ในสไตล์ที่ชนะ (ถ้าผสมนับรวมทั้งสองสไตล์) x ค่าความมั่นใจเฉลี่ย
    """
    style = score_styles([(i["cls"], i["conf"]) for i in items])
    if not style["styles"]:
        score = 0
    else:
        members = set().union(*(_STYLE_SET[s] for s in style["styles"]))
        coverage = sum(1 for i in items if i["cls"] in members) / len(items)
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
        "scores": ref_outfit["style"]["scores"],
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
        "pieces": list(STYLES[style]),
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
    style_items = [c for s in target["styles"] for c in STYLES[s]]
    style_items.sort(key=lambda c: _STYLE_COUNT[c])          # ชิ้นที่อยู่น้อยสไตล์ = เอกลักษณ์กว่า
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
                   ref_raw: list[dict] | None = None, target_style: str | None = None) -> dict:
    """all_raw[i] = raw detections ของรูปที่ i
    ref_raw = detections ของรูปเรฟ (ถ้ามี)  /  target_style = สไตล์ที่ผู้ใช้เลือก (ถ้ามี)
    ถ้าส่งมาทั้งคู่ ใช้รูปเรฟก่อน"""
    filenames = filenames or [""] * len(all_raw)
    outfits = [analyze_outfit(raw, i, filenames[i]) for i, raw in enumerate(all_raw)]

    target, note = None, None
    if ref_raw is not None:
        ref = analyze_outfit(ref_raw, -1, "ref")
        target = target_from_ref(ref)
        if target is None:
            note = "AI หาชิ้นเสื้อผ้าในรูปเรฟไม่เจอ จึงจัดลุคแบบปกติแทน"
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
        "mixmatch": mix_and_match(outfits, target=target),
        "suggestions": suggest_pieces(outfits, target),
    }
