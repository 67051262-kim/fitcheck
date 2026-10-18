"""รันเทสต์ตรรกะโดยไม่ต้องโหลดโมเดล:  python test_logic.py"""
import fitcheck as fc

# mapping จริงที่ใช้ในเว็บ (เก็บไว้เช็คแยก)
REAL_STYLES = {k: (dict(v) if isinstance(v, dict) else list(v)) for k, v in fc.STYLES.items()}
REAL_COLORS = {k: (dict(v) if isinstance(v, dict) else list(v)) for k, v in fc.STYLE_COLORS.items()}

# เทสต์ตรรกะใช้ mapping คงที่ชุดนี้ จะได้ไม่พังทุกครั้งที่แก้ STYLES ใน fitcheck.py
TEST_STYLES = {
    "Cleangirl":  ["tank-top", "long_pants", "suit_jacket", "short_sleeve_top"],
    "Formal":     ["suit_jacket", "long_pants", "long_dress", "long_skirt"],
    "Hyperpop":   ["Tube_top", "mini_skirt", "baby_tee", "mini_pants"],
    "Softgirl":   ["puff_sleeve_top", "long_skirt", "mini_dress", "baby_tee"],
    "Streetwear": ["baggy_pants", "short_sleeve_top", "tank-top", "long_sleeve"],
    "Y2K":        ["low_rise_jeans", "low_rise_pants", "baby_tee", "Tube_top", "mini_skirt"],
}
TEST_COLORS = {
    "Cleangirl": ["cream", "white"], "Formal": ["black", "navy"], "Hyperpop": ["vivid"],
    "Softgirl": ["pastel", "white"], "Streetwear": ["black", "olive", "denim"], "Y2K": ["pastel", "lightdenim"],
}
fc.configure(TEST_STYLES, TEST_COLORS)


def det(cls, conf, box=(0.1, 0.1, 0.5, 0.5)):
    return {"cls": cls, "conf": conf, "box": list(box)}


def test_normalize():
    assert fc.normalize_class("tank-top") == "tank-top"
    assert fc.normalize_class("Tank_Top") == "tank-top"
    assert fc.normalize_class("tube_top") == "Tube_top"
    assert fc.normalize_class("hat") is None


def test_look_check():
    L = lambda *d: fc.analyze_outfit(list(d), 0)["look"]["status"]
    assert L() == "none"
    assert L(det("mini_dress", .9)) == "complete"
    assert L(det("baby_tee", .9, (0, 0, .4, .3)), det("mini_skirt", .8, (0, .5, .4, .9))) == "complete"
    assert L(det("baby_tee", .9)) == "incomplete"
    assert L(det("mini_skirt", .9)) == "incomplete"
    assert L(det("suit_jacket", .9)) == "incomplete"
    r = fc.analyze_outfit([det("mini_skirt", .9)], 0)["look"]
    assert r["missing"] == ["top"]
    r = fc.analyze_outfit([det("suit_jacket", .9)], 0)["look"]
    assert r["missing"] == ["top", "bottom"]


def test_duplicate_boxes_same_category_removed():
    # baby_tee กับ short_sleeve_top ซ้อนกันบนเสื้อตัวเดียว -> เหลือตัวที่ conf สูงกว่า
    r = fc.analyze_outfit([det("baby_tee", .9), det("short_sleeve_top", .6)], 0)
    assert [d["cls"] for d in r["detections"]] == ["baby_tee"]
    # แต่เสื้อคลุมทับเสื้อข้างในต้องอยู่ครบ
    r = fc.analyze_outfit([det("suit_jacket", .9), det("tank-top", .8)], 0)
    assert len(r["detections"]) == 2


def test_style_unique_items_win():
    s = fc.score_styles([("puff_sleeve_top", .9), ("mini_dress", .9)])
    assert s["styles"] == ["Softgirl"]
    s = fc.score_styles([("low_rise_jeans", .9), ("baby_tee", .9)])
    assert s["styles"][0] == "Y2K"


def test_style_tie_gives_mixed():
    s = fc.score_styles([("baby_tee", .9), ("mini_skirt", .9)])
    assert set(s["styles"]) == {"Hyperpop", "Y2K"}
    assert "×" in s["label"]


def test_style_none():
    assert fc.score_styles([])["styles"] == []


def test_mixmatch_any_item_any_photo():
    o1 = fc.analyze_outfit([det("baby_tee", .9, (0, 0, .5, .4)), det("long_pants", .8, (0, .5, .5, 1))], 0)
    o2 = fc.analyze_outfit([det("Tube_top", .85, (0, 0, .5, .4)), det("mini_skirt", .9, (0, .5, .5, 1))], 1)
    mm = fc.mix_and_match([o1, o2])
    classes = [frozenset(i["cls"] for i in c["items"]) for c in mm]
    # จับคู่ได้ทุกแบบ รวมชุดเดิมด้วย
    for pair in [{"baby_tee", "mini_skirt"}, {"Tube_top", "long_pants"},
                 {"baby_tee", "long_pants"}, {"Tube_top", "mini_skirt"}]:
        assert frozenset(pair) in classes, pair
    scores = [c["score"] for c in mm]
    assert scores == sorted(scores, reverse=True)


def test_mixmatch_flatlay_many_items_one_photo():
    # รูปเดียววางหลายชิ้น: ต้องเก็บครบ ไม่ใช่แค่ชิ้นเดียวต่อหมวด
    o = fc.analyze_outfit([
        det("baby_tee", .9, (0, 0, .2, .2)), det("puff_sleeve_top", .8, (.3, 0, .5, .2)),
        det("mini_skirt", .9, (0, .4, .2, .6)), det("long_skirt", .7, (.3, .4, .5, .9)),
    ], 0)
    assert len(o["detections"]) == 4
    mm = fc.mix_and_match([o])
    assert len(mm) == 4          # 2 บน x 2 ล่าง
    best = mm[0]
    assert best["score"] >= 85


def test_mixmatch_outer_and_onepiece():
    o1 = fc.analyze_outfit([det("mini_dress", .9, (0, 0, .5, 1))], 0)
    o2 = fc.analyze_outfit([det("suit_jacket", .9, (0, 0, .5, .5)), det("long_pants", .8, (0, .5, .5, 1)),
                            det("tank-top", .8, (.1, .1, .4, .4))], 1)
    mm = fc.mix_and_match([o1, o2])
    assert any({"mini_dress"} == {i["cls"] for i in c["items"]} for c in mm)
    # เสื้อคลุมถูกเลือกใส่เมื่อคะแนนดีขึ้น (Cleangirl: tank-top + long_pants + suit_jacket)
    assert any({"tank-top", "long_pants", "suit_jacket"} == {i["cls"] for i in c["items"]} for c in mm)


def test_mixmatch_cap_same_signature():
    # baby_tee + mini_skirt จากต่างรูป 4 ชิ้นต่อหมวด -> แสดงคลาสคู่เดียวกันไม่เกิน MAX_PER_SIGNATURE
    outs = [fc.analyze_outfit([det("baby_tee", .9, (0, 0, .5, .4)), det("mini_skirt", .9, (0, .5, .5, 1))], i)
            for i in range(4)]
    mm = fc.mix_and_match(outs)
    assert len(mm) == fc.MAX_PER_SIGNATURE


def test_mixmatch_needs_top_and_bottom():
    o1 = fc.analyze_outfit([det("baby_tee", .9), det("tank-top", .8, (.6, .6, .9, .9))], 0)
    assert fc.mix_and_match([o1]) == []


def test_style_catalog():
    cat = fc.style_catalog()
    assert set(cat) == set(fc.STYLES)
    # mapping แบบ list = ทุกชิ้นน้ำหนัก 1 ไม่มีชิ้นเอกลักษณ์
    assert all(i["weight"] == 1 and not i["unique"] for items in cat.values() for i in items)
    fc.configure({"A": {"tank-top": 3, "long_pants": 1}, "B": {"tank-top": 1, "baggy_pants": 3}}, {"A": ["cream"], "B": ["black"]})
    try:
        cat = fc.style_catalog()
        assert [i["cls"] for i in cat["A"]] == ["tank-top", "long_pants"]     # เรียงน้ำหนักมากก่อน
        assert cat["A"][0]["unique"] and not cat["A"][1]["unique"]
    finally:
        fc.configure(TEST_STYLES, TEST_COLORS)


def test_weights_decide():
    # ชิ้นเดียวกันอยู่ทั้งสองสไตล์ แต่น้ำหนักต่างกัน -> สไตล์ที่ให้ความสำคัญมากกว่าชนะ
    fc.configure({"A": {"tank-top": 3, "mini_pants": 1}, "B": {"tank-top": 1, "mini_pants": 3}},
                 {"A": ["cream"], "B": ["cream"]})
    try:
        assert fc.score_styles([("tank-top", .9)])["styles"] == ["A"]
        assert fc.score_styles([("mini_pants", .9)])["styles"] == ["B"]
        assert set(fc.score_styles([("tank-top", .9), ("mini_pants", .9)])["styles"]) == {"A", "B"}
    finally:
        fc.configure(TEST_STYLES, TEST_COLORS)


TOP = (0, 0, .5, .4)
BOT = (0, .5, .5, 1)


def test_ref_reorders_ranking():
    closet = [
        fc.analyze_outfit([det("tank-top", .9, TOP), det("baggy_pants", .9, BOT)], 0),
        fc.analyze_outfit([det("baby_tee", .9, TOP), det("low_rise_jeans", .9, BOT)], 1),
    ]
    raw = [o["detections"] for o in closet]
    plain = fc.analyze_closet(raw)
    assert plain["target"] is None and "match" not in plain["mixmatch"][0]
    ref = [det("Tube_top", .9, TOP), det("low_rise_jeans", .9, BOT)]       # เรฟ Y2K
    r = fc.analyze_closet(raw, ref_raw=ref)
    assert r["target"]["source"] == "ref" and "Y2K" in r["target"]["styles"]
    best = {i["cls"] for i in r["mixmatch"][0]["items"]}
    assert "low_rise_jeans" in best
    matches = [c["match"] for c in r["mixmatch"]]
    assert matches == sorted(matches, reverse=True)
    assert r["target"]["detections"]


def test_target_style_choice():
    closet = [fc.analyze_outfit([det("tank-top", .9, TOP), det("long_pants", .9, BOT),
                                 det("puff_sleeve_top", .9, (.5, 0, 1, .4)), det("long_skirt", .9, (.5, .5, 1, 1))], 0)]
    r = fc.analyze_closet([closet[0]["detections"]], target_style="Softgirl")
    assert {i["cls"] for i in r["mixmatch"][0]["items"]} == {"puff_sleeve_top", "long_skirt"}


def test_ref_without_clothes_falls_back():
    r = fc.analyze_closet([[det("baby_tee", .9, TOP), det("mini_skirt", .9, BOT)]], ref_raw=[])
    assert r["target"] is None and r["target_note"]
    assert r["mixmatch"]


def test_unknown_style_falls_back():
    r = fc.analyze_closet([[det("baby_tee", .9, TOP), det("mini_skirt", .9, BOT)]], target_style="Goth")
    assert r["target"] is None and r["target_note"]


def test_suggest_missing_bottom_only_top_uploaded():
    r = fc.analyze_closet([[det("puff_sleeve_top", .9, TOP)]])
    assert r["mixmatch"] == []
    sug = r["suggestions"]
    assert sug and sug[0]["category"] == "bottom" and sug[0]["reason"] == "missing"
    assert all(fc.CATEGORY[c] == "bottom" for c in sug[0]["classes"])
    assert "long_skirt" in sug[0]["classes"]          # ชิ้นท่อนล่างของ Softgirl


def test_suggest_follows_ref():
    r = fc.analyze_closet([[det("tank-top", .9, TOP)]], ref_raw=[det("baby_tee", .9, TOP), det("low_rise_jeans", .9, BOT)])
    bottom = next(s for s in r["suggestions"] if s["category"] == "bottom")
    assert bottom["classes"][0] == "low_rise_jeans"   # ชิ้นในรูปเรฟมาก่อน
    top = next(s for s in r["suggestions"] if s["category"] == "top")
    assert top["reason"] == "off_target"


def test_no_suggestions_when_complete():
    r = fc.analyze_closet([[det("baby_tee", .9, TOP), det("mini_skirt", .9, BOT)]])
    assert r["suggestions"] == []


CREAM, NEON_PINK, BLACK, PASTEL_PINK, DENIM, WHITE = (232, 220, 196), (255, 40, 170), (20, 20, 24), (248, 190, 214), (96, 128, 170), (250, 250, 250)


def test_classify_color():
    cases = {
        CREAM: "cream", (210, 190, 160): "cream", WHITE: "white", (128, 128, 128): "grey",
        BLACK: "black", (35, 45, 85): "navy", (90, 60, 40): "brown", (95, 105, 60): "olive",
        NEON_PINK: "vivid", (200, 255, 60): "vivid", PASTEL_PINK: "pastel", (200, 190, 240): "pastel",
        DENIM: "denim", (165, 195, 230): "lightdenim",
    }
    for rgb, want in cases.items():
        assert fc.classify_color(rgb) == want, (rgb, fc.classify_color(rgb), want)


def test_color_breaks_tie():
    # เสื้อกล้าม + กางเกงขาสั้น: ไม่มีสี -> เท่ากัน, มีสี -> สีเป็นตัวตัดสิน
    fc.configure({"A": ["tank-top", "mini_pants"], "B": ["tank-top", "mini_pants"]},
                 {"A": ["cream"], "B": ["vivid"]})
    try:
        plain = fc.score_styles([("tank-top", .9), ("mini_pants", .9)])
        assert set(plain["styles"]) == {"A", "B"}
        cream = fc.score_styles([("tank-top", .9, "cream"), ("mini_pants", .9, "cream")])
        neon = fc.score_styles([("tank-top", .9, "vivid"), ("mini_pants", .9, "vivid")])
        assert cream["styles"] == ["A"] and neon["styles"] == ["B"]
    finally:
        fc.configure(TEST_STYLES, TEST_COLORS)


def test_color_weight_zero_ignores_color():
    old = fc.COLOR_WEIGHT
    try:
        fc.COLOR_WEIGHT = 0
        a = fc.score_styles([("tank-top", .9, "vivid"), ("long_pants", .9, "vivid")])
        b = fc.score_styles([("tank-top", .9), ("long_pants", .9)])
        assert a["scores"] == b["scores"]
    finally:
        fc.COLOR_WEIGHT = old


def test_rgb_flows_into_outfit():
    o = fc.analyze_outfit([{"cls": "tank-top", "conf": .9, "box": [0, 0, .5, .4], "rgb": list(NEON_PINK)},
                           {"cls": "mini_pants", "conf": .9, "box": [0, .5, .5, 1], "rgb": list(NEON_PINK)}], 0)
    assert all(d["color"]["group"] == "vivid" for d in o["detections"])
    assert o["detections"][0]["color"]["hex"].startswith("#")
    assert "Hyperpop" in o["style"]["styles"]
    mm = fc.mix_and_match([o])
    assert mm and "color" in mm[0]["items"][0]


def test_real_mapping_is_valid():
    fc.configure(REAL_STYLES, REAL_COLORS)
    try:
        assert fc.check_mapping() == [], fc.check_mapping()
        for st, items in REAL_STYLES.items():
            cats = {fc.CATEGORY[c] for c in items if c in fc.CATEGORY}
            assert "one_piece" in cats or {"top", "bottom"} <= cats, f"{st} ต้องมีท่อนบน+ล่าง หรือเดรส"
    finally:
        fc.configure(TEST_STYLES, TEST_COLORS)


def test_style_override_and_manual_piece():
    r = fc.analyze_closet([[{"cls": "baggy_pants", "conf": 1, "box": [0, 0, 1, 1], "manual": True}], []],
                          style_overrides={"1": "Y2K"})
    assert r["outfits"][0]["detections"][0]["manual"] is True
    assert r["outfits"][1]["style"]["styles"] == ["Y2K"] and r["outfits"][1]["style"]["manual"]
    assert fc.analyze_outfit([], 0, "", "ไม่มีสไตล์นี้")["style"]["styles"] == []     # สไตล์ไม่รู้จัก ไม่เอา


def test_manual_ref_style_used_as_target():
    r = fc.analyze_closet([[det("baby_tee", .9, TOP), det("mini_skirt", .9, BOT)]], ref_raw=[],
                          style_overrides={"ref": "Softgirl"})
    assert r["target"]["label"] == "Softgirl" and r["target_note"] is None
    assert r["target"]["scores"]["Softgirl"] == 100.0 and "match" in r["mixmatch"][0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("all passed")
