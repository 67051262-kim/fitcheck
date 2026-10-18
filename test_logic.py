"""รันเทสต์ตรรกะโดยไม่ต้องโหลดโมเดล:  python test_logic.py"""
import fitcheck as fc


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
    uniq = {s: {i["cls"] for i in items if i["unique"]} for s, items in cat.items()}
    assert uniq["Softgirl"] == {"puff_sleeve_top", "mini_dress"}
    assert uniq["Y2K"] == {"low_rise_jeans", "low_rise_pants"}
    assert uniq["Cleangirl"] == set()


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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("all passed")
