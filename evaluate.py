"""
วัดความแม่นของการทายสไตล์ ด้วยรูปจริงที่รู้คำตอบอยู่แล้ว

เตรียมรูปแบบนี้ (ชื่อโฟลเดอร์ = ชื่อสไตล์ใน fitcheck.STYLES):
    test_photos/
        Cleangirl/  a.jpg b.jpg ...
        Y2K/        ...
รัน:
    python evaluate.py test_photos
    python evaluate.py test_photos --conf 0.45 --csv results.csv

ผลที่ได้: ถูกเป๊ะกี่ %, ถูกแบบลุคผสมกี่ %, แยกตามสไตล์, ตารางว่าสไตล์ไหนโดนทายเป็นสไตล์ไหน (confusion matrix)
และรายชื่อรูปที่ทายผิด พร้อมชิ้นและสีที่ AI เห็น เอาไว้ปรับน้ำหนักใน fitcheck.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fitcheck

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
NONE = "ไม่รู้"


def collect(folder: Path) -> list[tuple[Path, str]]:
    photos, skipped = [], []
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        if sub.name not in fitcheck.STYLES:
            skipped.append(sub.name)
            continue
        photos += [(f, sub.name) for f in sorted(sub.iterdir()) if f.suffix.lower() in IMG_EXT]
    if skipped:
        print(f"ข้ามโฟลเดอร์ที่ไม่ใช่ชื่อสไตล์: {', '.join(skipped)}")
    return photos


def run(photos: list[tuple[Path, str]], detect) -> list[dict]:
    """detect(path) -> raw detections (แบบเดียวกับที่ main.py ส่งให้ fitcheck)"""
    rows = []
    for i, (path, expected) in enumerate(photos, 1):
        out = fitcheck.analyze_outfit(detect(path), 0, path.name)
        st = out["style"]
        styles = st["styles"]
        if styles == [expected]:
            result = "ถูก"
        elif expected in styles:
            result = "ผสม"
        else:
            result = "ผิด"
        rows.append({
            "file": str(path), "expected": expected, "predicted": st["label"] if styles else NONE,
            "top": styles[0] if styles else NONE, "result": result,
            "pieces": " + ".join(d["cls"] for d in out["detections"]) or "-",
            "colors": " + ".join(d["color"]["label"] for d in out["detections"] if d.get("color")) or "-",
            "scores": " ".join(f"{k}:{v:.0f}" for k, v in sorted(st["scores"].items(), key=lambda x: -x[1])[:3]),
        })
        print(f"\r  ตรวจแล้ว {i}/{len(photos)}", end="", flush=True)
    print()
    return rows


def report(rows: list[dict]) -> None:
    n = len(rows)
    if not n:
        print("ไม่มีรูปให้ทดสอบ")
        return
    c = Counter(r["result"] for r in rows)
    print(f"\nทั้งหมด {n} รูป")
    print(f"  ถูกเป๊ะ        {c['ถูก']:4d}  ({c['ถูก'] / n:.0%})")
    print(f"  ถูกแบบลุคผสม   {c['ผสม']:4d}  ({c['ผสม'] / n:.0%})   <- คำตอบที่ถูกอยู่ในลุคผสม")
    print(f"  ผิด           {c['ผิด']:4d}  ({c['ผิด'] / n:.0%})")

    styles = list(fitcheck.STYLES)
    print("\nแยกตามสไตล์ (ถูกเป๊ะ / ถูกรวมลุคผสม)")
    by = defaultdict(list)
    for r in rows:
        by[r["expected"]].append(r)
    for s in styles:
        rs = by.get(s, [])
        if rs:
            ok = sum(r["result"] == "ถูก" for r in rs)
            part = sum(r["result"] in ("ถูก", "ผสม") for r in rs)
            print(f"  {s:11s} {ok:3d}/{len(rs):<3d} ({ok / len(rs):4.0%})   รวมผสม {part / len(rs):4.0%}")

    cols = styles + [NONE]
    print("\nตารางสับสน: แถว = คำตอบจริง, คอลัมน์ = สไตล์อันดับ 1 ที่ AI ทาย")
    w = max(len(s) for s in cols) + 1
    print(" " * 12 + "".join(f"{s[:w - 1]:>{w}}" for s in cols))
    for s in styles:
        if s in by:
            cnt = Counter(r["top"] for r in by[s])
            print(f"  {s:10s}" + "".join(f"{cnt.get(t, 0) or '.':>{w}}" for t in cols))

    wrong = [r for r in rows if r["result"] == "ผิด"]
    if wrong:
        print(f"\nรูปที่ทายผิด ({len(wrong)})")
        for r in wrong:
            print(f"  {Path(r['file']).name}: จริง {r['expected']} -> ทาย {r['predicted']}")
            print(f"      ชิ้น: {r['pieces']}   สี: {r['colors']}   คะแนน: {r['scores']}")


def yolo_detector(conf: float):
    from PIL import Image, ImageOps
    from ultralytics import YOLO
    import colors

    model = YOLO(str(Path(__file__).parent / "best.pt"))

    def detect(path: Path) -> list[dict]:
        with Image.open(path) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        res = model.predict(img, conf=conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            box = [float(v) for v in b.xyxyn[0].tolist()]
            out.append({"cls": res.names[int(b.cls[0])], "conf": float(b.conf[0]), "box": box,
                        "rgb": colors.dominant_color(img, box)})
        return out

    return detect


def main() -> None:
    ap = argparse.ArgumentParser(description="วัดความแม่นการทายสไตล์")
    ap.add_argument("folder", help="โฟลเดอร์รูปทดสอบ (มีโฟลเดอร์ย่อยตามชื่อสไตล์)")
    ap.add_argument("--conf", type=float, default=0.45, help="ความมั่นใจขั้นต่ำของ YOLO (ค่าเริ่มต้น 0.45)")
    ap.add_argument("--csv", help="บันทึกผลรายรูปเป็นไฟล์ CSV")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"ไม่พบโฟลเดอร์ {folder}")
    for p in fitcheck.check_mapping():
        print(f"เตือน mapping: {p}")
    photos = collect(folder)
    if not photos:
        sys.exit("ไม่พบรูป ตรวจว่าโฟลเดอร์ย่อยชื่อตรงกับสไตล์ เช่น test_photos/Y2K/1.jpg")

    print(f"กำลังโหลดโมเดลและตรวจ {len(photos)} รูป ...")
    rows = run(photos, yolo_detector(args.conf))
    report(rows)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:      # utf-8-sig ให้ Excel อ่านภาษาไทยได้
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nบันทึกผลรายรูปที่ {args.csv}")


if __name__ == "__main__":
    main()
