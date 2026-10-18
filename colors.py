"""
หาสีหลักของเสื้อผ้าแต่ละชิ้นจากกรอบที่ YOLO ตรวจเจอ (ใช้แค่ Pillow)

วิธี: ตัดเฉพาะช่วงกลางกรอบ (ลดพื้นหลัง ผิว และขอบ) -> ย่อรูป -> แบ่งสีเป็นกลุ่ม -> เอากลุ่มที่ใหญ่ที่สุด
"""
from __future__ import annotations

from PIL import Image

INNER_X = (0.25, 0.75)   # ใช้ช่วงกลาง 50% ของความกว้างกรอบ
INNER_Y = (0.2, 0.8)     # และ 60% ของความสูง
PALETTE = 5              # แบ่งสีเป็นกี่กลุ่ม


def dominant_color(img: Image.Image, box: list[float]) -> list[int] | None:
    """img = รูปเต็ม (RGB), box = [x1, y1, x2, y2] เป็นสัดส่วน 0-1 -> [r, g, b] หรือ None"""
    W, H = img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    left = int((x1 + bw * INNER_X[0]) * W)
    right = int((x1 + bw * INNER_X[1]) * W)
    top = int((y1 + bh * INNER_Y[0]) * H)
    bottom = int((y1 + bh * INNER_Y[1]) * H)
    if right - left < 2 or bottom - top < 2:
        return None

    patch = img.crop((left, top, right, bottom)).convert("RGB")
    patch.thumbnail((64, 64))
    q = patch.quantize(colors=PALETTE, method=Image.Quantize.MEDIANCUT)
    counts = sorted(q.getcolors() or [], reverse=True)          # [(จำนวนพิกเซล, index), ...]
    if not counts:
        return None
    pal = q.getpalette()
    idx = counts[0][1]
    return pal[idx * 3: idx * 3 + 3]
