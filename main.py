"""
Fit Check API
รัน:  uvicorn main:app --reload
เปิด: http://127.0.0.1:8000
"""
from __future__ import annotations

import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

import colors
import fitcheck

log = logging.getLogger("fitcheck")

BASE = Path(__file__).parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE / "best.pt"))
IMGSZ = int(os.getenv("IMGSZ", "640"))
MAX_MB = float(os.getenv("MAX_MB", "10"))
MAX_FILES = int(os.getenv("MAX_FILES", "12"))
STYLE_IMG_DIR = BASE / "static" / "styles"      # รูปตัวอย่างสไตล์: static/styles/<ชื่อสไตล์>/*.jpg
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

_state: dict = {"model": None, "unknown_classes": []}
_lock = threading.Lock()  # YOLO predict ไม่ thread-safe


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"ไม่พบไฟล์โมเดล: {MODEL_PATH}  (วาง best.pt ไว้ข้าง main.py หรือกำหนด MODEL_PATH)")
    model = YOLO(str(MODEL_PATH))
    names = list(model.names.values())
    unknown = [n for n in names if fitcheck.normalize_class(n) is None]
    missing = [c for c in fitcheck.CATEGORY if c not in {fitcheck.normalize_class(n) for n in names}]
    for problem in fitcheck.check_mapping():
        log.warning("mapping สไตล์: %s", problem)
    if unknown:
        log.warning("คลาสในโมเดลที่ไม่อยู่ใน mapping (จะถูกข้าม): %s", unknown)
    if missing:
        log.warning("คลาสใน mapping ที่โมเดลไม่มี: %s", missing)
    _state.update(model=model, unknown_classes=unknown)
    yield


app = FastAPI(title="Fit Check", lifespan=lifespan)


def _read_image(upload: UploadFile) -> Image.Image:
    data = upload.file.read(int(MAX_MB * 1024 * 1024) + 1)
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"{upload.filename}: ไฟล์ใหญ่เกิน {MAX_MB:g} MB")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)  # หมุนตาม EXIF ให้ตรงกับที่เบราว์เซอร์แสดง
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, f"{upload.filename}: เปิดเป็นรูปภาพไม่ได้")


def _detect(img: Image.Image, conf: float) -> list[dict]:
    with _lock:
        res = _state["model"].predict(img, conf=conf, imgsz=IMGSZ, verbose=False)[0]
    out = []
    for b in res.boxes:
        box = [float(v) for v in b.xyxyn[0].tolist()]
        out.append({
            "cls": res.names[int(b.cls[0])],
            "conf": float(b.conf[0]),
            "box": box,
            "rgb": colors.dominant_color(img, box),   # สีหลักของชิ้น ใช้ช่วยตัดสินสไตล์
        })
    return out


@app.post("/api/analyze")
def analyze(
    files: list[UploadFile] = File(...),
    ref: Optional[UploadFile] = File(None, description="รูปเรฟ (ไม่บังคับ)"),
    target_style: Optional[str] = Form(None, description="สไตล์ที่อยากได้ (ไม่บังคับ) เช่น Y2K"),
    conf: float = Query(0.45, ge=0.1, le=0.95, description="ค่าความมั่นใจขั้นต่ำของ YOLO"),
):
    if not 1 <= len(files) <= MAX_FILES:
        raise HTTPException(400, f"ส่งรูปได้ 1-{MAX_FILES} รูป")

    all_raw, names = [], []
    for f in files:
        all_raw.append(_detect(_read_image(f), conf))
        names.append(f.filename or "")

    ref_raw = _detect(_read_image(ref), conf) if ref is not None and ref.filename else None
    result = fitcheck.analyze_closet(all_raw, names, ref_raw=ref_raw, target_style=target_style or None)
    result["conf"] = conf
    result["unknown_classes"] = _state["unknown_classes"]
    return result


@app.post("/api/rescore")
def rescore(payload: dict = Body(...)):
    """คำนวณผลใหม่จากชิ้นที่ผู้ใช้แก้เอง โดยไม่ต้องรัน YOLO ซ้ำ
    payload = {"outfits": [[{cls, conf, box, rgb}, ...], ...], "names": [...],
               "ref": [...] | null, "styles": {"0": "Y2K", "ref": "Softgirl"}}"""
    outfits = payload.get("outfits")
    if not isinstance(outfits, list) or not 1 <= len(outfits) <= MAX_FILES:
        raise HTTPException(400, f"ส่งได้ 1-{MAX_FILES} รูป")

    def clean(dets):
        out = []
        for d in dets or []:
            try:
                box = [float(v) for v in d.get("box", [0, 0, 1, 1])][:4]
                rgb = d.get("rgb")
                out.append({"cls": str(d["cls"]), "conf": max(0.0, min(1.0, float(d.get("conf", 1.0)))),
                            "box": box if len(box) == 4 else [0, 0, 1, 1],
                            "rgb": [int(c) for c in rgb][:3] if rgb else None,
                            "manual": bool(d.get("manual"))})
            except (KeyError, TypeError, ValueError):
                continue
        return out

    ref = payload.get("ref")
    result = fitcheck.analyze_closet([clean(o) for o in outfits], payload.get("names"),
                                     ref_raw=clean(ref) if ref is not None else None,
                                     style_overrides=payload.get("styles") or {})
    result["unknown_classes"] = _state["unknown_classes"]
    return result


@app.get("/api/styles")
def styles():
    return fitcheck.style_catalog()


@app.get("/api/style-colors")
def style_colors():
    return fitcheck.style_colors()


@app.get("/api/style-images")
def style_images():
    """รูปตัวอย่างที่วางไว้ใน static/styles/<ชื่อสไตล์>/ (สไตล์ไหนไม่มีรูป หน้าเว็บจะใช้ภาพประกอบแทน)"""
    out: dict[str, list[str]] = {}
    for style in fitcheck.STYLES:
        folder = STYLE_IMG_DIR / style
        if folder.is_dir():
            files = sorted(f.name for f in folder.iterdir() if f.suffix.lower() in IMG_EXT)
            if files:
                out[style] = [f"/static/styles/{style}/{name}" for name in files]
    return out


@app.get("/api/health")
def health():
    return {"ok": _state["model"] is not None, "model": MODEL_PATH.name,
            "classes": list(_state["model"].names.values()) if _state["model"] else []}


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
