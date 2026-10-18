FROM python:3.11-slim

# ไลบรารีระบบที่ OpenCV (ซึ่ง ultralytics ใช้) ต้องการ
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces รันคอนเทนเนอร์ด้วย user uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/home/user/app/best.pt

WORKDIR /home/user/app

# ติดตั้ง PyTorch รุ่น CPU ก่อน (เล็กกว่ารุ่น CUDA หลาย GB และ Space ฟรีไม่มี GPU อยู่แล้ว)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
