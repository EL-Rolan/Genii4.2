FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

WORKDIR /app

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN test -f /app/checkpoints/best.pt || (echo "ОШИБКА: нет checkpoints/best.pt" && exit 1)

RUN mkdir -p /data/input /data/output

RUN mkdir -p /tmp/smoke_in && python -c "import numpy as np; from PIL import Image; rng = np.random.default_rng(0); [Image.fromarray(rng.integers(0, 256, (128, 128), dtype=np.uint8)).save(f'/tmp/smoke_in/pic{i}.png') for i in range(1, 5)]" && python infer.py --input-dir /tmp/smoke_in --output-dir /tmp/smoke_out && rm -rf /tmp/smoke_in /tmp/smoke_out

ENTRYPOINT ["python", "infer.py", \
            "--input-dir", "/data/input", \
            "--output-dir", "/data/output", \
            "--checkpoint", "/app/checkpoints/best.pt"]
