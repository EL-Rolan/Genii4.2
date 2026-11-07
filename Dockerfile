FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# PyTorch только для CPU: сети на 128x128 GPU не нужен, а образ получается
# в разы меньше (~1 ГБ вместо ~5 ГБ с CUDA-колёсами).
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Если веса не попали в репозиторий, сборка должна упасть сразу и понятно,
# а не выдать образ, который ломается при запуске.
RUN test -f /app/checkpoints/best.pt || (echo "ОШИБКА: нет checkpoints/best.pt" && exit 1)

# Папки для входных и выходных данных (монтируются через -v).
RUN mkdir -p /data/input /data/output

# По умолчанию контейнер читает pic1.png ... pic4.png из /data/input
# и пишет ДОЭ, восстановленные изображения и метрики в /data/output.
# Дополнительные флаги дописываются после имени образа (например, --resize).
ENTRYPOINT ["python", "infer.py", \
            "--input-dir", "/data/input", \
            "--output-dir", "/data/output", \
            "--checkpoint", "/app/checkpoints/best.pt"]
