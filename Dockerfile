# 1. NVIDIA CUDA 12.1 베이스 이미지 사용
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 2. 시스템 패키지 설치 (OCR 및 이미지 처리 필수 라이브러리)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    libgl1 \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 파이썬 환경 및 가상환경 설정
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# 5. PaddlePaddle GPU + PaddleOCR 설치
# cu121 환경 호환성을 위해 cu118 wheel 인덱스를 사용
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "paddlepaddle-gpu>=3.0.0" -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ && \
    pip install --no-cache-dir "paddleocr==3.3.0" "paddlex[ocr]" \
    jupyterlab ipykernel py-cpuinfo opencv-python-headless pymupdf pillow

# 6. Jupyter Lab 설정 (포트 8890 지정)
RUN mkdir -p /app/.jupyter && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /app/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8890" >> /app/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /app/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /app/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> /app/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /app/.jupyter/jupyter_lab_config.py
ENV JUPYTER_CONFIG_DIR=/app/.jupyter

# 7. 소스 코드 복사 및 권한 설정
COPY . /app/
RUN chmod -R 755 /app

# 8. 포트 개방
EXPOSE 8890

# 9. 서비스 시작
CMD ["jupyter", "lab", "--config=/app/.jupyter/jupyter_lab_config.py", "--allow-root"]
