# 1. NVIDIA CUDA 12.1 베이스 이미지 사용
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 2. Python 3.10 및 OCR 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    libgl1 \
    libgomp1 \
    libglib2.0-0 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. 심볼릭 링크 설정(안전하게 덮어쓰기)
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# 4. 작업 디렉토리 설정
WORKDIR /app

# 5. 가상환경 생성 (/opt)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# 6. PaddlePaddle GPU + PaddleOCR 설치
# - torch 제거 (용량/시간 절감)
# - paddlepaddle-gpu는 cu118 인덱스 사용 (cu121에 해당 wheel이 없어서 빌드 실패하던 문제 회피)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "paddlepaddle-gpu>=3.0.0" -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ && \
    pip install --no-cache-dir "paddleocr==3.3.0" jupyterlab ipykernel py-cpuinfo opencv-python pymupdf pillow

# 7. Jupyter Lab 설정
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8889" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py

# 8. 포트 개방
EXPOSE 8889

# 9. Jupyter Lab 실행
CMD ["jupyter", "lab", "--allow-root"]
