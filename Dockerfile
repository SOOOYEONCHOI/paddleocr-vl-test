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

# 6. LLM 추출 관련 패키지 설치 (LangChain + Pydantic)
RUN pip install --no-cache-dir \
    langchain-core==0.3.68 \
    langchain-openai==0.3.27 \
    langchain-ollama==0.3.4 \
    langchain-community==0.3.27 \
    pydantic>=2.0.0 \
    python-dotenv==1.1.1

# 7. Jupyter Lab 설정 (포트 8890 지정)
RUN mkdir -p /root/.jupyter && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8890" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py
ENV JUPYTER_CONFIG_DIR=/root/.jupyter

# 8. 소스 코드 복사 및 권한 설정
COPY . /app/
RUN chmod -R 755 /app

# 9. 포트 개방
EXPOSE 8890

# 10. 서비스 시작
CMD ["jupyter", "lab", "--config=/root/.jupyter/jupyter_lab_config.py", "--allow-root"]
