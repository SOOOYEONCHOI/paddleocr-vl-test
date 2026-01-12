# GPU 지원을 위한 CUDA 베이스 이미지 사용
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 Python 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Python 심볼릭 링크 및 가상환경 생성
RUN ln -s /usr/bin/python3 /usr/bin/python 
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# 5. Python 의존성 설치 (Docling 및 Jupyter 포함)
COPY requirements.txt . 
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# 6. 애플리케이션 코드 및 폴더 구조 생성
COPY app/ ./app/
COPY start.sh . 
RUN chmod +x start.sh
RUN mkdir -p uploads outputs results

# 7. 환경 변수 및 포트 노출
EXPOSE 8008 8888
ENV PORT=8008
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 8. 서비스 시작 (start.sh 호출)
# 애플리케이션 실행 (FastAPI + Jupyter Lab)
CMD ["./start.sh"]