#!/bin/bash

# Python 캐시 삭제 (import 에러 방지)
find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find /app -name "*.pyc" -delete 2>/dev/null || true

# Jupyter Lab을 백그라운드에서 실행 (포트 8888)
jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --NotebookApp.allow_origin='*' \
    --NotebookApp.allow_remote_access=True \
    --ServerApp.allow_root=True \
    --notebook-dir=/app \
    > /tmp/jupyter.log 2>&1 &

# Jupyter Lab 시작 대기
sleep 2

# FastAPI를 포그라운드로 실행 (포트 8008) - 메인 프로세스로 유지
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8008

