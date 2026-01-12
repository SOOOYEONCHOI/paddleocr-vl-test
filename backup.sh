#!/bin/bash

# 1. 경로 설정
PROJECT_DIR="$(pwd)"
BACKUP_ROOT="$PROJECT_DIR/backup/archives"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="paddle_ocr_latest_$TIMESTAMP.tar.gz"

mkdir -p "$BACKUP_ROOT"

# 2. 새로운 백업 생성
tar -czf "$BACKUP_ROOT/$BACKUP_FILE" \
    --exclude="./data" \
    --exclude="./output" \
    --exclude="./backup" \
    --exclude="./models" \
    --exclude="./.git" \
    --exclude="./__pycache__" \
    -C "$PROJECT_DIR" .

# 3. 최신 1개 파일만 남기고 이전 파일 삭제
cd "$BACKUP_ROOT" && ls -t paddle_ocr_latest_*.tar.gz | tail -n +2 | xargs rm -f -- 2>/dev/null

echo "------------------------------------------"
echo "PaddleOCR 백업 완료 (8890 포트 버전): $BACKUP_FILE"
echo "최신 1개 파일만 유지됩니다."
echo "------------------------------------------"