import logging
import sys
import time
import json
import os
from pathlib import Path
from functools import wraps

def setup_logger(name="PDFExtractor", log_file="process.log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 파일 핸들러
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 콘솔 핸들러
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def get_logger(name="PDFExtractor"):
    return logging.getLogger(name)

def log_execution_time(logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            logger.info(f"Starting {func.__name__}...")
            try:
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                logger.info(f"Finished {func.__name__} in {elapsed:.4f} seconds.")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise e
        return wrapper
    return decorator

def save_result(data: dict, input_path: str):
    """추출된 데이터를 outputs 폴더에 JSON 파일로 저장합니다."""
    # input_path의 파일명만 가져옴
    p = Path(input_path)
    filename = p.stem + "_result.json"

    # outputs 디렉토리 설정 (프로젝트 루트 기준)
    output_dir = Path("outputs")

    # outputs 폴더가 없으면 생성
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_path

def save_markdown(content: str, input_path: str):
    """파싱된 Markdown 내용을 outputs 폴더에 저장합니다."""
    p = Path(input_path)
    filename = p.stem + "_parsed.md"

    output_dir = Path("outputs/raw")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path
