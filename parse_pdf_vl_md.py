import os
import sys
import argparse
import time
import re
import traceback
from pathlib import Path

# [설정] 모델 소스 연결 확인 건너뛰기 (실행 속도 향상)
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

# 1. PaddleOCRVL 전용 클래스 임포트
try:
    from paddleocr import PaddleOCRVL
except ImportError:
    print("[Error] PaddleOCRVL 클래스를 찾을 수 없습니다.")
    sys.exit(1)

def safe_stem(name: str) -> str:
    """파일명에서 특수문자 제거 및 안전한 이름 생성"""
    name = re.sub(r"[^\w\-.가-힣 ]+", "_", name).strip()
    return name[:100]

def main():
    # -------------------------------------------------------------------------
    # 1. 인자 설정
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="PaddleOCR-VL Official Method Parser")
    parser.add_argument("--input", default="./data/sapair-pdf", help="입력 파일(PDF/이미지) 경로")
    parser.add_argument("--output", default="./output/vl_result", help="결과 저장 최상위 경로")
    
    args = parser.parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output)

    # -------------------------------------------------------------------------
    # 2. 대상 파일 수집
    # -------------------------------------------------------------------------
    targets = []
    if input_path.is_file():
        targets = [input_path]
    elif input_path.is_dir():
        # PDF 및 다양한 이미지 포맷 지원
        extensions = ["*.pdf", "*.PDF", "*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
        for ext in extensions:
            targets.extend(list(input_path.glob(ext)))
        targets = sorted(list(set(targets)))
    else:
        print(f"[Error] 경로를 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    if not targets:
        print("[Warning] 처리할 파일이 없습니다.")
        return

    # -------------------------------------------------------------------------
    # 3. PaddleOCRVL 엔진 초기화 (수정됨: 인자 제거)
    # -------------------------------------------------------------------------
    print(f"Initializing PaddleOCRVL engine...")
    pipeline = None
    try:
        # [수정] 사용자가 확인한 대로 인자 없이 초기화
        # 환경에 GPU가 있다면 PaddlePaddle이 자동으로 감지하여 사용합니다.
        pipeline = PaddleOCRVL() 
        print(">> VL Engine initialized successfully.")

    except Exception as e:
        print(f"[Critical] 엔진 초기화 실패: {e}")
        if "dependency" in str(e).lower():
            print("Tip: pip install \"paddlex[ocr]\"")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 4. 문서 처리 루프
    # -------------------------------------------------------------------------
    for idx, file_path in enumerate(targets):
        print(f"\n[{idx+1}/{len(targets)}] Processing: {file_path.name}")
        start_time = time.perf_counter()

        file_stem = safe_stem(file_path.stem)
        
        # [중요] 저장 경로는 '파일'이 아니라 '폴더'여야 합니다.
        # save_to_markdown 등이 이 폴더 안에 알아서 파일을 생성합니다.
        save_dir = output_root / file_stem
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 예측 수행
            results = pipeline.predict(str(file_path))

            if not results:
                print("   [Info] No content detected.")
                continue

            # -------------------------------------------------------
            # [핵심] 라이브러리 공식 내장 메서드로 저장
            # -------------------------------------------------------
            for res in results:
                # 1. JSON 저장
                try:
                    res.save_to_json(save_path=str(save_dir))
                except Exception as e:
                    print(f"     [Warning] JSON save failed: {e}")

                # 2. Markdown 저장 (라이브러리가 알아서 표/텍스트 파싱함)
                try:
                    res.save_to_markdown(save_path=str(save_dir))
                except Exception as e:
                    print(f"     [Warning] Markdown save failed: {e}")

            print(f"   [Done] Saved to: {save_dir}")

        except Exception as e:
            print(f"   [Error] {file_path.name} 처리 중 실패: {e}")
            traceback.print_exc()

        elapsed = time.perf_counter() - start_time
        print(f"   [Time] Elapsed: {elapsed:.2f}s")

if __name__ == "__main__":
    main()