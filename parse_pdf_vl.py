import os
import sys
import argparse
import time
import re
import traceback
from pathlib import Path

# [경고문 해결] 모델 소스 연결 확인 건너뛰기 (실행 속도 향상)
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

# 1. PaddleOCRVL 전용 클래스 임포트
try:
    from paddleocr import PaddleOCRVL
except ImportError:
    print("[Error] PaddleOCRVL 클래스를 찾을 수 없습니다.")
    print("pip install \"paddleocr>=3.3.0\" 및 paddlepaddle 3.0 이상이 필요합니다.")
    sys.exit(1)

def safe_stem(name: str) -> str:
    """파일명에서 특수문자 제거 및 안전한 이름 생성"""
    name = re.sub(r"[^\w\-.가-힣 ]+", "_", name).strip()
    return name[:100]

def main():
    # -------------------------------------------------------------------------
    # 1. 인자 설정
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="PaddleOCR-VL Dedicated Parser")
    parser.add_argument("--input", default="./data/ocr-test", help="입력 PDF 파일 또는 디렉토리 경로")
    parser.add_argument("--output", default="./output/vl_result", help="결과 저장 최상위 디렉토리")
    
    args = parser.parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output)

    # -------------------------------------------------------------------------
    # 2. 대상 파일 수집
    # -------------------------------------------------------------------------
    if input_path.is_file():
        targets = [input_path]
    elif input_path.is_dir():
        targets = list(input_path.glob("*.pdf"))
    else:
        print(f"[Error] 경로를 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    if not targets:
        print("[Warning] 처리할 PDF 파일이 없습니다.")
        return

    # -------------------------------------------------------------------------
    # 3. PaddleOCRVL 엔진 초기화 (수정됨)
    # -------------------------------------------------------------------------
    print(f"Initializing PaddleOCRVL engine...")
    
    pipeline = None
    try:
        # [수정] lang 인자 제거. 
        # PaddleOCRVL은 기본적으로 use_gpu 등의 공통 인자만 허용하거나 
        # config 파일을 따릅니다.
        pipeline = PaddleOCRVL()
        print(">> VL Engine initialized successfully.")

    except RuntimeError as e:
        error_msg = str(e)
        if "dependency error" in error_msg.lower() or "paddlex" in error_msg.lower():
            print("\n" + "="*60)
            print("[Critical] PaddleX 의존성 패키지가 누락되었습니다.")
            print("다음 명령어를 터미널에 실행하여 추가 패키지를 설치해주세요:")
            print("\n    pip install \"paddlex[ocr]\"\n")
            print("="*60 + "\n")
            sys.exit(1)
        else:
            print(f"[Critical] 엔진 초기화 중 런타임 오류 발생: {e}")
            traceback.print_exc()
            sys.exit(1)
            
    except ValueError as e:
        print(f"[Critical] 인자 설정 오류 (ValueError): {e}")
        print(">> PaddleOCRVL 버전에 따라 지원하지 않는 인자가 포함되었을 수 있습니다.")
        sys.exit(1)

    except Exception as e:
        print(f"[Critical] 엔진 초기화 실패: {e}")
        traceback.print_exc()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 4. 문서 처리 루프
    # -------------------------------------------------------------------------
    for idx, pdf_path in enumerate(targets):
        print(f"\n[{idx+1}/{len(targets)}] Processing: {pdf_path.name}")
        start_time = time.perf_counter()

        pdf_stem = safe_stem(pdf_path.stem)
        save_dir = output_root / pdf_stem
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            # [파싱 수행]
            results = pipeline.predict(str(pdf_path))

            if not results:
                print("   [Info] No content detected.")
                continue

            for i, res in enumerate(results):
                page_name = f"page_{i+1}"
                
                # 1) JSON 저장
                try:
                    res.save_to_json(save_path=str(save_dir), model_name=page_name)
                except AttributeError:
                    import json
                    json_out = save_dir / f"{page_name}.json"
                    with open(json_out, 'w', encoding='utf-8') as f:
                        data_to_save = res if isinstance(res, (dict, list)) else str(res)
                        json.dump(data_to_save, f, ensure_ascii=False, indent=4)

                # 2) Markdown 저장
                try:
                    res.save_to_markdown(save_path=str(save_dir), model_name=page_name)
                except AttributeError:
                    pass 

            print(f"   [Done] Saved pages to: {save_dir}")

        except Exception as e:
            print(f"   [Error] {pdf_path.name} 처리 중 실패: {e}")
            traceback.print_exc()

        elapsed = time.perf_counter() - start_time
        print(f"   [Time] Elapsed: {elapsed:.2f}s")

if __name__ == "__main__":
    main()