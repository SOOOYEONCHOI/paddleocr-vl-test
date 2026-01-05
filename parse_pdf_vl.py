import os
import sys
import argparse
import time
import re
import traceback
import json # JSON 저장을 위해 필수
from pathlib import Path

# [경고문 해결] 모델 소스 연결 확인 건너뛰기
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
    parser = argparse.ArgumentParser(description="PaddleOCR-VL Dedicated Parser")
    parser.add_argument("--input", default="./data/ocr-test", help="입력 PDF 파일 또는 디렉토리 경로")
    parser.add_argument("--output", default="./output/vl_result", help="결과 저장 최상위 디렉토리")
    
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
        # 지원할 확장자 목록 정의
        extensions = ["*.pdf", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
        
        for ext in extensions:
            # 대소문자 구분 문제 해결을 위해 glob 패턴 사용
            # (리눅스 환경에서는 대소문자를 구분하므로 필요시 추가 로직 필요)
            targets.extend(list(input_path.glob(ext)))
            targets.extend(list(input_path.glob(ext.upper()))) # .PNG, .JPG 등도 포함
            
        # 중복 제거 (혹시 모를 경우 대비) 및 정렬
        targets = sorted(list(set(targets)))
    else:
        print(f"[Error] 경로를 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    if not targets:
        print(f"[Warning] 처리할 파일(PDF/이미지)이 '{input_path}'에 없습니다.")
        return

    # -------------------------------------------------------------------------
    # 3. PaddleOCRVL 엔진 초기화
    # -------------------------------------------------------------------------
    print(f"Initializing PaddleOCRVL engine...")
    
    pipeline = None
    try:
        pipeline = PaddleOCRVL()
        print(">> VL Engine initialized successfully.")

    except RuntimeError as e:
        error_msg = str(e)
        if "dependency error" in error_msg.lower() or "paddlex" in error_msg.lower():
            print("\n" + "="*60)
            print("[Critical] PaddleX 의존성 패키지가 누락되었습니다.")
            print("    pip install \"paddlex[ocr]\"")
            print("="*60 + "\n")
            sys.exit(1)
        else:
            print(f"[Critical] 엔진 초기화 중 런타임 오류 발생: {e}")
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

            # [수정된 저장 로직] 라이브러리 함수 대신 직접 저장
            for i, res in enumerate(results):
                page_name = f"page_{i+1}"
                
                # -------------------------------------------------------
                # 1) JSON 수동 저장 (안전한 방식)
                # -------------------------------------------------------
                json_path = save_dir / f"{page_name}.json"
                try:
                    # res 객체가 dict 형태가 아닐 수 있으므로 변환 시도
                    # PaddleX 결과 객체는 보통 리스트나 dict, 혹은 str으로 변환 가능
                    with open(json_path, 'w', encoding='utf-8') as f:
                        if hasattr(res, 'json'): # json 속성이 있다면 사용
                            json.dump(res.json, f, ensure_ascii=False, indent=4)
                        elif isinstance(res, (dict, list)):
                            json.dump(res, f, ensure_ascii=False, indent=4)
                        else:
                            # 만약 객체라면 문자열(str)로 변환하거나 속성 추출
                            # VL 모델 결과는 보통 'pred' 키에 텍스트가 있음
                            f.write(str(res)) 
                except Exception as e:
                    print(f"     [Warning] JSON Save failed for {page_name}: {e}")

                # -------------------------------------------------------
                # 2) Markdown 수동 저장 (텍스트 추출)
                # -------------------------------------------------------
                md_path = save_dir / f"{page_name}.md"
                try:
                    # PaddleOCR-VL 결과에서 마크다운 텍스트 추출 시도
                    # 보통 res['pred'] 혹은 res['rec_text'] 등에 담겨 있음
                    markdown_content = ""
                    if isinstance(res, dict):
                        markdown_content = res.get('pred', res.get('rec_text', ""))
                    elif hasattr(res, 'pred'):
                        markdown_content = res.pred
                    else:
                        # 객체 자체를 문자열로 변환하여 저장 (최후의 수단)
                        markdown_content = str(res)

                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                        
                except Exception as e:
                    print(f"     [Warning] Markdown Save failed for {page_name}: {e}")

            print(f"   [Done] Saved {len(results)} pages to: {save_dir}")

        except Exception as e:
            print(f"   [Error] {pdf_path.name} 처리 중 실패: {e}")
            traceback.print_exc()

        elapsed = time.perf_counter() - start_time
        print(f"   [Time] Elapsed: {elapsed:.2f}s")

if __name__ == "__main__":
    main()