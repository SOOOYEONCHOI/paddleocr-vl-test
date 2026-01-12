import os
import sys
import argparse
import time
import re
import traceback
from pathlib import Path

# [설정] 모델 소스 연결 확인 건너뛰기
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

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
    parser = argparse.ArgumentParser(description="PaddleOCR-VL Official Method Parser")
    parser.add_argument("--input", default="./data/sapair-pdf", help="입력 파일 경로")
    parser.add_argument("--output", default="./output/vl_result", help="결과 저장 최상위 경로")
    
    args = parser.parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output)

    # 대상 파일 수집
    targets = []
    if input_path.is_file():
        targets = [input_path]
    elif input_path.is_dir():
        extensions = ["*.pdf", "*.PDF", "*.png", "*.PNG", "*.jpg", "*.JPG"]
        for ext in extensions:
            targets.extend(list(input_path.glob(ext)))
        targets = sorted(list(set(targets)))
    else:
        print(f"[Error] 경로를 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    if not targets:
        print("[Warning] 처리할 파일이 없습니다.")
        return

    # 엔진 초기화
    print(f"Initializing PaddleOCRVL engine...")
    try:
        pipeline = PaddleOCRVL() 
        print(">> VL Engine initialized successfully.")
    except Exception as e:
        print(f"[Critical] 엔진 초기화 실패: {e}")
        traceback.print_exc()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 문서 처리 루프
    # -------------------------------------------------------------------------
    for idx, file_path in enumerate(targets):
        print(f"\n[{idx+1}/{len(targets)}] Processing: {file_path.name}")
        start_time = time.perf_counter()

        file_stem = safe_stem(file_path.stem)
        
        # 파일별 최상위 결과 폴더 (예: output/2024_보고서/)
        doc_save_dir = output_root / file_stem
        doc_save_dir.mkdir(parents=True, exist_ok=True)

        try:
            results = pipeline.predict(str(file_path))

            if not results:
                print("   [Info] No content detected.")
                continue

            # [추가 1] 전체 문서를 합치기 위한 버퍼
            full_doc_markdown = []
            full_doc_markdown.append(f"# {file_path.name} 분석 결과\n")

            # -------------------------------------------------------
            # 페이지별 순회 및 저장
            # -------------------------------------------------------
            for i, res in enumerate(results):
                page_num = i + 1
                
                # [추가 2] 페이지별 폴더 분리 (덮어쓰기 방지 및 관리 용이)
                # 예: output/2024_보고서/page_001/
                page_dir = doc_save_dir / f"page_{page_num:03d}"
                page_dir.mkdir(parents=True, exist_ok=True)

                # 1. 개별 페이지 저장 (라이브러리 메서드 사용)
                try:
                    res.save_to_json(save_path=str(page_dir))
                    res.save_to_markdown(save_path=str(page_dir))
                except Exception as e:
                    print(f"     [Warning] Page {page_num} save failed: {e}")
                    continue # 저장 실패 시 병합 건너뜀

                # 2. [핵심] 방금 저장된 MD 파일을 읽어서 통합본에 추가
                # 라이브러리가 생성한 md 파일을 찾습니다 (보통 *.md 하나만 생성됨)
                generated_mds = list(page_dir.glob("*.md"))
                
                if generated_mds:
                    # 첫 번째 md 파일을 읽음
                    page_content = generated_mds[0].read_text(encoding='utf-8')
                    
                    # 통합 버퍼에 추가 (페이지 구분선 포함)
                    full_doc_markdown.append(f"\n\n--- \n## Page {page_num}\n")
                    full_doc_markdown.append(page_content)
                else:
                    print(f"     [Warning] No MD file found in {page_dir}")

            # -------------------------------------------------------
            # [추가 3] 전체 통합 MD 파일 저장
            # -------------------------------------------------------
            if full_doc_markdown:
                merged_path = doc_save_dir / f"{file_stem}_combined.md"
                merged_path.write_text("\n".join(full_doc_markdown), encoding='utf-8')
                print(f"   [Success] Combined MD saved: {merged_path.name}")
            
            print(f"   [Done] Saved to: {doc_save_dir}")

        except Exception as e:
            print(f"   [Error] {file_path.name} 처리 중 실패: {e}")
            traceback.print_exc()

        elapsed = time.perf_counter() - start_time
        print(f"   [Time] Elapsed: {elapsed:.2f}s")

if __name__ == "__main__":
    main()