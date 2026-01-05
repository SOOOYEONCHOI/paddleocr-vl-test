import os
import sys
import argparse
import time
import re
import traceback
import json
import numpy as np # numpy 처리를 위해 임포트 (없으면 pip install numpy)
from pathlib import Path

# [설정] 모델 소스 연결 확인 건너뛰기
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

# 1. PaddleOCRVL 전용 클래스 임포트
try:
    from paddleocr import PaddleOCRVL
except ImportError:
    print("[Error] PaddleOCRVL 클래스를 찾을 수 없습니다.")
    sys.exit(1)

# -------------------------------------------------------------------------
# [해결책 1] JSON 저장 시 Numpy 배열과 객체를 처리하는 전용 인코더
# -------------------------------------------------------------------------
class PaddleCustomEncoder(json.JSONEncoder):
    def default(self, obj):
        # 1. Numpy 배열이면 리스트로 변환
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # 2. PaddleOCRVLBlock 같은 객체면 딕셔너리(__dict__)로 변환 시도
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        # 3. 그 외 알 수 없는 타입은 문자열로 변환
        return str(obj)

# -------------------------------------------------------------------------
# [해결책 2] 딕셔너리(.get)와 객체(.attribute) 모두에서 값을 꺼내는 함수
# -------------------------------------------------------------------------
def safe_get(obj, key, default=None):
    """
    obj가 dict면 obj.get(key)를, 
    obj가 class 객체면 getattr(obj, key)를 수행
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    else:
        return getattr(obj, key, default)

def safe_stem(name: str) -> str:
    name = re.sub(r"[^\w\-.가-힣 ]+", "_", name).strip()
    return name[:100]

def main():
    parser = argparse.ArgumentParser(description="PaddleOCR-VL Final Parser v2")
    parser.add_argument("--input", default="./data/ocr-test", help="입력 파일 경로")
    parser.add_argument("--output", default="./output/vl_result", help="결과 저장 경로")
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
        print(f"[Error] 경로 없음: {input_path}")
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
        # 의존성 에러 힌트
        if "dependency" in str(e).lower():
            print("Tip: pip install \"paddlex[ocr]\"")
        sys.exit(1)

    # 처리 루프
    for idx, file_path in enumerate(targets):
        print(f"\n[{idx+1}/{len(targets)}] Processing: {file_path.name}")
        start_time = time.perf_counter()

        file_stem = safe_stem(file_path.stem)
        save_dir = output_root / file_stem
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            results = pipeline.predict(str(file_path))

            if not results:
                print("   [Info] No content detected.")
                continue

            for i, res in enumerate(results):
                page_name = f"page_{i+1}"
                
                # -------------------------------------------------------
                # A. JSON 저장 (커스텀 인코더 적용)
                # -------------------------------------------------------
                json_path = save_dir / f"{page_name}.json"
                try:
                    # res 자체가 객체일 수 있으므로 __dict__ 변환 시도
                    data_to_save = res
                    if hasattr(res, 'json'): # 만약 json 메서드가 있다면 사용
                         data_to_save = res.json
                    
                    with open(json_path, 'w', encoding='utf-8') as f:
                        # cls=PaddleCustomEncoder 옵션이 핵심입니다.
                        json.dump(data_to_save, f, ensure_ascii=False, indent=4, cls=PaddleCustomEncoder)
                except Exception as e:
                    print(f"     [Warning] JSON Save failed: {e}")

                # -------------------------------------------------------
                # B. Markdown 저장 (safe_get 적용)
                # -------------------------------------------------------
                md_path = save_dir / f"{page_name}.md"
                try:
                    markdown_lines = []
                    
                    # 1. 데이터 리스트 추출
                    # res가 객체일 때와 딕셔너리일 때를 모두 고려
                    target_list = []
                    
                    # (1) res['parsing_res_list'] 또는 res.parsing_res_list 시도
                    res_list = safe_get(res, 'parsing_res_list')
                    
                    # (2) 없다면 res['res']['parsing_res_list'] 형태인지 확인
                    if not res_list:
                         inner_res = safe_get(res, 'res')
                         if inner_res:
                             res_list = safe_get(inner_res, 'parsing_res_list')

                    if res_list:
                        target_list = res_list

                    # 2. 구조화된 데이터 파싱
                    if target_list:
                        # [수정됨] safe_get을 사용하여 정렬 (AttributeError 방지)
                        target_list.sort(key=lambda x: safe_get(x, 'block_id', 9999) if safe_get(x, 'block_id') is not None else 9999)
                        
                        for block in target_list:
                            # [수정됨] safe_get 사용
                            label = safe_get(block, 'block_label', 'text')
                            content = safe_get(block, 'block_content', '')
                            
                            if not content: continue

                            if label == 'table':
                                markdown_lines.append(f"\n{content}\n")
                            elif label in ['header', 'footer']:
                                markdown_lines.append(f"\n_{content}_\n")
                            elif label in ['title', 'section_title']:
                                markdown_lines.append(f"\n## {content}\n")
                            else:
                                markdown_lines.append(f"{content}\n")
                    
                    # 3. 구조 데이터가 없는 경우 (단순 텍스트)
                    else:
                        # pred 또는 rec_text 찾기
                        simple_text = safe_get(res, 'pred')
                        if not simple_text:
                             simple_text = safe_get(res, 'rec_text')
                        
                        if simple_text:
                            markdown_lines.append(str(simple_text))
                        else:
                            # 아무 데이터도 못 찾았을 때 객체 정보 덤프
                            markdown_lines.append(f"\n{str(res)}")

                    # 파일 쓰기
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write("\n".join(markdown_lines))

                except Exception as e:
                    print(f"     [Warning] Markdown Generation failed: {e}")
                    traceback.print_exc()

            print(f"   [Done] Saved result to: {save_dir}")

        except Exception as e:
            print(f"   [Error] {file_path.name} 처리 중 실패: {e}")
            traceback.print_exc()

        elapsed = time.perf_counter() - start_time
        print(f"   [Time] Elapsed: {elapsed:.2f}s")

if __name__ == "__main__":
    main()