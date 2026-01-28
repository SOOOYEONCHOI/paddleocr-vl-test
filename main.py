"""
PaddleOCR-VL + LLM 추출 통합 파이프라인

파이프라인:
    PDF 입력 → 페이지 필터링 → PaddleOCR-VL 파싱 → Markdown → LLM 추출 → 구조화된 JSON

사용법:
    python main.py --input ./inputs/sample.pdf --llm_type ollama
    python main.py --input ./inputs/ --llm_type openai --model gpt-4.1
    python main.py  # 기본값: inputs/ → outputs/
"""

import os
import sys
import argparse
import time
import re
import traceback
from pathlib import Path
from typing import List, Optional

import config
from extractor import extract_data_with_llm

# 모델 소스 연결 확인 건너뛰기
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

# 로거 설정
from utils import setup_logger, save_result, save_markdown


logger = setup_logger("PaddleOCR-LLM")

try:
    from paddleocr import PaddleOCRVL
except ImportError:
    logger.error("PaddleOCRVL 클래스를 찾을 수 없습니다.")
    sys.exit(1)

try:
    import fitz  # PyMuPDF
except ImportError:
    logger.warning("PyMuPDF not found. Page filtering will be disabled.")
    fitz = None


def safe_stem(name: str) -> str:
    """파일명에서 특수문자 제거 및 안전한 이름 생성"""
    name = re.sub(r"[^\w\-.가-힣 ]+", "_", name).strip()
    return name[:100]


def find_target_pages(file_path: Path, keyword: str = None) -> Optional[List[int]]:
    """
    PyMuPDF로 PDF를 빠르게 스캔하여 키워드가 포함된 페이지 번호 반환

    Args:
        file_path: PDF 파일 경로
        keyword: 검색할 키워드 (None이면 모든 페이지 반환)

    Returns:
        List[int]: 0-based 페이지 인덱스 리스트, 또는 None (필터링 불가 시)
    """
    if fitz is None:
        return None

    if keyword is None:
        keyword = config.PDF_TARGET_KEYWORD

    # 키워드에서 공백 제거 (검색 시 공백 무시)
    keyword_normalized = keyword.replace(" ", "")

    try:
        doc = fitz.open(str(file_path))
        target_pages = []

        total_pages = len(doc)

        for page_num in range(total_pages):
            page = doc[page_num]
            
            # [수정] 전체 텍스트가 아니라 상단 헤더 영역만 검색
            # 페이지 높이의 상단 일정 비율(예: 15%)만 잘라서 텍스트 추출
            page_rect = page.rect
            header_height = page_rect.height * config.PDF_HEADER_HEIGHT_RATIO
            header_rect = fitz.Rect(page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y0 + header_height)
            
            # 상단 영역 텍스트 추출
            header_text = page.get_text(clip=header_rect)
            
            # 텍스트에서 공백/줄바꿈 제거 후 비교
            text_normalized = header_text.replace(" ", "").replace("\n", "")

            if keyword_normalized in text_normalized:
                target_pages.append(page_num)
                logger.debug(f"  Page {page_num + 1}: 상단 헤더에서 키워드 발견")

        doc.close()

        if target_pages:
            # [수정] 모든 페이지 반환 (상세 내역이 여러 페이지일 수 있음)
            logger.info(f"키워드 '{keyword}' 발견: {len(target_pages)}개 페이지 ([{', '.join([str(p+1) for p in target_pages])}])")
            return target_pages
        else:
            # 키워드 미발견 시 fallback: 처음 N페이지
            fallback_pages = list(range(min(config.PDF_MAX_FALLBACK_PAGES, total_pages)))
            logger.warning(f"키워드 미발견. Fallback: 처음 {len(fallback_pages)}페이지 처리")
            return fallback_pages

    except Exception as e:
        logger.warning(f"페이지 필터링 실패: {e}")
        return None


def extract_filtered_pdf(file_path: Path, target_pages: List[int], output_dir: Path) -> Path:
    """
    원본 PDF에서 타겟 페이지만 추출하여 임시 PDF 생성

    Args:
        file_path: 원본 PDF 경로
        target_pages: 추출할 페이지 인덱스 (0-based)
        output_dir: 임시 파일 저장 디렉토리

    Returns:
        Path: 추출된 PDF 경로
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(file_path))
    new_doc = fitz.open()

    for page_num in target_pages:
        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

    filtered_path = output_dir / f"{file_path.stem}_filtered.pdf"
    new_doc.save(str(filtered_path))

    new_doc.close()
    doc.close()

    logger.info(f"필터링된 PDF 생성: {len(target_pages)}페이지 → {filtered_path.name}")
    return filtered_path


def get_markdown_from_result(res) -> str:
    """
    PaddleOCRVL 결과 객체에서 Markdown 텍스트 추출
    * 개선: JSON 데이터를 직접 파싱하여 Figure/Footer 등으로 분류되어 누락되는 텍스트까지 모두 추출 시도
    """
    # 1. [Strongest Method] JSON 데이터 직접 파싱 (모든 텍스트 강제 추출)
    try:
        data = None
        if hasattr(res, 'json'):
            val = res.json
            if callable(val):
                data = val()
            else:
                data = val
        else:
            try:
                data = dict(res)
            except:
                pass
        
        # data가 문자열(JSON string 또는 파일경로)일 수 있음
        if isinstance(data, str):
            import json
            if data.strip().startswith('{') or data.strip().startswith('['):
                data = json.loads(data)
            elif os.path.exists(data): # 파일 경로인 경우
                with open(data, 'r', encoding='utf-8') as f:
                    data = json.load(f)
        
        # 데이터가 있고, regions 정보를 담고 있는지 확인
        regions = []
        if isinstance(data, dict):
            # [Case 1] res -> parsing_res_list 구조
            if 'res' in data and isinstance(data['res'], dict):
                regions = data['res'].get('parsing_res_list')
            # [Case 2] 바로 리스트가 있는 경우
            if not regions:
                regions = data.get('regions') or data.get('structure_list') or data.get('res')
                
        elif isinstance(data, list):
            regions = data
            
        if regions and isinstance(regions, list) and len(regions) > 0:
            md_lines = []
            
            # [개선] 영역을 Y 좌표 순으로 정렬 (문서 순서 유지)
            # bbox 포맷이 [x1, y1, x2, y2] 또는 [[x1,y1], [x2,y1]...] 형태일 수 있음
            def get_y_coord(region):
                bbox = region.get('bbox')
                if not bbox: return 0
                if isinstance(bbox[0], list): return bbox[0][1] # Polygon
                return bbox[1] # Rect
            
            try:
                regions.sort(key=get_y_coord)
            except:
                pass # 정렬 실패 시 원본 순서 유지

            for r in regions:
                if not isinstance(r, dict): continue
                
                # 키 매핑: block_label / block_content
                r_type = (r.get('type') or r.get('block_label') or 'text').lower()
                content = r.get('block_content') or r.get('res')
                
                # 구버전 호환 (res가 리스트인 경우)
                if isinstance(content, list):
                    texts = [item.get('text') if isinstance(item, dict) else str(item) for item in content]
                    content = " ".join(texts)
                elif isinstance(content, dict) and 'text' in content: # key-value pair 등
                    content = content['text']

                # Table 처리
                if r_type == 'table' and 'html' in r:
                    content = r['html']
                    
                # 텍스트 정제
                if content:
                    content = str(content).strip()
                    if not content: continue
                    
                    # 제목/헤더 처리
                    if r_type in ['title', 'header', 'doc_title', 'section_header']:
                        content = f"## {content}"
                    # 캡션 등 처리
                    elif r_type in ['figure_caption', 'table_caption']:
                        content = f"**{content}**"
                        
                    md_lines.append(content)
            
            # [Fallback] 구조 분석에서 누락된 하단 텍스트(서명 등) 강제 추출
            # 전체 OCR 결과(ocr_res)가 있으면, Y좌표가 맨 아래인 텍스트들을 찾아 추가함
            try:
                # 1. 텍스트 추출용 Raw 데이터 찾기
                raw_ocr_list = []
                if isinstance(data, dict):
                    # case: {'res': {'ocr_res': [...]}}
                    if 'res' in data and isinstance(data['res'], dict):
                        raw_ocr_list = data['res'].get('ocr_res', [])
                    # case: {'ocr_res': [...]}
                    elif 'ocr_res' in data:
                        raw_ocr_list = data.get('ocr_res', [])
                
                # 2. 하단부 텍스트 필터링 (페이지 하단 25% 영역)
                # 좌표 정보가 없으면 전체 텍스트를 대상으로 함 (중복 체크 필수)
                bottom_texts = []
                if raw_ocr_list:
                    # Y좌표 최대값 찾기 (페이지 높이 추정)
                    max_y = 0
                    for item in raw_ocr_list:
                        bbox = item.get('bbox')
                        if bbox:
                            y = bbox[1] if isinstance(bbox[0], (int, float)) else bbox[0][1]
                            max_y = max(max_y, y)
                    
                    threshold_y = max_y * 0.75  # 하단 25%
                    
                    for item in raw_ocr_list:
                        text = item.get('text')
                        bbox = item.get('bbox')
                        if not text: continue
                        
                        # 좌표 확인
                        y = 0
                        if bbox:
                            y = bbox[1] if isinstance(bbox[0], (int, float)) else bbox[0][1]
                        
                        # 하단부에 있고, 기존 md_lines에 포함되지 않은 텍스트만 추가
                        clean_text = text.strip()
                        if y > threshold_y:
                            is_duplicate = False
                            for existing in md_lines:
                                if clean_text in existing:
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                bottom_texts.append((y, clean_text))
                    
                    # Y좌표 순 정렬 후 추가
                    bottom_texts.sort(key=lambda x: x[0])
                    if bottom_texts:
                        md_lines.append("\n\n---\n**[누락된 하단 텍스트 추가]**")
                        for _, txt in bottom_texts:
                            md_lines.append(txt)
                            # logger.debug(f"Recovered bottom text: {txt}")

            except Exception as e:
                # logger.warning(f"Raw OCR fallback failed: {e}")
                pass
            
            if md_lines:
                full_text = "\n\n".join(md_lines)
                return full_text

    except Exception as e:
        logger.debug(f"Manual JSON parsing failed: {e}")


    # [Fallback] 기존 방법 1: 임시 파일로 저장 후 읽기 (가장 안정적)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            res.save_to_markdown(save_path=tmpdir)
            md_files = list(Path(tmpdir).glob("*.md"))
            if md_files:
                return md_files[0].read_text(encoding='utf-8')
        except Exception:
            pass
            
    # [Ultimate Fallback] 객체 전체 텍스트 덤프 후 정규식 추출
    # 구조화 실패 시 텍스트라도 건지기 위함
    try:
        raw_dump = str(res)
        # 'text': '...' 패턴이나 한글 포함된 문자열 추출
        # 1. 딕셔너리 형태의 text 필드 추출
        texts_from_dict = re.findall(r"'text':\s*'([^']*)'", raw_dump)
        
        # 2. 결과가 없으면 한글/영문 혼합된 긴 문자열 검색
        if not texts_from_dict:
            # 특수문자 제외하고 한글+공백이 2글자 이상 연속되는 패턴
            texts_from_dict = re.findall(r"[가-힣\w\s.,()-]{2,}", raw_dump)
            
        if texts_from_dict:
            # 너무 짧거나 의미 없는 것 필터링
            filtered = [t for t in texts_from_dict if len(t.strip()) > 1 and "uni" not in t]
            if filtered:
                return "\n\n".join(filtered)
    except:
        pass

    # 방법 2: markdown 속성 직접 접근 시도
    if hasattr(res, 'markdown'):
        md = res.markdown
        if isinstance(md, str):
            return md
        elif isinstance(md, dict):
            # dict인 경우 텍스트 추출 시도
            return str(md.get('text', '') or md.get('content', '') or '')

    # 방법 3: to_markdown() 메서드 시도
    if hasattr(res, 'to_markdown'):
        md = res.to_markdown()
        if isinstance(md, str):
            return md

    # 방법 4: str 변환 시도
    text = str(res)
    if text and len(text) > 10 and not text.startswith('<'):
        return text

    return ""


def parse_with_paddleocr_vl(
    pipeline,
    file_path: Path,
    output_dir: Path,
    target_pages: List[int] = None,
    save_intermediate: bool = False
) -> str:
    """
    PaddleOCR-VL로 PDF를 파싱하여 Markdown 텍스트 반환 (최적화 버전)

    Args:
        pipeline: PaddleOCRVL 인스턴스
        file_path: 입력 파일 경로
        output_dir: 결과 저장 디렉토리
        target_pages: 처리할 페이지 인덱스 (None이면 전체)
        save_intermediate: 중간 결과 파일 저장 여부

    Returns:
        str: 파싱된 전체 문서의 Markdown 텍스트
    """
    file_stem = safe_stem(file_path.stem)
    doc_save_dir = output_dir / file_stem

    # 페이지 필터링이 있고, PDF 파일인 경우
    process_path = file_path
    if target_pages is not None and file_path.suffix.lower() == '.pdf' and fitz is not None:
        # 필터링된 PDF 생성
        process_path = extract_filtered_pdf(file_path, target_pages, doc_save_dir / "temp")

    # PaddleOCR-VL 실행
    results = pipeline.predict(str(process_path))

    if not results:
        logger.warning(f"No content detected in {file_path.name}")
        return ""

    # 전체 문서 Markdown 버퍼 (메모리에서 처리)
    full_doc_markdown = [f"# {file_path.name} 분석 결과\n"]

    # 페이지별 처리
    for i, res in enumerate(results):
        page_num = i + 1

        # 메모리에서 직접 Markdown 추출 (I/O 최소화)
        page_content = get_markdown_from_result(res)

        if page_content:
            full_doc_markdown.append(f"\n\n---\n## Page {page_num}\n")
            full_doc_markdown.append(page_content)
            logger.debug(f"Page {page_num} content length: {len(page_content)}")
        else:
            logger.warning(f"Page {page_num}: Markdown 추출 실패")


        # 중간 결과 저장 (옵션)
        if save_intermediate:
            page_dir = doc_save_dir / f"page_{page_num:03d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            try:
                res.save_to_json(save_path=str(page_dir))
                res.save_to_markdown(save_path=str(page_dir))
            except Exception as e:
                logger.debug(f"Page {page_num} intermediate save failed: {e}")

    # 통합 Markdown
    combined_markdown = "\n".join(full_doc_markdown)

    # 최종 결과만 저장
    if combined_markdown:
        doc_save_dir.mkdir(parents=True, exist_ok=True)
        merged_path = doc_save_dir / f"{file_stem}_combined.md"
        merged_path.write_text(combined_markdown, encoding='utf-8')
        logger.info(f"Combined MD saved: {merged_path}")

    return combined_markdown


def main():
    parser = argparse.ArgumentParser(
        description="PaddleOCR-VL + LLM 추출 통합 파이프라인"
    )
    parser.add_argument("--input", default="./inputs", help="입력 파일 또는 디렉토리 경로")
    parser.add_argument("--output", default="./outputs", help="결과 저장 경로")
    parser.add_argument("--llm_type", choices=["openai", "ollama"], default="ollama",
                        help="LLM 백엔드 선택 (default: ollama)")
    parser.add_argument("--model", default=None, help="사용할 모델명 (미지정 시 기본값 사용)")
    parser.add_argument("--retry", type=int, default=1, help="LLM 추출 재시도 횟수")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="LLM 추출 건너뛰기 (파싱만 수행)")
    parser.add_argument("--no_filter", action="store_true",
                        help="페이지 필터링 비활성화 (전체 페이지 처리)")
    parser.add_argument("--save_intermediate", action="store_true",
                        help="페이지별 중간 결과 파일 저장")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

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
        logger.error(f"경로를 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    if not targets:
        logger.warning("처리할 파일이 없습니다.")
        return

    logger.info(f"총 {len(targets)}개 파일 처리 예정")

    # PaddleOCR-VL 엔진 초기화
    logger.info("Initializing PaddleOCRVL engine...")
    try:
        pipeline = PaddleOCRVL()
        logger.info("VL Engine initialized successfully.")
    except Exception as e:
        logger.error(f"엔진 초기화 실패: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 결과 요약
    success_count = 0
    fail_count = 0

    # 문서 처리 루프
    for idx, file_path in enumerate(targets):
        logger.info(f"\n[{idx+1}/{len(targets)}] Processing: {file_path.name}")
        start_time = time.perf_counter()

        try:
            # Step 0: 페이지 필터링 (PDF만 해당)
            target_pages = None
            if not args.no_filter and file_path.suffix.lower() == '.pdf':
                logger.info("Step 0: 페이지 필터링 중...")
                target_pages = find_target_pages(file_path)

            # Step 1: PaddleOCR-VL 파싱
            logger.info("Step 1: PaddleOCR-VL 파싱 시작...")
            parsed_markdown = parse_with_paddleocr_vl(
                pipeline,
                file_path,
                output_root / "vl_result",
                target_pages=target_pages,
                save_intermediate=args.save_intermediate
            )

            if not parsed_markdown:
                logger.warning(f"파싱 결과가 없습니다: {file_path.name}")
                fail_count += 1
                continue

            # 파싱된 마크다운 저장
            save_markdown(parsed_markdown, str(file_path))
            logger.info(f"파싱 완료: {len(parsed_markdown)} chars")

            # Step 2: LLM 추출 (옵션)
            if not args.skip_extraction:
                logger.info("Step 2: LLM 추출 시작...")

                extracted_items = None
                last_error = None

                for attempt in range(args.retry):
                    try:
                        extracted_items = extract_data_with_llm(
                            content=parsed_markdown,
                            llm_type=args.llm_type,
                            model_name=args.model
                        )
                        break
                    except Exception as e:
                        last_error = e
                        logger.warning(f"추출 시도 {attempt+1}/{args.retry} 실패: {e}")
                        if attempt < args.retry - 1:
                            time.sleep(2)

                if extracted_items:
                    # 결과 저장
                    result_data = {
                        "source_file": file_path.name,
                        "item_count": len(extracted_items),
                        "items": extracted_items
                    }
                    saved_path = save_result(result_data, str(file_path))
                    logger.info(f"추출 완료: {len(extracted_items)}개 항목 → {saved_path}")
                    success_count += 1
                else:
                    logger.error(f"LLM 추출 실패: 추출된 항목이 0개입니다. (Last error: {last_error})")
                    fail_count += 1
            else:
                logger.info("LLM 추출 건너뜀 (--skip_extraction)")
                success_count += 1

        except Exception as e:
            logger.error(f"{file_path.name} 처리 중 오류: {e}")
            traceback.print_exc()
            fail_count += 1

        elapsed = time.perf_counter() - start_time
        logger.info(f"소요 시간: {elapsed:.2f}s")

    # 최종 요약
    logger.info(f"\n{'='*50}")
    logger.info(f"처리 완료: 성공 {success_count}, 실패 {fail_count}")
    logger.info(f"결과 저장 위치: {output_root}")


if __name__ == "__main__":
    main()
