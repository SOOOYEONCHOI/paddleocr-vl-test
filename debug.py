#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
[PaddleOCR 3.x 통합 파서] PDF 구조화 (Fixed)
- BGR/RGB 이미지 처리 수정
- OCR 결과 파싱 강화
- 디버깅 출력 추가
"""

import os
import re
import json
import argparse
import time
import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import fitz  # PyMuPDF
import numpy as np
import cv2


# =============================================================================
# Utils
# =============================================================================
def safe_stem(name: str) -> str:
    name = re.sub(r"[^\w\-.가-힣 ]+", "_", name).strip()
    return name[:100]


def render_pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 200) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    image_paths: List[Path] = []
    print(f"   [Rendering] {pdf_path.name} -> {len(doc)} pages (DPI={dpi})")

    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_filename = f"{safe_stem(pdf_path.stem)}_p{i+1:03d}.png"
        img_path = out_dir / img_filename
        pix.save(str(img_path))
        image_paths.append(img_path)

    return image_paths


def load_image_cv2(path: Path) -> np.ndarray:
    """
    Read image and convert to RGB.
    PaddleOCR expects RGB format (not BGR).
    """
    img_array = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    # Convert BGR to RGB for PaddleOCR
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _extract_texts_from_any(obj: Any) -> List[str]:
    """DOC/VL의 predict() 출력 포맷이 설치/버전에 따라 달라서, 텍스트를 최대한 회수."""
    texts: List[str] = []

    def _walk(x: Any):
        if x is None:
            return
        if isinstance(x, str):
            if len(x) <= 2000:
                s = x.strip()
                if s:
                    texts.append(s)
            return
        if isinstance(x, dict):
            for k, v in x.items():
                lk = str(k).lower()
                if lk in ("text", "rec_text", "transcription", "content") and isinstance(v, str):
                    s = v.strip()
                    if s:
                        texts.append(s)
                else:
                    _walk(v)
            return
        if isinstance(x, (list, tuple)):
            for it in x:
                _walk(it)
            return

    _walk(obj)

    uniq: List[str] = []
    seen = set()
    for t in texts:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def save_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def save_json(path: Path, content: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2, default=str)


def debug_versions():
    try:
        import paddleocr
        print(f"   [Debug] PaddleOCR Version: {getattr(paddleocr, '__version__', 'unknown')}")
    except Exception as e:
        print(f"   [Debug] paddleocr import failed: {e}")


# =============================================================================
# Engine builders (3.x friendly)
# =============================================================================
def _build_with_signature(cls, preferred_kwargs: Dict[str, Any]) -> Any:
    """
    cls.__init__ 시그니처를 보고, 지원되는 파라미터만 골라서 넣음.
    PaddleOCR 3.3.0은 파라미터가 대폭 변경되어 signature 기반 필터링 필수.
    """
    try:
        sig = inspect.signature(cls.__init__)
        accepted = set(sig.parameters.keys())
        kwargs = {k: v for k, v in preferred_kwargs.items() if k in accepted}
        print(f"      Accepted params: {list(kwargs.keys())}")
        return cls(**kwargs)
    except Exception as e:
        print(f"      Signature-based init failed: {e}")
        # 최후의 수단: lang만 사용
        if 'lang' in preferred_kwargs:
            print(f"      Trying with lang only...")
            return cls(lang=preferred_kwargs['lang'])
        else:
            print(f"      Trying with no args...")
            return cls()


def build_ocr_engine(lang: str, use_gpu: bool) -> Any:
    """
    PaddleOCR 3.3.0은 파라미터가 대폭 변경됨
    - use_gpu, show_log, use_angle_cls 등 많은 파라미터가 제거됨
    - 최소한의 파라미터만 사용
    """
    from paddleocr import PaddleOCR

    # 3.3.0에서는 거의 모든 파라미터가 제거되어 lang만 사용
    base_kwargs = {"lang": lang}
    
    # GPU는 환경변수나 다른 방식으로 설정될 수 있음
    if use_gpu:
        print("      Note: GPU setting may be ignored in PaddleOCR 3.3.0")

    return _build_with_signature(PaddleOCR, base_kwargs)


def build_vl_engine() -> Any:
    from paddleocr import PaddleOCRVL
    return PaddleOCRVL()


def build_doc_engine(lang: str, use_gpu: bool) -> Tuple[Any, str]:
    """설치 조합에 따라 문서 파서 클래스명이 다를 수 있어 후보를 순차 시도"""
    import paddleocr

    candidates = [
        "PPDocParser",
        "DocParser",
        "PPStructureV3",
        "PPStructure",
    ]

    last_err: Optional[Exception] = None
    for name in candidates:
        try:
            cls = getattr(paddleocr, name)
        except Exception as e:
            last_err = e
            continue

        try:
            base_kwargs = {"lang": lang, "use_gpu": use_gpu}
            engine = _build_with_signature(cls, base_kwargs)
            return engine, name
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"DOC engine init failed. last_err={last_err}")


def get_engine(prefer: str, lang: str, use_gpu: bool) -> Tuple[str, Any, str]:
    """returns: (engine_type, engine_obj, engine_name)"""
    print(f"   [Init] prefer={prefer}, gpu={use_gpu}, lang={lang}")

    if prefer == "doc":
        order = ["DOC", "VL", "OCR"]
    elif prefer == "vl":
        order = ["VL", "DOC", "OCR"]
    else:
        order = ["OCR", "DOC", "VL"]

    last_err: Optional[Exception] = None

    for t in order:
        try:
            if t == "DOC":
                eng, name = build_doc_engine(lang=lang, use_gpu=use_gpu)
                print(f"   [Init] DOC engine loaded: {name}")
                return "DOC", eng, name

            if t == "VL":
                eng = build_vl_engine()
                print("   [Init] VL engine loaded: PaddleOCRVL")
                return "VL", eng, "PaddleOCRVL"

            eng = build_ocr_engine(lang=lang, use_gpu=use_gpu)
            print("   [Init] OCR engine loaded: PaddleOCR")
            return "OCR", eng, "PaddleOCR"

        except Exception as e:
            last_err = e
            print(f"   [Init] {t} init failed: {e}")

    raise RuntimeError(f"Failed to initialize any engine. last_err={last_err}")


# =============================================================================
# Result normalization (enhanced)
# =============================================================================
def is_regions_list(obj: Any) -> bool:
    return isinstance(obj, list) and all(isinstance(x, dict) for x in obj)


def region_to_text_and_tables(
    regions: List[Dict[str, Any]],
    table_dir: Path,
    pdf_stem: str,
    page_num: int,
    table_counter: int,
) -> Tuple[List[str], List[Dict[str, Any]], int]:
    """PP-Structure 유사 regions에서 text/table(html) 추출 (가능하면)"""
    md_lines: List[str] = []
    saved_tables: List[Dict[str, Any]] = []

    def _y(r: Dict[str, Any]) -> float:
        bbox = r.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
            try:
                return float(bbox[1])
            except Exception:
                return 0.0
        return 0.0

    for r in sorted(regions, key=_y):
        r_type = str(r.get("type", "unknown")).lower()
        r_res = r.get("res")

        if r_type == "table":
            html = ""
            if isinstance(r_res, dict):
                html = r_res.get("html", "") or ""
            elif isinstance(r.get("html"), str):
                html = r.get("html", "")

            if html:
                table_dir.mkdir(parents=True, exist_ok=True)
                t_name = f"{pdf_stem}_p{page_num:03d}_t{table_counter:04d}.html"
                save_text(table_dir / t_name, html)

                md_lines.append(f"\n[Table Path: tables/{t_name}]\n")
                md_lines.append(html + "\n")

                saved_tables.append({"page": page_num, "table_file": f"tables/{t_name}"})
                table_counter += 1
            else:
                md_lines.append("\n(Table detected - No HTML)\n")
            continue

        txt = ""
        if isinstance(r_res, dict) and isinstance(r_res.get("text"), str):
            txt = r_res["text"]
        elif isinstance(r_res, list):
            parts = []
            for it in r_res:
                if isinstance(it, dict) and isinstance(it.get("text"), str):
                    parts.append(it["text"])
            txt = " ".join(parts).strip()
        elif isinstance(r.get("text"), str):
            txt = r["text"]

        if not txt:
            continue

        if r_type == "title":
            md_lines.append(f"### {txt}\n")
        elif r_type in ("header", "footer"):
            continue
        else:
            md_lines.append(txt + "\n")

    return md_lines, saved_tables, table_counter


def ocr_out_to_text(ocr_out: Any, page_num: int) -> Tuple[str, Any]:
    """
    OCR 결과를 텍스트로 변환 (강화된 파싱 + 디버깅)
    
    OCR 결과 구조:
    - [[[bbox], (text, conf)], ...] 또는
    - [[[[bbox], (text, conf)], ...]] (페이지 단위 래핑)
    """
    
    # None 체크
    if ocr_out is None:
        print(f"      [Warning] Page {page_num}: OCR returned None")
        return "", []
    
    # 빈 결과 체크
    if isinstance(ocr_out, list) and len(ocr_out) == 0:
        print(f"      [Warning] Page {page_num}: OCR returned empty list")
        return "", []
    
    lines = []
    
    # OCR 결과 구조 파싱
    if isinstance(ocr_out, list):
        # 첫 번째 요소가 리스트면 한 단계 언래핑
        if len(ocr_out) > 0 and isinstance(ocr_out[0], list):
            lines = ocr_out[0]
        else:
            lines = ocr_out
    
    # 빈 라인 체크
    if len(lines) == 0:
        print(f"      [Warning] Page {page_num}: No OCR lines found")
        return "", []
    
    # 텍스트 추출
    blocks: List[str] = []
    failed_count = 0
    
    for idx, line in enumerate(lines):
        try:
            # line 구조: [[[x1,y1], [x2,y2], ...], (text, confidence)]
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                text_info = line[1]
                
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                    text = str(text_info[0]).strip()
                    if text:
                        blocks.append(text)
                else:
                    failed_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            failed_count += 1
            if idx < 3:  # 처음 3개만 출력
                print(f"      [Debug] Page {page_num} Line {idx} parse error: {e}")
    
    result_text = "\n".join(blocks).strip()
    
    # 결과 통계
    if result_text:
        print(f"      [Success] Page {page_num}: Extracted {len(blocks)} lines, {len(result_text)} chars")
        if failed_count > 0:
            print(f"      [Warning] Page {page_num}: {failed_count} lines failed to parse")
    else:
        print(f"      [Warning] Page {page_num}: No text extracted from {len(lines)} OCR lines")
        # 디버깅용: 첫 번째 라인 구조 출력
        if len(lines) > 0:
            print(f"      [Debug] First line structure: {type(lines[0])}")
            print(f"      [Debug] First line sample: {str(lines[0])[:200]}")
    
    return result_text, lines


# =============================================================================
# Main processing
# =============================================================================
def process_single_pdf(
    pdf_path: Path,
    output_root: Path,
    engine_type: str,
    engine: Any,
    engine_name: str,
    dpi: int,
    lang: str,
    use_gpu: bool,
    autosave: bool = True,
):
    pdf_stem = safe_stem(pdf_path.stem)
    save_dir = output_root / pdf_stem
    img_dir = save_dir / "images"
    table_dir = save_dir / "tables"

    image_paths = render_pdf_to_images(pdf_path, img_dir, dpi)

    full_doc_result: List[Dict[str, Any]] = []
    markdown_lines: List[str] = [
        f"# {pdf_path.stem}",
        f"EngineType: {engine_type}",
        f"EngineName: {engine_name}",
        "",
    ]

    table_counter = 0
    total_chars = 0

    # DOC/VL은 predict() 포맷 차이로 텍스트가 비어지는 경우가 있어 OCR 폴백 준비
    ocr_fallback = None
    if engine_type in ("DOC", "VL"):
        try:
            ocr_fallback = build_ocr_engine(lang=lang, use_gpu=use_gpu)
            print(f"   [Init] OCR fallback engine ready")
        except Exception as e:
            print(f"   [Warn] OCR fallback init failed: {e}")

    for idx, img_path in enumerate(image_paths):
        page_num = idx + 1
        print(f"\n   [Page {page_num}/{len(image_paths)}]")

        try:
            img = load_image_cv2(img_path)
            print(f"      Image loaded: {img.shape} (RGB)")
        except Exception as e:
            print(f"      [Error] Failed to load image: {e}")
            markdown_lines.append(f"## Page {page_num}\n[ERROR: Failed to load image]\n")
            continue

        page_data: Dict[str, Any] = {"page": page_num, "image_path": str(img_path)}
        markdown_lines.append(f"## Page {page_num}\n")
        page_text = ""

        try:
            if engine_type in ("DOC", "VL"):
                print(f"      Running {engine_type} predict...")
                pred_out = engine.predict(img)

                if is_regions_list(pred_out):
                    page_data["regions"] = pred_out
                    md_lines, tables_meta, table_counter = region_to_text_and_tables(
                        regions=pred_out,
                        table_dir=table_dir,
                        pdf_stem=pdf_stem,
                        page_num=page_num,
                        table_counter=table_counter,
                    )
                    page_text = "\n".join(md_lines)
                    markdown_lines.extend(md_lines)
                    if tables_meta:
                        page_data["tables"] = tables_meta
                else:
                    # 포맷이 달라도 text 키를 재귀적으로 찾아서 최대한 회수
                    texts = _extract_texts_from_any(pred_out)
                    if texts:
                        page_text = "\n".join(texts)
                        page_data["raw_text"] = page_text
                        markdown_lines.append(page_text + "\n")
                        print(f"      Extracted {len(texts)} text fragments")
                    else:
                        print(f"      [Warning] No text found in {engine_type} result")
                        # OCR 폴백
                        if ocr_fallback is not None:
                            print(f"      Trying OCR fallback...")
                            try:
                                try:
                                    ocr_out = ocr_fallback.ocr(img, cls=True)
                                except TypeError:
                                    ocr_out = ocr_fallback.ocr(img)

                                page_text, raw_lines = ocr_out_to_text(ocr_out, page_num)
                                if page_text:
                                    page_data["raw_text"] = page_text
                                    page_data["raw_structure"] = raw_lines
                                    markdown_lines.append(page_text + "\n")
                            except Exception as e:
                                print(f"      [Error] OCR fallback failed: {e}")
                                page_data["ocr_fallback_error"] = str(e)

            else:
                # OCR 전용
                print(f"      Running OCR...")
                try:
                    ocr_out = engine.ocr(img, cls=True)
                except TypeError:
                    ocr_out = engine.ocr(img)

                page_text, raw_lines = ocr_out_to_text(ocr_out, page_num)
                if page_text:
                    page_data["raw_text"] = page_text
                    page_data["raw_structure"] = raw_lines
                    markdown_lines.append(page_text + "\n")

        except Exception as e:
            page_data["error"] = str(e)
            markdown_lines.append(f"\n[ERROR on page {page_num}] {e}\n")
            print(f"      [Error] {e}")
            import traceback
            traceback.print_exc()

        total_chars += len(page_text)
        full_doc_result.append(page_data)

        if autosave and page_num % 10 == 0:
            save_json(save_dir / "structure_result.partial.json", full_doc_result)
            save_text(save_dir / "parsed_document.partial.md", "\n".join(markdown_lines))

    # 최종 저장
    save_json(save_dir / "structure_result.json", full_doc_result)
    save_text(save_dir / "parsed_document.md", "\n".join(markdown_lines))
    
    print(f"\n   [Done] Saved to: {save_dir}")
    print(f"   [Stats] Total extracted: {total_chars} characters")
    print(f"   [Stats] Average per page: {total_chars/len(image_paths):.0f} chars")


def main():
    debug_versions()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./data/new/")
    parser.add_argument("--output", default="./output/final_result", help="Output root directory")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--lang", default="korean")
    parser.add_argument("--prefer", choices=["doc", "vl", "ocr"], default="ocr")
    parser.add_argument("--gpu", action="store_true", default=False)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output)

    if input_path.is_file():
        targets = [input_path]
    elif input_path.is_dir():
        targets = list(input_path.glob("*.pdf"))
    else:
        targets = []

    if not targets:
        print("No PDF files found.")
        return

    try:
        engine_type, engine, engine_name = get_engine(prefer=args.prefer, lang=args.lang, use_gpu=args.gpu)
    except Exception as e:
        print(f"   [Error] Failed to initialize engine: {e}")
        return

    for pdf in targets:
        print(f"\n{'='*70}")
        print(f"Target: {pdf.name}")
        print(f"{'='*70}")
        start_t = time.perf_counter()
        try:
            process_single_pdf(
                pdf_path=pdf,
                output_root=output_root,
                engine_type=engine_type,
                engine=engine,
                engine_name=engine_name,
                dpi=args.dpi,
                lang=args.lang,
                use_gpu=args.gpu,
                autosave=True,
            )
        except Exception as e:
            print(f"   [Error] Processing failed for {pdf.name}: {e}")
            import traceback
            traceback.print_exc()

        elapsed = time.perf_counter() - start_t
        print(f"   [Time] Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()