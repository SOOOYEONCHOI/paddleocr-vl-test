#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
[PaddleOCR 3.x 통합 파서] PDF 구조화 (Robust)
- PDF -> 이미지 렌더링(PyMuPDF)
- 엔진 로드: (DOC/VL/OCR) 우선순위 또는 --prefer 지정
- 페이지별 파싱 후 JSON/MD/HTML 저장
  output/<pdf_stem>/
    images/*.png
    tables/*.html (가능한 경우)
    structure_result.json
    parsed_document.md

중요:
- PaddleOCR 3.x는 생성자 인자(예: show_log, use_gpu, use_angle_cls)가 버전/설치조합에 따라 달라질 수 있어
  signature 기반으로 "지원되는 인자만" 주입하도록 처리함.
- 폐쇄망/제한망이면 아래 환경변수 권장:
  export DISABLE_MODEL_SOURCE_CHECK=True
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
    img_array = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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
    """
    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters.keys())
    kwargs = {k: v for k, v in preferred_kwargs.items() if k in accepted}
    return cls(**kwargs)


def build_ocr_engine(lang: str, use_gpu: bool) -> Any:
    """
    PaddleOCR 3.x에서 인자명이 바뀌거나 제거될 수 있어 signature 기반으로 안전하게 생성
    - use_angle_cls deprecated -> use_textline_orientation 우선
    - use_gpu도 환경에 따라 없을 수 있으니 있을 때만 주입
    """
    from paddleocr import PaddleOCR

    base_kwargs = {"lang": lang}

    # GPU 옵션 (지원될 때만)
    # (너 로그에서 use_gpu가 Unknown argument로 뜨는 환경이 있어서 반드시 필터링 필요)
    base_kwargs["use_gpu"] = use_gpu

    # 방향/각도 옵션 (3.x 권장: use_textline_orientation)
    base_kwargs["use_textline_orientation"] = True
    base_kwargs["use_angle_cls"] = True  # 구버전 호환용(있을 때만 들어감)

    # 로그 옵션도 있을 때만
    base_kwargs["show_log"] = False

    return _build_with_signature(PaddleOCR, base_kwargs)


def build_vl_engine() -> Any:
    from paddleocr import PaddleOCRVL
    # VL은 버전에 따라 인자가 달라서 일단 최소 생성
    return PaddleOCRVL()


def build_doc_engine(lang: str, use_gpu: bool) -> Tuple[Any, str]:
    """
    설치 조합에 따라 문서 파서 클래스명이 다를 수 있어 후보를 순차 시도
    성공 시 (engine, class_name) 반환
    """
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
            base_kwargs = {
                "lang": lang,
                "use_gpu": use_gpu,
                "show_log": False,
            }
            engine = _build_with_signature(cls, base_kwargs)
            return engine, name
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"DOC engine init failed. last_err={last_err}")


def get_engine(prefer: str, lang: str, use_gpu: bool) -> Tuple[str, Any, str]:
    """
    returns: (engine_type, engine_obj, engine_name)
      engine_type: DOC | VL | OCR
    """
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

            # OCR
            eng = build_ocr_engine(lang=lang, use_gpu=use_gpu)
            print("   [Init] OCR engine loaded: PaddleOCR")
            return "OCR", eng, "PaddleOCR"

        except Exception as e:
            last_err = e
            print(f"   [Init] {t} init failed: {e}")

    raise RuntimeError(f"Failed to initialize any engine. last_err={last_err}")


# =============================================================================
# Result normalization (best-effort)
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
    """
    PP-Structure 유사 regions에서 text/table(html) 추출 (가능하면)
    """
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

        # text
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


def ocr_out_to_text(ocr_out: Any) -> Tuple[str, Any]:
    lines = []
    if isinstance(ocr_out, list) and len(ocr_out) > 0:
        if isinstance(ocr_out[0], list):
            lines = ocr_out[0]
        else:
            lines = ocr_out

    blocks: List[str] = []
    for line in lines:
        try:
            if len(line) >= 2:
                blocks.append(str(line[1][0]))
        except Exception:
            continue
    return "\n".join(blocks).strip(), lines


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

    for idx, img_path in enumerate(image_paths):
        page_num = idx + 1
        print(f"   [Processing] Page {page_num}/{len(image_paths)}...")

        img = load_image_cv2(img_path)
        page_data: Dict[str, Any] = {"page": page_num, "image_path": str(img_path)}

        markdown_lines.append(f"## Page {page_num}\n")

        try:
            if engine_type in ("DOC", "VL"):
                pred_out = engine.predict(img)
                page_data["predict_raw"] = pred_out

                if is_regions_list(pred_out):
                    page_data["regions"] = pred_out
                    md_lines, tables_meta, table_counter = region_to_text_and_tables(
                        regions=pred_out,
                        table_dir=table_dir,
                        pdf_stem=pdf_stem,
                        page_num=page_num,
                        table_counter=table_counter,
                    )
                    markdown_lines.extend(md_lines)
                    if tables_meta:
                        page_data["tables"] = tables_meta
                else:
                    # 구조가 다르면 raw를 md에 덤프(디버깅용)
                    markdown_lines.append("```json\n")
                    markdown_lines.append(json.dumps(pred_out, ensure_ascii=False, indent=2, default=str))
                    markdown_lines.append("\n```\n")

            else:
                # OCR
                # cls=True는 일부 버전에서 파라미터가 다를 수 있어 안전하게 처리
                try:
                    ocr_out = engine.ocr(img, cls=True)
                except TypeError:
                    ocr_out = engine.ocr(img)

                full_text, raw_lines = ocr_out_to_text(ocr_out)
                page_data["raw_text"] = full_text
                page_data["raw_structure"] = raw_lines
                markdown_lines.append(full_text + "\n")

        except Exception as e:
            page_data["error"] = str(e)
            markdown_lines.append(f"\n[ERROR on page {page_num}] {e}\n")

        full_doc_result.append(page_data)

        # autosave: 중간 실패해도 md/json이 남게 함
        if autosave:
            save_json(save_dir / "structure_result.partial.json", full_doc_result)
            save_text(save_dir / "parsed_document.partial.md", "\n".join(markdown_lines))

    # 최종 저장
    save_json(save_dir / "structure_result.json", full_doc_result)
    save_text(save_dir / "parsed_document.md", "\n".join(markdown_lines))
    print(f"   [Done] Saved to: {save_dir}")


def main():
    debug_versions()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./data/new/")
    parser.add_argument("--output", default="./output/final_result", help="Output root directory")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--lang", default="korean")
    parser.add_argument("--prefer", choices=["doc", "vl", "ocr"], default="doc")

    # 기본 False가 안전: GPU가 실제로 되는 환경에서만 --gpu를 명시적으로 켬
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

    # 엔진 로드
    try:
        engine_type, engine, engine_name = get_engine(prefer=args.prefer, lang=args.lang, use_gpu=args.gpu)
    except Exception as e:
        print(f"   [Error] Failed to initialize engine: {e}")
        return

    for pdf in targets:
        print(f"\nTarget: {pdf.name}")
        start_t = time.perf_counter()
        try:
            process_single_pdf(
                pdf_path=pdf,
                output_root=output_root,
                engine_type=engine_type,
                engine=engine,
                engine_name=engine_name,
                dpi=args.dpi,
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
