#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PaddleOCR PP-Structure 기반 PDF 파싱 (표 구조 인식 중심)

- 입력: /data 폴더 내 PDF (기본: *.pdf)
- 처리:
  1) PDF 페이지를 이미지로 렌더링(PyMuPDF)
  2) PP-Structure(layout + table + ocr)로 페이지 구조 분석
  3) 테이블은 HTML(구조 포함)로 저장 + JSON으로 메타/셀 정보 저장
- 출력: /output/paddleocr_vl/<pdf_stem>/
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import fitz  # PyMuPDF
import numpy as np

from paddleocr import PPStructure  # PaddleOCR 문서 구조/표 인식


# ---------------------------
# 유틸
# ---------------------------
def safe_stem(name: str) -> str:
    # 파일/폴더명 안전화
    name = re.sub(r"[^\w\-.가-힣 ]+", "_", name).strip()
    name = name.replace(" ", "_")
    return name[:180] if len(name) > 180 else name


def render_pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 200) -> List[Path]:
    """
    PDF 페이지를 PNG로 렌더링
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))

    zoom = dpi / 72.0  # 기본 72dpi -> 원하는 dpi로 스케일
    mat = fitz.Matrix(zoom, zoom)

    image_paths: List[Path] = []
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_path = out_dir / f"{pdf_path.stem}_p{i+1:03d}.png"
        pix.save(str(img_path))
        image_paths.append(img_path)

    return image_paths


def imread_to_rgb_np(image_path: Path) -> np.ndarray:
    """
    PPStructure 입력은 numpy array (H,W,3) RGB가 가장 무난
    PyMuPDF로 렌더링된 PNG를 fitz 없이 읽을 수도 있지만,
    여기서는 Pillow를 쓰는 대신 OpenCV 없이도 안전하게 읽기 위해
    PyMuPDF로 재로딩하는 방식도 가능함.

    현재 Dockerfile에 pillow가 있으니 Pillow 사용.
    """
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------
# 메인 처리
# ---------------------------
def process_pdf(
    pdf_path: Path,
    out_root: Path,
    dpi: int,
    use_gpu: bool,
    lang: str,
    table_max_len: int = 1000000,
) -> Dict[str, Any]:
    """
    PDF 1개 처리 결과(통합 JSON) 반환
    """
    pdf_stem = safe_stem(pdf_path.stem)
    pdf_out = out_root / pdf_stem
    img_dir = pdf_out / "rendered_pages"
    pages_json_dir = pdf_out / "pages_json"
    tables_html_dir = pdf_out / "tables_html"

    pdf_out.mkdir(parents=True, exist_ok=True)

    # 1) PDF -> page images
    image_paths = render_pdf_to_images(pdf_path, img_dir, dpi=dpi)

    # 2) PPStructure init
    # layout=True: 문서 레이아웃(제목/본문/표/그림 등) 분류
    # table=True: 표 구조 인식 활성화 (HTML/셀 구조)
    # ocr=True: 레이아웃 요소 내부 텍스트 OCR
    # show_log=False: 로그 최소화
    engine = PPStructure(
        lang=lang,
        use_gpu=use_gpu,
        layout=True,
        table=True,
        ocr=True,
        show_log=False,
    )

    all_pages: List[Dict[str, Any]] = []

    for page_idx, img_path in enumerate(image_paths, start=1):
        img_rgb = imread_to_rgb_np(img_path)

        # PPStructure returns: List[dict], 각 dict는 region(표/텍스트/그림 등)
        # 일반적으로 dict keys 예: type, bbox, res(표면 html/셀), text, score ...
        regions = engine(img_rgb)

        page_result: Dict[str, Any] = {
            "page": page_idx,
            "image": str(img_path),
            "regions": [],
            "tables": [],
        }

        # region 정리 + 테이블 HTML 저장
        table_count = 0
        for r in regions:
            r_type = r.get("type")
            bbox = r.get("bbox")

            region_obj: Dict[str, Any] = {
                "type": r_type,
                "bbox": bbox,
            }

            # 텍스트/문단의 경우
            if "text" in r and r.get("text"):
                region_obj["text"] = r.get("text")

            # 표(table)의 경우 res에 html이 들어오는 경우가 많음
            # PaddleOCR 버전/모델에 따라 res 구조가 약간씩 다름
            if r_type in ("table", "Table") or ("res" in r and isinstance(r.get("res"), dict) and "html" in r["res"]):
                table_count += 1

                res = r.get("res") or {}
                html = res.get("html")

                # html이 문자열로 있을 때 저장
                if isinstance(html, str) and html.strip():
                    if len(html) > table_max_len:
                        html = html[:table_max_len] + "\n<!-- truncated -->\n"

                    html_name = f"{pdf_stem}_p{page_idx:03d}_t{table_count:02d}.html"
                    html_path = tables_html_dir / html_name
                    save_text(html_path, html)

                    table_obj = {
                        "page": page_idx,
                        "table_index": table_count,
                        "bbox": bbox,
                        "html_path": str(html_path),
                    }

                    # 추가로 셀 정보가 들어오는 경우도 있음(버전별로 다름)
                    # 안전하게 포함
                    for k in ("cell_bbox", "cells", "structure", "regions"):
                        if k in res:
                            table_obj[k] = res.get(k)

                    page_result["tables"].append(table_obj)
                    region_obj["table_html_path"] = str(html_path)
                else:
                    # html이 없으면 res 전체 저장
                    region_obj["res"] = res

            page_result["regions"].append(region_obj)

        # 페이지 JSON 저장
        page_json_path = pages_json_dir / f"{pdf_stem}_p{page_idx:03d}.json"
        save_json(page_json_path, page_result)

        all_pages.append(page_result)

    # 3) 통합 결과 저장
    merged = {
        "pdf": str(pdf_path),
        "dpi": dpi,
        "use_gpu": use_gpu,
        "lang": lang,
        "num_pages": len(image_paths),
        "pages": all_pages,
        "output_dir": str(pdf_out),
    }

    save_json(pdf_out / "all_pages.json", merged)

    # 4) 사람이 보기 쉬운 MD 요약 생성 (표는 HTML 파일 링크 중심)
    md_lines = []
    md_lines.append(f"# Parsed Result: {pdf_path.name}")
    md_lines.append("")
    md_lines.append(f"- Pages: {len(image_paths)}")
    md_lines.append(f"- DPI: {dpi}")
    md_lines.append(f"- Output: {pdf_out}")
    md_lines.append("")

    for p in all_pages:
        md_lines.append(f"## Page {p['page']}")
        if p["tables"]:
            md_lines.append(f"- Tables: {len(p['tables'])}")
            for t in p["tables"]:
                md_lines.append(f"  - Table {t['table_index']}: {t['html_path']}")
        else:
            md_lines.append("- Tables: 0")

        # 텍스트 타입 region 일부만 간단 출력(너무 길어지는 것 방지)
        texts = [r.get("text", "") for r in p["regions"] if r.get("type") in ("text", "Text", "title", "Title") and r.get("text")]
        if texts:
            preview = "\n".join(texts[:10])
            md_lines.append("")
            md_lines.append("**Text preview (first blocks):**")
            md_lines.append("")
            md_lines.append("```")
            md_lines.append(preview[:2000])
            md_lines.append("```")
        md_lines.append("")

    save_text(pdf_out / "all_pages.md", "\n".join(md_lines))

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="/data", help="PDF input directory")
    parser.add_argument("--out_dir", type=str, default="/output/paddleocr_vl", help="Output directory")
    parser.add_argument("--glob", type=str, default="*.pdf", help="PDF glob pattern")
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI (higher helps table detection; slower)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--lang", type=str, default="korean", help="PaddleOCR language (korean/ch/en etc.)")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(in_dir.glob(args.glob))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found: {in_dir} / pattern={args.glob}")

    results = []
    for pdf_path in pdfs:
        print(f"[PP-Structure] Processing: {pdf_path.name}")
        merged = process_pdf(
            pdf_path=pdf_path,
            out_root=out_dir,
            dpi=args.dpi,
            use_gpu=args.gpu,
            lang=args.lang,
        )
        results.append({
            "pdf": str(pdf_path),
            "output_dir": merged["output_dir"],
            "num_pages": merged["num_pages"],
        })

    save_json(out_dir / "index.json", results)
    print("DONE")
    print("Index:", out_dir / "index.json")


if __name__ == "__main__":
    main()
