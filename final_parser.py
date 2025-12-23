#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
[최종 통합 파서] PaddleOCR 기반 PDF 구조화 도구 (Robust Version)
- 핵심 모델: PaddleOCR-VL (가능한 경우) 또는 PP-Structure 또는 기본 PaddleOCR
- 기능:
  1. PDF를 고해상도 이미지로 변환 (PyMuPDF)
  2. 엔진 로드 (PPStructure 우선, 실패시 VLM/OCR 폴백)
  3. 문서 구조 분석 및 저장
"""

import os
import re
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List

import fitz  # PyMuPDF
import numpy as np
import cv2

# =============================================================================
# 1. 엔진 임포트 및 타입 결정
# =============================================================================
ENGINE_TYPE = None

# 1순위: PPStructureV3 (3.3.0 이상 권장, 구조화/테이블 최적화)
try:
    from paddleocr import PPStructureV2 # V3가 별도 클래스가 아닌 파라미터로 제어될 수 있으나, 명시적 클래스 확인
except ImportError:
    pass

# 일반적인 PPStructure 임포트 (V2/V3 파라미터 제어)
try:
    from paddleocr import PPStructure, save_structure_res
    ENGINE_TYPE = "STRUCTURE"
except ImportError as e1:
    print(f"   [Debug] 'paddleocr.PPStructure' import failed: {e1}")
    try:
        from paddleocr.ppstructure.predict_system import PPStructure
        from paddleocr.ppstructure.utility import save_structure_res
        ENGINE_TYPE = "STRUCTURE"
    except ImportError as e2:
        print(f"   [Debug] 'paddleocr.ppstructure.predict_system.PPStructure' import failed: {e2}")
        pass

# 2순위: PaddleOCRVL (2025 VL 모델)
if not ENGINE_TYPE:
    try:
        from paddleocr import PaddleOCRVL
        ENGINE_TYPE = "VL"
    except ImportError:
        pass

# 3순위: 기본 PaddleOCR (텍스트 전용)
if not ENGINE_TYPE:
    try:
        from paddleocr import PaddleOCR
        ENGINE_TYPE = "OCR"
    except ImportError:
        pass

# 디버깅: 모듈 정보 출력
try:
    import paddleocr
    print(f"   [Debug] PaddleOCR Version: {getattr(paddleocr, '__version__', 'unknown')}")
except:
    pass

if not ENGINE_TYPE:
    print("   [Critical] No PaddleOCR class could be imported. Please check installation.")
    # 실행 지속을 위해 일단 OCR로 설정하되, get_engine에서 실패할 것임
    ENGINE_TYPE = "OCR" 

# ---------------------------
# 유틸리티 함수
# ---------------------------
def safe_stem(name: str) -> str:
    name = re.sub(r"[^\w\-.가-힣 ]+", "_", name).strip()
    return name[:100]

def render_pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 200) -> List[Path]:
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    image_paths = []
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
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def save_json(path: Path, content: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2, default=str)

# ---------------------------
# 엔진 로드
# ---------------------------
def get_engine(use_gpu: bool = True, lang: str = 'korean'):
    print(f"   [Init] Selected Engine Mode: {ENGINE_TYPE} (GPU={use_gpu}, Lang={lang})")
    
    if ENGINE_TYPE == "STRUCTURE":
        print(f"   [Init] Initializing PP-Structure (Logic: {ENGINE_TYPE})...")
        # note: layout=True activates layout analysis (often V3/V2 based)
        # note: table=True activates table recognition (SLANet etc.)
        try:
            return PPStructure(
                show_log=False, 
                image_orientation=True, 
                layout=True, 
                table=True, 
                ocr=True, 
                recovery=True,
                lang=lang, 
                use_gpu=use_gpu
                # structure_version='PP-StructureV2' # 필요한 경우 명시
            )
        except NameError: 
            # PPStructure 클래스가 없는데 ENGINE_TYPE이 STRUCTURE인 경우 폴백
            from paddleocr.ppstructure.predict_system import PPStructure
            return PPStructure(show_log=False, image_orientation=True, layout=True, table=True, ocr=True, recovery=True, lang=lang, use_gpu=use_gpu)

    elif ENGINE_TYPE == "VL":
        from paddleocr import PaddleOCRVL
        # PaddleOCRVL 초기화 인자 최소화 (버전별 상이함 대응)
        # 필요한 경우 config 등을 통해 설정
        return PaddleOCRVL()
        
    else: # OCR
        from paddleocr import PaddleOCR
        return PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False)

# ---------------------------
# 파싱 프로세스
# ---------------------------
def process_single_pdf(pdf_path: Path, output_root: Path, engine, dpi: int):
    pdf_stem = safe_stem(pdf_path.stem)
    save_dir = output_root / pdf_stem
    img_dir = save_dir / "images"
    table_dir = save_dir / "tables"
    
    # 이미지 렌더링
    image_paths = render_pdf_to_images(pdf_path, img_dir, dpi)
    
    full_doc_result = []
    markdown_lines = [f"# {pdf_path.stem}", f"Engine: {ENGINE_TYPE}", ""]
    
    for idx, img_path in enumerate(image_paths):
        page_num = idx + 1
        print(f"   [Processing] Page {page_num}/{len(image_paths)}...")
        
        img = load_image_cv2(img_path)
        page_data = {"page": page_num, "image_path": str(img_path)}
        markdown_lines.append(f"## Page {page_num}\n")
        
        # 엔진별 실행 및 결과 정규화
        if ENGINE_TYPE == "STRUCTURE":
            # PPStructure: returns list of dicts (regions)
            res = engine(img)
            # 결과가 None이거나 비어있을 수 있음
            if not res: res = []
            
            # Y좌표 정렬
            res.sort(key=lambda x: x['bbox'][1])
            
            page_data["regions"] = res
            
            for r in res:
                r_type = r.get('type', 'unknown').lower()
                r_res = r.get('res', {})
                
                if r_type == 'table':
                    html = r_res.get('html', '') if isinstance(r_res, dict) else ''
                    if html:
                        if not table_dir.exists(): table_dir.mkdir(parents=True)
                        t_name = f"{pdf_stem}_p{page_num:03d}_t{len(full_doc_result)}.html"
                        save_text(table_dir / t_name, html)
                        markdown_lines.append(f"\n[Table Path: tables/{t_name}]\n")
                        markdown_lines.append(html)
                    else:
                        markdown_lines.append("\n(Table detected - No HTML)\n")
                
                else: 
                    # 텍스트, 타이틀 등
                    txt = ""
                    if isinstance(r_res, dict): txt = r_res.get('text', '')
                    elif isinstance(r_res, list): txt = " ".join([line.get('text', '') for line in r_res])
                    
                    if r_type == 'title': markdown_lines.append(f"### {txt}\n")
                    elif r_type not in ['header', 'footer']: markdown_lines.append(f"{txt}\n")

        else:
            # OCR / VL: returns [ [ [box, (text, score)], ... ] ]
            # PaddleOCR의 ocr 결과는 [ [라인1], [라인2], ... ] 형태 (페이지가 1개일 때)
            # 그러나 입력이 이미지 하나면 return val[0]이 결과 리스트임
            res = engine.ocr(img, cls=True)
            
            # 결과 구조 정규화
            lines = []
            if res and isinstance(res, list) and len(res) > 0:
                lines = res[0] # 첫번째 이미지 결과
            
            # 텍스트 추출
            page_text_blocks = []
            if lines:
                for line in lines:
                    # line structure: [ [[x,y],...], ("text", score) ]
                    if len(line) >= 2:
                        txt = line[1][0]
                        page_text_blocks.append(txt)
            
            full_text = "\n".join(page_text_blocks)
            markdown_lines.append(full_text + "\n")
            page_data["raw_text"] = full_text
            page_data["raw_structure"] = lines

        full_doc_result.append(page_data)
    
    # 결과 저장
    save_json(save_dir / "structure_result.json", full_doc_result)
    save_text(save_dir / "parsed_document.md", "\n".join(markdown_lines))
    print(f"   [Done] Saved to: {save_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./data", help="Input directory or PDF file")
    parser.add_argument("--output", default="./output/final_result", help="Output root directory")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--lang", default="korean")
    parser.add_argument("--gpu", action='store_true', default=True)
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_root = Path(args.output)
    
    if input_path.is_file(): targets = [input_path]
    elif input_path.is_dir(): targets = list(input_path.glob("*.pdf"))
    else: targets = []
    
    if not targets:
        print("No PDF files found.")
        return

    # 엔진 로드
    try:
        engine = get_engine(args.gpu, args.lang)
    except Exception as e:
        print(f"   [Error] Failed to initialize engine: {e}")
        return

    for pdf in targets:
        print(f"\nTarget: {pdf.name}")
        start_t = time.perf_counter()
        try:
            process_single_pdf(pdf, output_root, engine, args.dpi)
        except Exception as e:
            print(f"   [Error] Processing failed for {pdf.name}: {e}")
            import traceback
            traceback.print_exc()
            
        elapsed = time.perf_counter() - start_t
        print(f"   [Time] Elapsed: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
