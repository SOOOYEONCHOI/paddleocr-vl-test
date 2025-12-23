import os
import glob
import json
import fitz  # PyMuPDF
import cv2
import numpy as np

# PaddleOCR 라이브러리 임포트
# PaddleOCR-VL(2025.10.16 3.3.0 버전)이 설치
try:
    from paddleocr import PaddleOCRVL
    MODEL_TYPE = "VL"
except ImportError:
    print("PaddleOCRVL 클래스를 찾을 수 없습니다. 기본 PaddleOCR로 대체합니다.")
    from paddleocr import PaddleOCR
    MODEL_TYPE = "OCR"

def pdf_to_images(pdf_path):
    """
    PDF 파일을 이미지 리스트(numpy array)로 변환합니다.
    """
    doc = fitz.open(pdf_path)
    images = []
    for page_num, page in enumerate(doc):
        # 해상도 향상을 위해 zoom 설정 (2.0 = 2배 확대)
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        
        # 버퍼를 numpy 배열로 변환
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        # RGBA일 경우 RGB로 변환
        if pix.n == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif pix.n == 3:
            # PyMuPDF는 RGB로 리턴하므로 그대로 사용 가능하나, OpenCV는 BGR을 선호할 수 있음.
            # PaddleOCR은 RGB/BGR 모두 대체로 잘 처리하나 RGB로 통일합니다.
            pass
            
        images.append(img_array)
    return images

def main():
    # 설정: 데이터 및 결과 폴더
    DATA_DIR = "./data"
    OUTPUT_DIR = "./output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 모델 초기화
    print(f"[{MODEL_TYPE}] 모델 초기화 중...")
    if MODEL_TYPE == "VL":
        # PaddleOCR-VL 초기화 (한국어 설정)
        ocr = PaddleOCRVL(lang="korean") 
    else:
        # 기존 PaddleOCR 초기화
        ocr = PaddleOCR(use_angle_cls=True, lang="korean")

    # PDF 파일 검색
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_files:
        print(f"{DATA_DIR} 폴더에 PDF 파일이 없습니다.")
        return

    for pdf_file in pdf_files:
        filename = os.path.basename(pdf_file)
        print(f"\nProcessing: {filename}")
        
        # 1. PDF -> Image 변환
        images = pdf_to_images(pdf_file)
        
        file_result = []
        
        # 2. 각 페이지별 OCR 수행
        for i, img in enumerate(images):
            print(f" - Page {i+1}/{len(images)}...")
            
            # OCR 수행
            # cls=True는 텍스트 방향 분류 사용
            result = ocr.ocr(img, cls=True)
            
            page_info = {
                "page_index": i + 1,
                "result": result
            }
            file_result.append(page_info)
        
        # 3. 결과 저장 (JSON)
        output_filename = os.path.splitext(filename)[0] + "_result.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # JSON 직렬화 불가 객체(예: numpy 타입) 처리
        # PaddleOCR 결과는 보통 리스트 구조이므로 바로 dump 가능할 수 있으나 안전하게 처리
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(file_result, f, ensure_ascii=False, indent=2, default=str)
            
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
