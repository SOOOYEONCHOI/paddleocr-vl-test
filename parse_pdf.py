import os
from paddleocr import PPStructure, save_structure_res

# 1. PP-Structure 엔진 초기화
# Dockerfile에서 GPU 환경(cu118/CUDA 12.1)을 설정했으므로 use_gpu=True를 사용합니다.
# layout: 레이아웃 분석(제목, 본문, 표 구분) 활성화
# table: 표 내부 데이터 추출(HTML 변환) 활성화
engine = PPStructure(
    show_log=True, 
    image_orientation=True, 
    layout=True, 
    table=True, 
    lang='korean', # 한국어 모델 사용
    use_gpu=True
)

# 2. 경로 설정 (docker-compose.yml의 볼륨 설정 기준)
input_pdf_path = './data/2023 새로운 위험성평가 안내서.pdf'
output_dir = './output'

# 3. PDF 분석 실행
# PDF의 모든 페이지를 순차적으로 처리합니다.
result = engine(input_pdf_path)

# 4. 결과 저장 및 출력
for i, page_result in enumerate(result):
    # 각 페이지별 결과를 시각화하여 output 폴더에 저장
    # (이미지 위에 영역 표시 및 텍스트/표 정보 포함)
    save_structure_res(page_result, output_dir, f'page_{i+1}_res')
    
    print(f"\n--- [Page {i+1} 분석 결과] ---")
    for region in page_result:
        # 영역 타입 출력 (text, title, table, figure 등)
        region_type = region['type']
        
        if region_type == 'table':
            # 표 데이터인 경우 추출된 HTML 코드 출력
            print(f"[영역: {region_type}] 표 데이터를 감지했습니다.")
            # print(region['res']['html']) # 주석 해제 시 HTML 소스 출력 가능
        else:
            # 일반 텍스트 영역인 경우 인식된 글자 출력
            text_content = [line['text'] for line in region['res']]
            print(f"[영역: {region_type}] 내용: {' '.join(text_content)[:100]}...")

print(f"\n분석 완료! 상세 결과물은 '{output_dir}' 폴더 확인")