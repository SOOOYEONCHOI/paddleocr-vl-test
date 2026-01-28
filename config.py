import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# ==========================================
# Server Configuration
# ==========================================
# Ollama 서버 주소
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://49.247.45.119:11434")

# ==========================================
# Model Configuration
# ==========================================
# 기본 OpenAI 모델
OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL_DEFAULT", "gpt-4.1")

# 기본 Ollama 모델
# OLLAMA_MODEL_DEFAULT = os.getenv("OLLAMA_MODEL_DEFAULT", "llama3.1:8b")
OLLAMA_MODEL_DEFAULT = os.getenv("OLLAMA_MODEL_DEFAULT", "qwen2.5:14b-instruct")

# ==========================================
# Cost & Currency Configuration
# ==========================================
# 원/달러 환율 (2025.12.29 기준)
EXCHANGE_RATE = float(os.getenv("EXCHANGE_RATE", "1430.0"))

# 가격 정보 (USD per 1M tokens)
PRICING = {
    "gpt-4.1": {"input": 2.00, "output": 8.00},
}

# ==========================================
# Extraction Configuration
# ==========================================
# 표준 항목 리스트 (Context Caching용 고정 데이터)
STANDARD_ITEMS = """
안전·보건관리자 임금 등
안전 시설비 등
보호구 등
안전보건진단비 등
안전보건교육비 등
근로자 건강장해예방비 등
건설재해 예방전문지도기관 기술지도비
본사 전담조직 근로자 임금 등
위험성평가 등에 따른 소요비용
"""

# 후처리 직급/직책 용어 리스트
REMOVE_TERMS = [
    '안전관리자', '현장대리인', '안전', '관리자', '사원', '대리', '과장', '차장', '부장', '팀장', '소장', '주임', '담당', '결재', '승인'
]

# ==========================================
# PDF Processing Configuration
# ==========================================
# 텍스트 추출 최소 문자 수 (이하일 경우 OCR 모드)
PDF_CHAR_THRESHOLD = 50

# 한글 문자 최소 비율 (미만일 경우 OCR 모드)
PDF_MIN_KOREAN_RATIO = 0.3

# /uniXXXX 패턴 최대 허용 개수 (초과 시 OCR 모드)
PDF_MAX_UNI_PATTERN_COUNT = 10

# 페이지 상단 키워드 검색 영역 비율
PDF_HEADER_HEIGHT_RATIO = 0.15

# 키워드 미발견 시 처리할 최대 페이지 수
PDF_MAX_FALLBACK_PAGES = 3

# 페이지 필터링 키워드
PDF_TARGET_KEYWORD = "산업안전보건관리비사용내역서"
